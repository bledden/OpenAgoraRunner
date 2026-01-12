#!/usr/bin/env python3
"""
OpenAgora Multi-Agent Runner

A lightweight, standalone runner that polls the OpenAgora marketplace API
and executes jobs using multiple AI agents.

Each agent has:
- Weighted keywords for job matching
- Capability scores
- Its own LLM for job execution

Required env vars:
- BAZAAR_API_URL: The OpenAgora API URL
- FIREWORKS_API_KEY: For LLM inference

Optional:
- AGENT_POLL_INTERVAL: Seconds between polls (default: 30)
"""

import os
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
import httpx
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ]
)
logger = structlog.get_logger()

# Configuration
API_URL = os.getenv("BAZAAR_API_URL", "https://open-agora-production.up.railway.app")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
POLL_INTERVAL = int(os.getenv("AGENT_POLL_INTERVAL", "30"))


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_id: str
    name: str
    description: str
    model: str
    keywords: dict[str, float]  # keyword -> weight (0-1)
    capabilities: dict[str, float]  # capability -> score (0-1)
    base_rate_usd: float = 0.02
    jobs_bid_on: set = field(default_factory=set)


# Built-in agents
AGENTS = [
    AgentConfig(
        agent_id="agent_24a17a8f",
        name="SchemaArchitect",
        description="Expert in database schema design, API architecture, and data modeling",
        model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        keywords={
            "schema": 1.0, "database": 1.0, "api": 1.0, "graphql": 1.0,
            "postgresql": 0.9, "mysql": 0.9, "mongodb": 0.9,
            "data model": 0.9, "erd": 0.8, "normalization": 0.8,
            "rest": 0.8, "openapi": 0.8, "swagger": 0.8,
        },
        capabilities={"api_design": 0.95, "database_design": 0.95, "data_modeling": 0.9},
    ),
    AgentConfig(
        agent_id="agent_7b3f9e2c",
        name="AnomalyHunter",
        description="Specialist in anomaly detection, monitoring, and root cause analysis",
        model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        keywords={
            "anomaly": 1.0, "detection": 1.0, "monitoring": 1.0,
            "alert": 0.9, "metric": 0.9, "observability": 0.9,
            "root cause": 0.9, "incident": 0.8, "sre": 0.8,
            "prometheus": 0.8, "grafana": 0.8, "datadog": 0.8,
        },
        capabilities={"anomaly_detection": 0.95, "monitoring": 0.9, "root_cause_analysis": 0.9},
    ),
    AgentConfig(
        agent_id="agent_5d8c1a4e",
        name="CodeReviewer",
        description="Expert code reviewer focusing on security, performance, and best practices",
        model="accounts/fireworks/models/llama-v3p3-70b-instruct",
        keywords={
            "code review": 1.0, "review": 0.9, "security": 0.9,
            "vulnerability": 0.9, "best practice": 0.8, "refactor": 0.8,
            "performance": 0.8, "optimization": 0.8, "testing": 0.8,
            "lint": 0.7, "static analysis": 0.8,
        },
        capabilities={"code_review": 0.95, "security_review": 0.9, "testing": 0.85},
    ),
]


class OpenAgoraClient:
    """HTTP client for the OpenAgora API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=60.0)

    async def close(self):
        await self.client.aclose()

    async def get_open_jobs(self) -> list[dict]:
        """Get all open jobs from the marketplace."""
        try:
            resp = await self.client.get(f"{self.base_url}/api/jobs", params={"status": "open"})
            resp.raise_for_status()
            data = resp.json()
            return data.get("jobs", data) if isinstance(data, dict) else data
        except Exception as e:
            logger.error("failed_to_get_jobs", error=str(e))
            return []

    async def send_heartbeat(self, agent_id: str) -> bool:
        """Send heartbeat for an agent."""
        try:
            resp = await self.client.post(f"{self.base_url}/api/agents/{agent_id}/heartbeat")
            return resp.status_code == 200
        except Exception as e:
            logger.warning("heartbeat_failed", agent_id=agent_id, error=str(e))
            return False

    async def submit_bid(
        self,
        job_id: str,
        agent_id: str,
        price_usd: float,
        confidence: float,
        approach: str,
    ) -> dict | None:
        """Submit a bid on a job."""
        try:
            resp = await self.client.post(
                f"{self.base_url}/api/jobs/{job_id}/bids",
                json={
                    "agent_id": agent_id,
                    "price_usd": price_usd,
                    "estimated_time_seconds": 60,
                    "confidence": confidence,
                    "approach": approach,
                },
            )
            if resp.status_code in (200, 201):
                return resp.json()
            else:
                logger.warning("bid_rejected", status=resp.status_code, body=resp.text[:200])
                return None
        except Exception as e:
            logger.error("bid_failed", job_id=job_id, error=str(e))
            return None

    async def get_assigned_jobs(self, agent_id: str) -> list[dict]:
        """Get jobs assigned to an agent."""
        try:
            resp = await self.client.get(
                f"{self.base_url}/api/jobs",
                params={"assigned_agent_id": agent_id, "status": "assigned"},
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("jobs", data) if isinstance(data, dict) else data
        except Exception as e:
            logger.error("failed_to_get_assigned_jobs", agent_id=agent_id, error=str(e))
            return []

    async def submit_result(self, job_id: str, result: dict) -> bool:
        """Submit job execution result."""
        try:
            resp = await self.client.post(
                f"{self.base_url}/api/jobs/{job_id}/execute",
                json=result,
            )
            return resp.status_code in (200, 201)
        except Exception as e:
            logger.error("result_submit_failed", job_id=job_id, error=str(e))
            return False


class JobMatcher:
    """Matches jobs to agents based on weighted keywords."""

    @staticmethod
    def calculate_match_score(job: dict, agent: AgentConfig) -> float:
        """Calculate how well a job matches an agent's keywords."""
        job_text = f"{job.get('title', '')} {job.get('description', '')}".lower()

        total_weight = 0.0
        matched_weight = 0.0

        for keyword, weight in agent.keywords.items():
            total_weight += weight
            if keyword.lower() in job_text:
                matched_weight += weight

        if total_weight == 0:
            return 0.0

        return matched_weight / total_weight

    @staticmethod
    def meets_capability_requirements(job: dict, agent: AgentConfig) -> bool:
        """Check if agent meets the job's required capabilities."""
        required_caps = job.get("required_capabilities", [])
        min_score = job.get("min_capability_score", 0.7)

        for cap in required_caps:
            if agent.capabilities.get(cap, 0) < min_score:
                return False
        return True


class LLMExecutor:
    """Executes jobs using Fireworks LLM."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=120.0)

    async def close(self):
        await self.client.aclose()

    async def execute(self, job: dict, agent: AgentConfig) -> dict:
        """Execute a job using the agent's LLM."""
        prompt = f"""You are {agent.name}, an AI agent specializing in: {agent.description}

Job Title: {job.get('title', 'Untitled')}
Job Description: {job.get('description', 'No description')}

Please complete this task. Provide a detailed, high-quality response."""

        try:
            resp = await self.client.post(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": agent.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                    "temperature": 0.7,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            output = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)

            return {
                "success": True,
                "output": output,
                "tokens_used": tokens,
                "model": agent.model,
                "executed_by": agent.agent_id,
            }
        except Exception as e:
            logger.error("llm_execution_failed", agent=agent.name, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "executed_by": agent.agent_id,
            }


class MultiAgentRunner:
    """Main runner that manages multiple agents."""

    def __init__(self):
        self.api = OpenAgoraClient(API_URL)
        self.llm = LLMExecutor(FIREWORKS_API_KEY) if FIREWORKS_API_KEY else None
        self.agents = {a.agent_id: a for a in AGENTS}
        self.running = False

    async def start(self):
        """Start the runner."""
        if not FIREWORKS_API_KEY:
            logger.error("FIREWORKS_API_KEY not set")
            return

        logger.info(
            "starting_multi_runner",
            api_url=API_URL,
            agents=[a.name for a in AGENTS],
            poll_interval=POLL_INTERVAL,
        )

        self.running = True

        while self.running:
            try:
                await self._poll_cycle()
            except Exception as e:
                logger.error("poll_cycle_error", error=str(e))

            await asyncio.sleep(POLL_INTERVAL)

    async def stop(self):
        """Stop the runner."""
        self.running = False
        await self.api.close()
        if self.llm:
            await self.llm.close()

    async def _poll_cycle(self):
        """Single poll cycle: heartbeats, check jobs, bid, execute."""
        # Send heartbeats
        for agent in self.agents.values():
            await self.api.send_heartbeat(agent.agent_id)

        # Get open jobs
        jobs = await self.api.get_open_jobs()
        logger.info("poll_cycle", open_jobs=len(jobs))

        # Try to bid on matching jobs
        for job in jobs:
            job_id = job.get("job_id") or job.get("_id")
            if not job_id:
                continue

            for agent in self.agents.values():
                # Skip if already bid
                if job_id in agent.jobs_bid_on:
                    continue

                # Check capability requirements
                if not JobMatcher.meets_capability_requirements(job, agent):
                    agent.jobs_bid_on.add(job_id)  # Don't check again
                    continue

                # Calculate match score
                score = JobMatcher.calculate_match_score(job, agent)
                if score < 0.3:  # Minimum threshold
                    continue

                # Submit bid
                budget = job.get("budget_usd", 0.10)
                bid_price = min(budget * 0.8, agent.base_rate_usd + (score * 0.05))

                result = await self.api.submit_bid(
                    job_id=job_id,
                    agent_id=agent.agent_id,
                    price_usd=round(bid_price, 4),
                    confidence=round(score, 2),
                    approach=f"{agent.name} will handle this using expertise in {agent.description[:50]}...",
                )

                if result:
                    logger.info(
                        "bid_submitted",
                        job_id=job_id,
                        agent=agent.name,
                        price=bid_price,
                        confidence=score,
                    )

                agent.jobs_bid_on.add(job_id)

        # Execute assigned jobs
        for agent in self.agents.values():
            assigned = await self.api.get_assigned_jobs(agent.agent_id)
            for job in assigned:
                job_id = job.get("job_id") or job.get("_id")
                if not job_id:
                    continue

                logger.info("executing_job", job_id=job_id, agent=agent.name)
                result = await self.llm.execute(job, agent)

                if await self.api.submit_result(job_id, result):
                    logger.info("job_completed", job_id=job_id, success=result.get("success"))
                else:
                    logger.error("job_result_submit_failed", job_id=job_id)


async def main():
    """Entry point."""
    runner = MultiAgentRunner()

    try:
        await runner.start()
    except KeyboardInterrupt:
        logger.info("shutting_down")
    finally:
        await runner.stop()


if __name__ == "__main__":
    asyncio.run(main())
