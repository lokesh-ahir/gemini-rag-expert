"""
Cost Tracker - Tracks model usage and costs for billing/analytics.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from .model_router import ModelType, MODEL_CONFIGS


@dataclass
class UsageRecord:
    """Record of a single model usage"""
    model_type: ModelType
    intent: str
    token_count: int
    cost: float
    timestamp: datetime
    design_id: Optional[str] = None
    workspace_id: Optional[str] = None


@dataclass
class CostReport:
    """Cost report for a period"""
    period_start: datetime
    period_end: datetime
    total_cost: float
    total_tokens: int
    usage_by_model: Dict[str, int] = field(default_factory=dict)
    usage_by_intent: Dict[str, int] = field(default_factory=dict)


class CostTracker:
    """
    Tracks model usage and calculates costs.
    
    Usage:
        tracker = CostTracker()
        tracker.record_usage(ModelType.GEMINI_FLASH, "general_query", 1000)
        report = tracker.get_report(last_24_hours=True)
    """
    
    def __init__(self):
        self.usage_records: List[UsageRecord] = []
        self._daily_usage: Dict[str, List[UsageRecord]] = defaultdict(list)
    
    def record_usage(
        self,
        model_type: ModelType,
        intent: str,
        token_count: int,
        design_id: Optional[str] = None,
        workspace_id: Optional[str] = None
    ) -> UsageRecord:
        """
        Record a model usage.
        
        Args:
            model_type: Which model was used
            intent: The intent that triggered this usage
            token_count: Number of tokens processed
            design_id: Optional design ID for attribution
            workspace_id: Optional workspace ID for attribution
        
        Returns:
            The created usage record
        """
        config = MODEL_CONFIGS[model_type]
        cost = config.cost_per_token * token_count
        
        record = UsageRecord(
            model_type=model_type,
            intent=intent,
            token_count=token_count,
            cost=cost,
            timestamp=datetime.utcnow(),
            design_id=design_id,
            workspace_id=workspace_id
        )
        
        self.usage_records.append(record)
        
        # Also store by date for daily reports
        date_key = datetime.utcnow().strftime("%Y-%m-%d")
        self._daily_usage[date_key].append(record)
        
        return record
    
    def get_total_cost(self) -> float:
        """Get total cost across all usage"""
        return sum(r.cost for r in self.usage_records)
    
    def get_total_tokens(self) -> int:
        """Get total tokens across all usage"""
        return sum(r.token_count for r in self.usage_records)
    
    def get_report(
        self,
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None,
        workspace_id: Optional[str] = None
    ) -> CostReport:
        """
        Generate a cost report for a period.
        
        Args:
            period_start: Start of period (default: 24 hours ago)
            period_end: End of period (default: now)
            workspace_id: Filter by workspace
        
        Returns:
            CostReport with aggregated stats
        """
        period_start = period_start or (datetime.utcnow() - timedelta(days=1))
        period_end = period_end or datetime.utcnow()
        
        filtered = [
            r for r in self.usage_records
            if period_start <= r.timestamp <= period_end
            and (workspace_id is None or r.workspace_id == workspace_id)
        ]
        
        total_cost = sum(r.cost for r in filtered)
        total_tokens = sum(r.token_count for r in filtered)
        
        usage_by_model: Dict[str, int] = defaultdict(int)
        usage_by_intent: Dict[str, int] = defaultdict(int)
        
        for r in filtered:
            usage_by_model[r.model_type.value] += r.token_count
            usage_by_intent[r.intent] += r.token_count
        
        return CostReport(
            period_start=period_start,
            period_end=period_end,
            total_cost=total_cost,
            total_tokens=total_tokens,
            usage_by_model=dict(usage_by_model),
            usage_by_intent=dict(usage_by_intent)
        )
    
    def get_daily_report(self, date: Optional[str] = None) -> CostReport:
        """Get report for a specific day"""
        date_key = date or datetime.utcnow().strftime("%Y-%m-%d")
        records = self._daily_usage.get(date_key, [])
        
        if not records:
            return CostReport(
                period_start=datetime.strptime(date_key, "%Y-%m-%d"),
                period_end=datetime.strptime(date_key, "%Y-%m-%d") + timedelta(days=1),
                total_cost=0.0,
                total_tokens=0
            )
        
        return self.get_report(
            period_start=min(r.timestamp for r in records),
            period_end=max(r.timestamp for r in records)
        )
    
    def get_workspace_costs(self) -> Dict[str, float]:
        """Get total cost by workspace"""
        costs: Dict[str, float] = defaultdict(float)
        
        for r in self.usage_records:
            if r.workspace_id:
                costs[r.workspace_id] += r.cost
        
        return dict(costs)
    
    def reset(self) -> None:
        """Clear all usage data"""
        self.usage_records.clear()
        self._daily_usage.clear()


# Singleton instance
_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the singleton cost tracker"""
    global _tracker
    if _tracker is None:
        _tracker = CostTracker()
    return _tracker


def reset_cost_tracker() -> None:
    """Reset the singleton"""
    global _tracker
    _tracker = None
