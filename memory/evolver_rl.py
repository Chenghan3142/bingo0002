import json
import math
import os
import re
from typing import Dict, List, Any


_STOP_WORDS = {
    "the", "a", "an", "and", "or", "to", "of", "for", "with", "from", "in", "on", "at",
    "is", "are", "was", "were", "be", "by", "as", "this", "that", "it", "we", "you",
    "buy", "sell", "hold", "system", "general", "unknown",
    "的", "了", "和", "是", "在", "及", "与", "或", "对", "中", "将", "为", "后", "前", "上", "下",
}


def _normalize_tokens(text: str) -> List[str]:
    lowered = str(text or "").lower()
    raw = re.findall(r"[a-z0-9_\-\u4e00-\u9fff]+", lowered)
    tokens = [tok for tok in raw if len(tok) >= 2 and tok not in _STOP_WORDS]
    return tokens


def extract_learning_signals(text: str, extras: List[str] = None, max_count: int = 24) -> List[str]:
    pool = _normalize_tokens(text)
    if extras:
        for item in extras:
            pool.extend(_normalize_tokens(item))

    freq: Dict[str, int] = {}
    for tok in pool:
        freq[tok] = freq.get(tok, 0) + 1

    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:max_count]]


def score_tag_overlap(left: List[str], right: List[str]) -> float:
    lset = set(left or [])
    rset = set(right or [])
    if not lset or not rset:
        return 0.0
    inter = len(lset & rset)
    union = len(lset | rset)
    return inter / max(union, 1)


class EvolverSelector:
    """A readable Evolver-style selector: signals -> memory advice -> drift-aware scoring."""

    def __init__(self, state_file: str):
        self.state_file = state_file
        self.signal_graph: Dict[str, Dict[str, float]] = {}
        self.action_stats: Dict[str, Dict[str, float]] = {}
        self.episodes = 0
        self._load_state()

    def _load_state(self):
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.signal_graph = state.get("signal_graph", {})
            self.action_stats = state.get("action_stats", {})
            self.episodes = int(state.get("episodes", 0))
        except Exception:
            self.signal_graph = {}
            self.action_stats = {}
            self.episodes = 0

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        payload = {
            "signal_graph": self.signal_graph,
            "action_stats": self.action_stats,
            "episodes": self.episodes,
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def register_outcome(self, signals: List[str], action: str, pnl_percent: float):
        action = str(action or "HOLD").upper()
        self.episodes += 1

        stat = self.action_stats.setdefault(action, {"wins": 0.0, "losses": 0.0, "sum_pnl": 0.0, "count": 0.0})
        stat["count"] += 1
        stat["sum_pnl"] += float(pnl_percent)
        if pnl_percent > 0:
            stat["wins"] += 1
        elif pnl_percent < 0:
            stat["losses"] += 1

        for sig in signals or []:
            node = self.signal_graph.setdefault(sig, {"wins": 0.0, "losses": 0.0, "sum_pnl": 0.0, "count": 0.0})
            node["count"] += 1
            node["sum_pnl"] += float(pnl_percent)
            if pnl_percent > 0:
                node["wins"] += 1
            elif pnl_percent < 0:
                node["losses"] += 1

        self._save_state()

    def build_memory_advice(self, query_signals: List[str]) -> Dict[str, Any]:
        matched = []
        for sig in query_signals or []:
            node = self.signal_graph.get(sig)
            if node and node.get("count", 0) > 0:
                avg_pnl = node["sum_pnl"] / max(node["count"], 1.0)
                matched.append(avg_pnl)

        signal_bias = sum(matched) / len(matched) if matched else 0.0

        preferred_actions = []
        avoid_actions = []
        for action, stat in self.action_stats.items():
            if stat.get("count", 0) < 3:
                continue
            avg_pnl = stat["sum_pnl"] / max(stat["count"], 1.0)
            if avg_pnl > 0.25:
                preferred_actions.append(action)
            elif avg_pnl < -0.25:
                avoid_actions.append(action)

        drift_intensity = min(1.0, max(0.0, abs(signal_bias) / 2.0))
        drift_mode = "selection:drift" if drift_intensity >= 0.35 else "selection:stable"

        return {
            "signal_bias": signal_bias,
            "preferred_actions": preferred_actions,
            "avoid_actions": avoid_actions,
            "drift_intensity": drift_intensity,
            "selection_path": drift_mode,
        }

    def outcome_score_delta(self, action: str, pnl_percent: float, advice: Dict[str, Any]) -> float:
        action = str(action or "HOLD").upper()
        base = 0.1 if pnl_percent > 0 else (-0.2 if pnl_percent < 0 else 0.0)

        drift = float(advice.get("drift_intensity", 0.0) or 0.0)
        if action in advice.get("preferred_actions", []):
            base += 0.03 * (1.0 + drift)
        if action in advice.get("avoid_actions", []):
            base -= 0.05 * (1.0 + drift)

        return base

    def rerank_candidates(self, candidates: List[Dict[str, Any]], query_signals: List[str]) -> List[Dict[str, Any]]:
        advice = self.build_memory_advice(query_signals)
        preferred = set(advice.get("preferred_actions", []))
        avoided = set(advice.get("avoid_actions", []))
        signal_bias = float(advice.get("signal_bias", 0.0) or 0.0)

        scored = []
        for item in candidates:
            metadata = item.get("metadata", {}) or {}
            content = item.get("content", "")
            base_score = float(metadata.get("score", 1.0) or 1.0)
            action = str(metadata.get("action_taken", "HOLD")).upper()
            doc_signals = metadata.get("learning_signals") or extract_learning_signals(content)

            overlap = score_tag_overlap(doc_signals, query_signals)
            score = base_score + overlap * 0.8

            if action in preferred:
                score += 0.08 + max(0.0, signal_bias) * 0.05
            if action in avoided:
                score -= 0.12 + max(0.0, -signal_bias) * 0.08

            if advice.get("selection_path") == "selection:drift":
                score += 0.05 * (1 if overlap > 0.2 else -1)

            enriched = dict(item)
            enriched["selector_score"] = round(score, 6)
            enriched["selection_path"] = advice.get("selection_path")
            scored.append(enriched)

        scored.sort(key=lambda x: x.get("selector_score", 0.0), reverse=True)
        return scored
