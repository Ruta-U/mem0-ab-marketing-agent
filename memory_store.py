"""Memory layer: Mem0 Platform when MEM0_API_KEY is set, plus a local JSON mirror.

The local mirror keeps the UI instant (Mem0 adds are processed asynchronously)
and lets the demo run offline if the Mem0 key is missing.

Two shelves, as in memory.py:
- user_id  = one business: its brand and its own lessons (private)
- agent_id = the marketing agent: anonymized lessons shared across every business
"""
import hashlib
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
MIRROR_PATH = DATA_DIR / "memories.json"
AGENT_ID = "ab_marketing_agent"  # the shared shelf; memory.py uses the same id
SHARED_OWNER = "__shared__"      # how shared lessons are keyed in the local mirror


class MemoryStore:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.client = None
        self.last_error = None
        api_key = os.getenv("MEM0_API_KEY")
        if api_key:
            try:
                from mem0 import MemoryClient

                self.client = MemoryClient(api_key=api_key)
            except Exception as e:  # keep the demo alive without Mem0
                self.last_error = str(e)

    @property
    def source(self) -> str:
        """A stable, non-reversible tag for this business on the shared shelf."""
        return hashlib.sha1(self.user_id.encode()).hexdigest()[:12]

    @property
    def backend(self) -> str:
        return "Mem0 Platform" if self.client else "Local fallback (set MEM0_API_KEY)"

    # ---------- local mirror ----------
    def _load(self) -> list:
        if MIRROR_PATH.exists():
            return json.loads(MIRROR_PATH.read_text())
        return []

    def _save(self, items: list) -> None:
        MIRROR_PATH.write_text(json.dumps(items, indent=2))

    # ---------- public API ----------
    def add(self, text: str, metadata: dict, infer: bool = False) -> dict:
        """Write a memory. infer=False stores the text verbatim (used for learnings)."""
        item = {
            "id": str(uuid.uuid4()),
            "user_id": self.user_id,
            "memory": text,
            "metadata": metadata,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "synced_to_mem0": False,
        }
        if self.client:
            try:
                self.client.add(
                    [{"role": "user", "content": text}],
                    user_id=self.user_id,
                    metadata=metadata,
                    infer=infer,
                )
                item["synced_to_mem0"] = True
            except Exception as e:
                self.last_error = str(e)
        items = self._load()
        items.append(item)
        self._save(items)
        return item

    def search(self, query: str, filters: dict | None = None, top_k: int = 8) -> list:
        """Retrieve memories relevant to a query. Mem0 semantic search, else keyword match."""
        if self.client:
            try:
                res = self.client.search(
                    query, filters={"user_id": self.user_id}, top_k=top_k
                )
                results = res.get("results", res) if isinstance(res, dict) else res
                out = [
                    {"memory": r.get("memory", ""), "metadata": r.get("metadata") or {},
                     "score": r.get("score"), "source": "mem0"}
                    for r in results
                ]
                if filters:
                    out = [r for r in out if all(r["metadata"].get(k) == v for k, v in filters.items())
                           or not r["metadata"]]
                if out:
                    return out
            except Exception as e:
                self.last_error = str(e)

        words = {w.lower() for w in query.split() if len(w) > 3}
        local = [m for m in self._load() if m["user_id"] == self.user_id]
        if filters:
            local = [m for m in local if all(m["metadata"].get(k) == v for k, v in filters.items())]
        scored = []
        for m in local:
            score = sum(w in m["memory"].lower() for w in words)
            scored.append({"memory": m["memory"], "metadata": m["metadata"],
                           "score": score, "source": "local"})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]

    def all(self, kind: str | None = None) -> list:
        items = [m for m in self._load() if m["user_id"] == self.user_id]
        if kind:
            items = [m for m in items if m["metadata"].get("kind") == kind]
        return items

    # ---------- shared shelf (cross-business lessons) ----------
    def add_shared(self, text: str, metadata: dict) -> dict:
        """Write an anonymized lesson every business can learn from."""
        metadata = {**metadata, "kind": "shared_lesson", "source": self.source}
        item = {"id": str(uuid.uuid4()), "user_id": SHARED_OWNER, "memory": text, "metadata": metadata,
                "created_at": datetime.now().isoformat(timespec="seconds"), "synced_to_mem0": False}
        if self.client:
            try:
                self.client.add([{"role": "user", "content": text}], agent_id=AGENT_ID,
                                metadata=metadata, infer=False)
                item["synced_to_mem0"] = True
            except Exception as e:
                self.last_error = str(e)
        items = self._load()
        items.append(item)
        self._save(items)
        return item

    def search_shared(self, query: str, channel: str | None = None, top_k: int = 5) -> list:
        """Lessons other businesses shared (never this business's own)."""
        def keep(md):
            return md.get("kind") == "shared_lesson" and md.get("source") != self.source and \
                (channel is None or md.get("channel") == channel)
        if self.client:
            try:
                res = self.client.search(query, filters={"agent_id": AGENT_ID}, top_k=top_k * 3)
                results = res.get("results", res) if isinstance(res, dict) else res
                out = [{"memory": r.get("memory", ""), "metadata": r.get("metadata") or {},
                        "score": r.get("score"), "source": "mem0"} for r in results]
                out = [r for r in out if keep(r["metadata"])][:top_k]
                if out:
                    return out
            except Exception as e:
                self.last_error = str(e)
        shared = [m for m in self._load() if m["user_id"] == SHARED_OWNER and keep(m["metadata"])]
        return [{"memory": m["memory"], "metadata": m["metadata"], "score": None, "source": "local"}
                for m in reversed(shared)][:top_k]

    def reset(self) -> None:
        items = [m for m in self._load() if m["user_id"] != self.user_id and not (
            m["user_id"] == SHARED_OWNER and m["metadata"].get("source") == self.source)]
        self._save(items)
        if self.client:
            # Delete per record id: delete_all(filters={"user_id": ...}) is rejected by the
            # Mem0 API (400), and delete_all's async sweep can also wipe adds made right after.
            try:
                filters = {"AND": [{"user_id": self.user_id}]}
                while True:
                    batch = self.client.get_all(filters=filters, page_size=50)["results"]
                    if not batch:
                        break
                    for m in batch:
                        self.client.delete(memory_id=m["id"])
                # and this business's contributions to the shared shelf
                shared = self.client.get_all(filters={"AND": [{"agent_id": AGENT_ID}]}, page_size=100)["results"]
                for m in shared:
                    if (m.get("metadata") or {}).get("source") == self.source:
                        self.client.delete(memory_id=m["id"])
            except Exception as e:
                self.last_error = str(e)
