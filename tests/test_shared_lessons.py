"""Cross-business learning: anonymized lessons on the shared shelf (local mirror, Mem0 off)."""
import pytest

import agent
import memory_store
import strategist
from memory_store import MemoryStore

CAFE = {"brand_name": "Sunrise Cafe", "biz_type": "Coffee shop", "audience": "Locals and remote workers"}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_store, "MIRROR_PATH", tmp_path / "memories.json")
    for k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)


def store(user_id):
    m = MemoryStore(user_id)
    m.client = None
    return m


def clear_win(memory, dim="subject_hook", new="question"):
    p = agent.make_plan(agent.NAIVE_DEFAULTS["email"], dim, new, "test")
    results = {"A": {"open_rate": 0.2, "ctr": 0.03, "conversion_rate": 0.01},
               "B": {"open_rate": 0.2, "ctr": 0.04, "conversion_rate": 0.01}}
    out = agent.learn(memory, "email", p, results, "Email #1: Sunrise Cafe launch")
    return p, out


def test_clear_win_is_shared_without_brand_or_numbers():
    cafe = store("sunrise-cafe")
    p, out = clear_win(cafe)
    item = agent.share_lesson(cafe, "email", p, out, CAFE)
    assert item["memory"] == ("For a coffee shop serving locals and remote workers, on email, "
                              "a question subject line beat an urgent subject line.")
    assert "Sunrise" not in item["memory"] and "%" not in item["memory"]
    assert item["metadata"]["source"] != "sunrise-cafe"  # hashed, not the business id


def test_ties_are_not_shared():
    cafe = store("sunrise-cafe")
    p = agent.make_plan(agent.NAIVE_DEFAULTS["email"], "emoji", "no", "t")
    same = {"open_rate": 0.2, "ctr": 0.03, "conversion_rate": 0.01}
    out = agent.learn(cafe, "email", p, {"A": same, "B": same}, "Email #1")
    assert agent.share_lesson(cafe, "email", p, out, CAFE) is None


def test_other_business_recalls_the_lesson_but_not_its_own():
    cafe, bakery = store("sunrise-cafe"), store("golden-crust")
    p, out = clear_win(cafe)
    agent.share_lesson(cafe, "email", p, out, CAFE)
    assert [m["memory"] for m in agent.recall(bakery, "email", "weekend promo")["shared"]] == \
        ["For a coffee shop serving locals and remote workers, on email, "
         "a question subject line beat an urgent subject line."]
    assert agent.recall(cafe, "email", "weekend promo")["shared"] == []
    assert agent.recall(bakery, "instagram", "weekend promo")["shared"] == []


def test_new_business_first_test_builds_on_shared_lesson():
    cafe, bakery = store("sunrise-cafe"), store("golden-crust")
    p, out = clear_win(cafe)
    agent.share_lesson(cafe, "email", p, out, CAFE)
    plan = strategist.suggest(bakery, "email", "Weekend pastry special", {"brand_name": "Golden Crust"}, [], True, 1)
    s = plan["strategy"]
    assert (plan["plan"]["tested_dimension"], plan["plan"]["B"]["subject_hook"]) == ("subject_hook", "question")
    assert any(i.startswith("Another business:") for i in s["insights"])
    assert "won for another coffee shop" in s["rationale"]
    assert s["confidence"] == "medium"


def test_reset_removes_this_business_shared_lessons_only():
    cafe, bakery = store("sunrise-cafe"), store("golden-crust")
    p, out = clear_win(cafe)
    agent.share_lesson(cafe, "email", p, out, CAFE)
    p2, out2 = clear_win(bakery, "offer", "free_trial")
    agent.share_lesson(bakery, "email", p2, out2, {"biz_type": "Bakery"})
    cafe.reset()
    texts = [m["memory"] for m in store("someone-else").search_shared("lessons", "email")]
    assert len(texts) == 1 and texts[0].startswith("For a bakery")


# ---------------- live Mem0 ----------------
import os  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

load_dotenv()
live = pytest.mark.skipif(not os.getenv("MEM0_API_KEY"), reason="MEM0_API_KEY not set")


@live
@pytest.mark.live
def test_live_shared_lesson_reaches_another_business():
    giver, taker = MemoryStore(f"test_giver_{uuid.uuid4().hex[:8]}"), MemoryStore(f"test_taker_{uuid.uuid4().hex[:8]}")
    assert giver.client is not None, giver.last_error
    try:
        p, out = clear_win(giver)
        agent.share_lesson(giver, "email", p, out, {"biz_type": f"Test shop {giver.source}"})
        assert giver.last_error is None
        hits = []
        for _ in range(40):  # Mem0 processes adds asynchronously
            hits = [h for h in taker.search_shared(f"Test shop {giver.source} question subject line", "email", top_k=10)
                    if h["source"] == "mem0" and h["metadata"].get("source") == giver.source]
            if hits:
                break
            time.sleep(3)
        assert hits and "question subject line beat an urgent subject line" in hits[0]["memory"]
        assert giver.search_shared("question subject line", "email") == [] or all(
            h["metadata"].get("source") != giver.source for h in giver.search_shared("question subject line", "email"))
    finally:
        giver.reset()
        taker.reset()
