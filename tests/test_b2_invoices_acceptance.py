"""Acceptance check for B-06: the B2 invoice pack (packs/b2_invoices.json).

This proves the CTO's acceptance line independently of the pack author's own
mirror tests in test_task_packs.py: loading the pack through the real
lab.task_packs.load_pack must yield 30–50 tasks, every check must be 'json',
and every expected vat_amount must match the total and rate *as stated in the
invoice text* (8.1% or 2.6%) within Rappen rounding.

Unlike the mirror tests, which trust the expected dict's own `total`/`vat_rate`,
these read the gross amount and the rate straight out of each German invoice
text, so a pack whose expected dict disagreed with the text it ships would fail
here even if it were internally consistent.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab"))

import task_packs  # noqa: E402  (import after sys.path is set)

B2_PACK_PATH = ROOT / "packs" / "b2_invoices.json"
B2_RATES = {8.1, 2.6}

# The single gross amount in a text, written "CHF 1'860.00" or "Fr. 178.00",
# and the MWST rate, written "8.1%" / "2.6%".
_AMOUNT = re.compile(r"(?:CHF|Fr\.?)\s*([\d']+(?:\.\d{2})?)")
_RATE = re.compile(r"(\d+\.\d)\s*%")


def _stated_amount(text: str) -> float:
    found = _AMOUNT.findall(text)
    assert len(found) == 1, f"expected exactly one stated amount, found {found}"
    return float(found[0].replace("'", ""))


def _stated_rate(text: str) -> float:
    found = {float(r) for r in _RATE.findall(text)}
    assert len(found) == 1, f"expected exactly one stated rate, found {found}"
    return found.pop()


@pytest.fixture(scope="module")
def b2_pack():
    return task_packs.load_pack(B2_PACK_PATH)


def test_load_pack_yields_30_to_50_tasks(b2_pack):
    assert 30 <= len(b2_pack.tasks) <= 50


def test_every_check_is_json(b2_pack):
    assert [t.check for t in b2_pack.tasks] == ["json"] * len(b2_pack.tasks)


def test_vat_amount_matches_stated_total_and_rate(b2_pack):
    for t in b2_pack.tasks:
        total = _stated_amount(t.prompt)
        rate = _stated_rate(t.prompt)
        assert rate in B2_RATES, f"{t.task_id}: rate {rate} not in {B2_RATES}"
        # The expected dict must describe the invoice its own text states.
        assert t.expected["total"] == total, f"{t.task_id}: total {t.expected['total']} != stated {total}"
        assert t.expected["vat_rate"] == rate, f"{t.task_id}: rate {t.expected['vat_rate']} != stated {rate}"
        # total is gross incl. MWST, so the contained VAT is total * rate / (100 + rate).
        contained_vat = round(total * rate / (100 + rate), 2)
        assert abs(t.expected["vat_amount"] - contained_vat) <= 0.01, (
            f"{t.task_id}: vat_amount {t.expected['vat_amount']} != {contained_vat}"
        )
