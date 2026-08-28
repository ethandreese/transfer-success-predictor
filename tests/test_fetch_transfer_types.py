"""
Unit tests for the pure parsing/classification helpers in
scripts/fetch_transfer_types.py - no network access, just the string
handling that turns a transfermarkt fee string into paid/free/loan/unknown
and a club href into a numeric id.
"""
import pytest

from scripts.fetch_transfer_types import classify_fee, club_id_from_href


@pytest.mark.parametrize("fee_raw,expected", [
    ("loan transfer", "loan"),
    ("End of loan", "loan"),
    ('Loan fee:<br /><i class="normaler-text">€5.90m</i>', "loan"),
    ("free transfer", "free"),
    ("€85.00m", "paid"),
    ("€250k", "paid"),
    ("-", "unknown"),
    ("?", "unknown"),
    ("", "unknown"),
    (None, "unknown"),
])
def test_classify_fee(fee_raw, expected):
    """Every case here is a real raw fee string pulled from transfermarkt's live transferHistory API (see Jadon Sancho's transfer history)."""
    assert classify_fee(fee_raw) == expected


def test_classify_fee_is_case_insensitive():
    """"LOAN TRANSFER" or "Free Transfer" should classify the same as their lowercase forms - the API's casing isn't guaranteed to be consistent."""
    assert classify_fee("LOAN TRANSFER") == "loan"
    assert classify_fee("Free Transfer") == "free"


def test_club_id_from_href():
    """Real href format: '/dortmund/transfers/verein/16/saison_id/2022' -> club id 16."""
    assert club_id_from_href("/dortmund/transfers/verein/16/saison_id/2022") == 16


def test_club_id_from_href_handles_missing_or_malformed():
    """A "Without Club"/retirement entry has no href with a club id - should return None, not raise."""
    assert club_id_from_href(None) is None
    assert club_id_from_href("") is None
    assert club_id_from_href("/without-club") is None
