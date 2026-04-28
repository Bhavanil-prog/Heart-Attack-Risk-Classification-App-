"""Unit tests for requirements.txt - validates all dependencies use pinned versions.

Validates: Requirements 5.1
"""

import os


def get_requirements_lines():
    """Read and parse non-comment, non-empty lines from requirements.txt."""
    req_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
    with open(req_path) as f:
        lines = f.readlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


def test_all_requirements_are_pinned():
    """Every dependency in requirements.txt must use == (pinned version)."""
    lines = get_requirements_lines()
    assert lines, "requirements.txt has no dependency entries"
    unpinned = [line for line in lines if "==" not in line]
    assert unpinned == [], (
        f"The following requirements are not pinned with '==': {unpinned}"
    )
