import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Ensures synthetic fixtures exist (generated once, committed or built)."""
    train = FIXTURES / "synthetic_train.csv"
    test = FIXTURES / "synthetic_test.csv"
    if not train.exists() or not test.exists():
        subprocess.run(
            [sys.executable, str(FIXTURES / "generate_fixtures.py"),
             "--samples", "600", "--out", str(FIXTURES)],
            check=True, cwd=REPO,
        )
    return FIXTURES


@pytest.fixture(scope="session")
def synthetic_train(fixtures_dir: Path) -> Path:
    return fixtures_dir / "synthetic_train.csv"


@pytest.fixture(scope="session")
def synthetic_test(fixtures_dir: Path) -> Path:
    return fixtures_dir / "synthetic_test.csv"


@pytest.fixture(scope="session")
def synthetic_ood(fixtures_dir: Path) -> Path:
    return fixtures_dir / "synthetic_ood.csv"