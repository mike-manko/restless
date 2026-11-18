"""Smoke test: pytest exits with code 5 when it collects no tests, so keep at least one."""

import restless


def test_package_imports() -> None:
    assert callable(restless.main)
