import io
from importlib import metadata

import pytest
from rich.console import Console

from restless.banner import _TITLE_BOTTOM_ROW, _display_version, print_banner

# Rows of the RESTLESS title, to find where the title landed:
TITLE_ROW = "/ _, _/ / /___"
TITLE_LAST_ROW = "/_/ |_| /_____/"


def banner_lines(*, width: int, terminal: bool = True) -> list[str]:
    """The banner as plain text lines, printed to a console of the given width."""

    output = io.StringIO()
    print_banner(Console(file=output, force_terminal=terminal, width=width, color_system=None))

    return output.getvalue().splitlines()


def has_dots(line: str) -> bool:
    """Whether the line holds part of the goblin, which is drawn in braille dot characters."""

    return any("⠀" <= char <= "⣿" for char in line)


def test_wide_terminal_puts_goblin_and_title_side_by_side() -> None:
    lines = banner_lines(width=160)

    assert any(TITLE_ROW in line and has_dots(line) for line in lines)


def test_title_ends_on_the_row_where_the_right_ear_lime_ends() -> None:
    lines = banner_lines(width=160)

    title_bottom = next(index for index, line in enumerate(lines) if TITLE_LAST_ROW in line)

    assert title_bottom == _TITLE_BOTTOM_ROW


def test_version_and_project_address_sit_under_the_title() -> None:
    lines = banner_lines(width=160)

    title_bottom = next(index for index, line in enumerate(lines) if TITLE_LAST_ROW in line)

    # A blank line under the title, then the installed version, then the address:
    assert _display_version(metadata.version("restless")) in lines[title_bottom + 2]
    assert "https://github.com/mike-manko/restless" in lines[title_bottom + 3]


@pytest.mark.parametrize(
    ("pep440", "shown"),
    [("1.0.0a0", "v1.0.0-alpha"), ("1.0.0b2", "v1.0.0-beta.2"), ("2.1.0rc1", "v2.1.0-rc.1"), ("1.0.0", "v1.0.0"), ("1.0.0.dev1", "v1.0.0.dev1")],
    ids=["first-alpha", "later-beta", "release-candidate", "release", "unhandled-form"],
)
def test_version_is_shown_the_way_semver_writes_pre_releases(pep440: str, shown: str) -> None:
    assert _display_version(pep440) == shown


def test_narrow_terminal_stacks_them_instead_of_wrapping() -> None:
    lines = banner_lines(width=100)

    assert any(has_dots(line) for line in lines)
    assert any(TITLE_ROW in line for line in lines)
    assert not any(TITLE_ROW in line and has_dots(line) for line in lines)
    assert all(len(line) <= 100 for line in lines)


def test_nothing_is_printed_outside_a_terminal() -> None:
    assert banner_lines(width=160, terminal=False) == []


def test_nothing_is_printed_to_a_terminal_that_cannot_encode_braille() -> None:
    raw = io.BytesIO()
    cp1252 = io.TextIOWrapper(raw, encoding="cp1252")

    print_banner(Console(file=cp1252, force_terminal=True, width=160))
    cp1252.flush()

    assert raw.getvalue() == b""
