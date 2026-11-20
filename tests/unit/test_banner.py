import io

from rich.console import Console

from restless.banner import _TITLE_BOTTOM_ROW, print_banner

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
