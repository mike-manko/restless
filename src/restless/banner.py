import colorsys
import contextlib
import re
from importlib import metadata

from rich.console import Console
from rich.table import Table
from rich.text import Text

# The goblin's lime. Public, so other output can use the same colour:
GOBLIN_LIME = "#aaf028"

# Colours for the letters in _GOBLIN_MASK:
_GOBLIN_COLOURS: dict[str, str] = {
    "P": "#be6eff",  # purple, lit
    "p": "#7c3aed",  # purple, shaded
    "g": GOBLIN_LIME,  # lime: hair streaks, ears, visor HUD
    "w": "#c8f5ff",  # highlights: teeth, lens glare
    "s": "#82879a",  # grey: visor frame, earpiece
}

# The goblin's head as a dithered dot image. Each braille character is a 2x4 grid of dots, so the dots form an even grid in the terminal.
# Black outlines are left empty: the shape comes from the gaps between lit dots, like dot-matrix pixel art.
_GOBLIN = """
                                            ⢀⢔⣤⠄
                                       ⢀  ⢴⢽⣵⣿⠁
                                      ⢰⢽⢅⢼⣽⣿⣿⡇
                                     ⢴⢿⣽⢿⣿⣿⣿⡟       ⢀⢀⢴⠕
                  ⢀        ⠤⢄⣄     ⢄⢀⣽⢿⢽⣿⣿⣿⡟⠁  ⢀⠄⢀⢴⣴⢿⠝⠁
                  ⠙⠝        ⠙⢽⢿⣵⢄  ⢽⢿⣽⢿⣿⣿⣿⢿⠅⢀⢴⢿⢽⢟⣽⣿⢽⠝⠁
  ⢀⣴⢶⣴⢶⣴⢶⢄ ⢀                 ⠈⢿⣽⢿⣵⠐⣽⢿⣽⣽⣿⢟⢽⢟⣴⣽⣽⣿⣿⣿⢿⢿⢽⢥⢤⢄
  ⠁⠉⠉⠉⠙⠙⠿⢽⢿⢝⢝⣤⢤⣄      ⠁  ⢤⢤⢤⢄⣄⢹⢽⢿⠽⠅⢽⢿⣽⢿⣿⢽⢽⢽⣽⣿⢿⢟⢽⠿⠽⠙⠙⠉⠁⠁⠁
      ⠄⣄ ⠄⢅⢍⠙⠽⢿⣽⢿⣅⠐⢄⢀     ⠁⢉⠝⠝⠽⠽⢿⢵⢄⢽⢿⠝⢟⣽⢿⢵⣿⢿⠽⠝⠉⠁    ⠁
       ⠹⢿⣷⣴⣝⡕⣴⢄⠙⠿⢽⠄⢽⢗⢷⢥⠄ ⢔⠝⢉⢵⣴⢷⣵⢴⣍⢙⠽⢿⢵⠜⢽⢿⢽⠽⢅⢵⣽⠿⢍⢁⢴⢗⠄⠁   ⢀⣄⢀⣴⢷⢵⢷⢥⠄
        ⠈⢿⣿⣿⣿⣿⣝⣽⢇⢄⠁⠽⠿⠝⢽⣽⢷⣄⢴⢽⢿⣽⢿⣽⢿⣽⢿⢵⢷⣍⠙⠄⢿⢽⢷⣽⠿⠅⢤⣕⢝⢱⢽⠕⠅⠄⢴⣽⢿⢽⠟⠝⠉⠁⠁
          ⢿⣿⣿⣿⣿⣷⡟⠁⠔⠅⠕⢄⠝⢽⢿⣽⠿⠽⠿⠽⠿⠽⠿⠽⠿⠽⠿⠽⠵⠤⠙⢽⢿⠝⢡⠽⠿⠽⠽⠝⠁ ⠴⠴⠿⠝⠉⠄⢔⣡⠔
          ⢿⣿⣿⣿⣿⣿⡇ ⢁⢵⠕⠝⠕⠌⠉⠅ ⠔⠙⠑⠙⠑⠑⠁⠑⢶⣶⣶⡖⠄⠐ ⠕⠄⠐⠔⠑⠁⠑⢁⢖⠖⠖ ⠑  ⢵⣿⠟⠁
          ⠙⢿⣿⣿⣿⣿⣇ ⠁⠉⠁⠅⠁          ⢀⣴⣿⣿⠟⠁⠅⠁     ⣑⣑⢁⢵⠟⠝⠑⠄  ⢀⣽⠿⠁
            ⠙⢿⣿⣿⣿⣷⣤⣄⡁  ⢰⢕⢅⠄⠄    ⢔⢽⢝⠝⠁⠄⠁   ⢄   ⠐⠔⠝⠝⠅⠙⠁⠍⠁⢀⢌⡅
              ⠙⢿⣿⣿⠿⠟⢥⢴⠿⢍⢝⢽⢭⢵⠄  ⠔⠝⠕⠁⠄⠁   ⢀⠴⢿⢅ ⠈⠑⠝⠁ ⠁⠉⠁  ⠈⠝⠁
              ⠁⢸⠟⠁   ⢀⢴⢽⢽⢽⢽⢵⢅      ⢀⢀⢄⢄⢴⢷⢴⢄⣽⢷⣤⢄⠄ ⢀⢀⢄⠄⢰⣖
                    ⠰⠝⠙⠉⠉⢽⢟⢝⢿⢥⠁⠴⢅⠄⣤⣤⣅⠁⢉⢉⢙⠙⢙⢙⣉⢅⢀⠄⢀⢝⠵⠅  ⠑
                        ⠁ ⠙⠙⠷⢽⢿⢤⠙⠵⢹⡿⢱⣷⢝⣿⡟⣡⣍⢿⠟⠝⠉⢄⢿⠝⠁
                              ⠙⠽⢽⢵⢄⠁⠙⠿⠷⠝⠱⠿⠟⢁⢁⢴⠗⠉⠁
                                ⠁⠉⠙⠽⠿⢵⢵⢴⢷⢴⠿⠝⠉⠁
"""

# One letter per character of _GOBLIN, naming its colour in _GOBLIN_COLOURS. A dot marks an empty character:
_GOBLIN_MASK = """
............................................PPgg
.......................................P..PPggg
......................................PPPPgggg
.....................................PPPPgggg.......PPPP
..................g........pPP.....PPPPPggggg..PPPPPPPP
..................gg........PPPPP..PPPPgggPgPPPPPPPPPp
..PPPPPPPP.P.................PPPPPPPPPgggPPPPggggggPPPP
..pPPPPPPPPPPPPP......s..PPPPPPPPPPPPPggPPPggggPPPPPPPPP
......gg.ssPPPPPPPPPg.....pPPPPPPPPPPPPPPgggPPPP....s
.......ggggPpPPPPPpPPgPP.pPPPPPPPPPPPPPPPPPPPPPPPPPps...PPPPPPPPP
........ggggggPPPpPPPPPPPPPPPPPPPPPPPPPPPPPPPPgPPPPPppPPPPPPPPP
..........ggggggPPssssPPPPPPPPPPPPPPPPPPPPPPPPPPPPP.PPPPPpPgg
..........ggggggg.swssssPP.swwwwssswwwwss.wwwssssswww.s..gggg
..........ggggggg.sggss..........wwwwwwss.....gggsgggg..gggg
............ggggggggs..Ppppp....sswwwss...p...gggggggggggg
..............ggggggPPPPPPPPp..ssssss...PPPP.gggg.sgg..ggg
..............sggg...PPPPgPPP......ppppPPPPPPPpp.PPppgg
....................PPPPPPPPPPswwwwwwpPPPPPPPPwwgPPP..g
........................p.pPPPPPwwwwwwwwwwwwwwssgPP
..............................PPPPPwwwwwwwwwwPPgP
................................pPPPPPPPPPPPPP
"""

_RESTLESS = r"""
    ____    ______  _____  ______  __     ______  _____  _____
   / __ \  / ____/ / ___/ /_  __/ / /    / ____/ / ___/ / ___/
  / /_/ / / __/    \__ \   / /   / /    / __/    \__ \  \__ \
 / _, _/ / /___   ___/ /  / /   / /___ / /___   ___/ / ___/ /
/_/ |_| /_____/  /____/  /_/   /_____//_____/  /____/ /____/
"""

# Columns between the goblin and the title when they sit side by side:
_GAP = 4

# The goblin's row that the title's last row lines up with: where the lime of the right ear ends:
_TITLE_BOTTOM_ROW = 17

_PROJECT_URL = "https://github.com/mike-manko/restless"


# Print the goblin's head with the RESTLESS title, side by side when the terminal is wide enough
def print_banner(console: Console) -> None:
    """Print the banner. Prints nothing when the console isn't an interactive terminal (CI logs, a pipe) or can't encode braille characters."""

    # A non-UTF-8 console (e.g. cp1252) would raise UnicodeEncodeError on the braille dots, and the banner must never break a run:
    if not console.is_terminal or not console.encoding.startswith("utf"):
        return

    goblin = _paint_goblin(_lines(_GOBLIN), _lines(_GOBLIN_MASK))
    title = _lines(_RESTLESS)
    about = _about()

    # Side by side needs both widths plus the gap. A narrower terminal gets them stacked, rather than wrapped into a mess:
    if console.width >= _width(_lines(_GOBLIN)) + _GAP + _width(title):
        # Blank lines above the title push its last row down to _TITLE_BOTTOM_ROW:
        above = [""] * (_TITLE_BOTTOM_ROW + 1 - len(title))

        # Under the title, a blank line and then the plain text, which ends on the goblin's last row:
        right = _paint_gradient([*above, *title])
        right.append("\n\n")
        right.append_text(about)

        layout = Table.grid(padding=(0, _GAP))
        layout.add_column(no_wrap=True)
        layout.add_column(no_wrap=True)
        layout.add_row(goblin, right)
        console.print(layout)
    else:
        console.print(goblin)
        console.print()
        console.print(_paint_gradient(title))
        console.print()
        console.print(about)

    console.print()
    console.print()


# The version and the project's address, in plain text:
def _about() -> Text:
    """The installed version, then the project's address on the next line, as a link where the terminal supports links."""

    about = Text()

    # A source checkout that was never installed has no version to show, and the banner must never break a run:
    with contextlib.suppress(metadata.PackageNotFoundError):
        about.append(f"{_display_version(metadata.version('restless'))}\n")

    about.append(_PROJECT_URL, style=f"link {_PROJECT_URL}")

    return about


# The installed version, which Python packaging stores in PEP 440 form (1.0.0a0), the way people expect to read it:
def _display_version(pep440: str) -> str:
    """The version as shown to people: 1.0.0a0 becomes v1.0.0-alpha and 1.0.0b2 becomes v1.0.0-beta.2, the way SemVer writes pre-releases. Other forms are shown as they are."""

    match = re.fullmatch(r"(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?", pep440)
    if match is None:
        return f"v{pep440}"

    release, stage, number = match.groups()
    if stage is None:
        return f"v{release}"

    # PEP 440's a0 is SemVer's plain -alpha. Later pre-releases carry their number after a dot, as in -alpha.1:
    label = {"a": "alpha", "b": "beta", "rc": "rc"}[stage]
    suffix = label if number == "0" else f"{label}.{number}"

    return f"v{release}-{suffix}"


def _paint_goblin(art: list[str], mask: list[str]) -> Text:
    """Each character of the art in the colour its mask letter names."""

    goblin = Text()
    for art_line, mask_line in zip(art, mask, strict=True):
        for char, letter in zip(art_line, mask_line, strict=True):
            goblin.append(char, style=_GOBLIN_COLOURS.get(letter, ""))

        goblin.append("\n")

    goblin.rstrip()

    return goblin


def _paint_gradient(lines: list[str]) -> Text:
    """The lines with a colour gradient across the columns, purple on the left to lime on the right."""

    width = _width(lines)
    text = Text()
    for line in lines:
        for column, char in enumerate(line):
            text.append(char, style=_gradient_colour(column / (width - 1)))

        text.append("\n")

    text.rstrip()

    return text


def _gradient_colour(position: float) -> str:
    """Purple at 0.0, lime at 1.0, moving round the colour wheel through blue and green. A straight RGB blend would pass through grey."""

    hue = 0.75 + (0.23 - 0.75) * position
    red, green, blue = colorsys.hls_to_rgb(hue, 0.6, 0.85)

    return f"#{round(red * 255):02x}{round(green * 255):02x}{round(blue * 255):02x}"


def _lines(art: str) -> list[str]:
    """The art's lines, without the empty first and last line of the triple-quoted string."""

    return art.strip("\n").splitlines()


def _width(lines: list[str]) -> int:
    return max(len(line) for line in lines)
