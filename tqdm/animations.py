"""
Animated styles for the `{bar}` portion of a progress bar.

Usage:
>>> from tqdm import tqdm
>>> for _ in tqdm(range(100), animation="wave"):
...     ...

Styles are looked up in `registry` by name; see `Animation` for valid
names. Rendering degrades gracefully depending on the output terminal:
truecolour -> 256 -> 16 colours -> plain glyphs, and unicode -> ascii.
"""
import os
import re
import sys
from enum import Enum
from math import cos, pi
from warnings import warn

from .std import Bar, TqdmWarning
from .utils import IS_WIN, _is_ascii

__all__ = ['Animation', 'AnimatedBar', 'BarAnimation', 'registry', 'resolve',
           'colour_tier', 'NOCOLOUR', 'C16', 'C256', 'TRUECOLOUR']

# colour capability tiers (comparable: higher supports more colours)
NOCOLOUR, C16, C256, TRUECOLOUR = 0, 16, 256, 1 << 24

RE_RGB = re.compile(r'\x1b\[38;2;(\d+);(\d+);(\d+)m')
# rgb equivalents of the named `Bar.COLOURS`
_NAMED_RGB = {'BLACK': (0, 0, 0), 'RED': (205, 0, 0), 'GREEN': (0, 205, 0),
              'YELLOW': (205, 205, 0), 'BLUE': (60, 120, 240), 'MAGENTA': (205, 0, 205),
              'CYAN': (0, 205, 205), 'WHITE': (229, 229, 229)}
# (r, g, b, ansi fg code) for 16-colour approximation
_BASIC = ((0, 0, 0, 30), (205, 0, 0, 31), (0, 205, 0, 32), (205, 205, 0, 33),
          (60, 120, 240, 34), (205, 0, 205, 35), (0, 205, 205, 36), (229, 229, 229, 37),
          (127, 127, 127, 90), (255, 90, 90, 91), (90, 255, 90, 92), (255, 255, 90, 93),
          (120, 150, 255, 94), (255, 90, 255, 95), (90, 255, 255, 96), (255, 255, 255, 97))


def colour_tier(fp=None):
    """
    Colour capability of output `fp`, from the environment.

    Returns one of `NOCOLOUR`, `C16`, `C256`, `TRUECOLOUR`.
    Conservative: degrades on ambiguity rather than risk garbage output.
    """
    env = os.environ
    if env.get('NO_COLOR'):
        return NOCOLOUR
    force = env.get('FORCE_COLOR')
    if force is not None:
        return {'0': NOCOLOUR, 'false': NOCOLOUR, '2': C256,
                '3': TRUECOLOUR}.get(force.lower(), C16)
    try:
        if not fp.isatty():
            return NOCOLOUR
    except (AttributeError, OSError, ValueError):
        return NOCOLOUR
    if env.get('COLORTERM', '').lower() in ('truecolor', '24bit'):
        return TRUECOLOUR
    if IS_WIN:
        build = getattr(sys.getwindowsversion(), 'build', 0)
        if build >= 14931:  # conhost VT + 24-bit colour
            return TRUECOLOUR
        # colorama translates basic colours on older consoles
        return C256 if build >= 10586 else C16
    program = env.get('TERM_PROGRAM')
    if program == 'iTerm.app':
        return TRUECOLOUR
    if program == 'Apple_Terminal':  # no 24-bit support
        return C256
    term = env.get('TERM', '')
    if term in ('xterm-kitty', 'xterm-ghostty', 'wezterm', 'alacritty'):
        return TRUECOLOUR
    if '256color' in term:
        return C256
    if env.get('COLORTERM') or any(t in term for t in (
            'screen', 'tmux', 'xterm', 'vt100', 'vt220', 'rxvt', 'color',
            'ansi', 'cygwin', 'linux')):
        return C16
    return NOCOLOUR


def _cube256(r, g, b):
    """nearest xterm-256 palette index for an rgb colour"""
    if abs(r - g) < 12 and abs(g - b) < 12 and abs(r - b) < 12:  # greyscale ramp
        v = (r + g + b) // 3
        if v < 4:
            return 16
        if v > 246:
            return 231
        return 232 + min(23, max(0, (v - 8) * 24 // 240))
    return 16 + 36 * (r * 6 // 256) + 6 * (g * 6 // 256) + b * 6 // 256


def _fg(rgb, tier):
    """ANSI foreground code for `rgb` at colour capability `tier`"""
    if tier >= TRUECOLOUR:
        return '\x1b[38;2;%d;%d;%dm' % rgb
    if tier >= C256:
        return '\x1b[38;5;%dm' % _cube256(*rgb)
    if tier >= C16:
        r, g, b = rgb
        return '\x1b[%dm' % min(
            _BASIC, key=lambda c: (c[0] - r) ** 2 + (c[1] - g) ** 2 + (c[2] - b) ** 2)[3]
    return ''


def compose(cells, tier):
    """
    Join `(char, rgb)` cells into a string, coalescing colour codes.

    `rgb` may be `None` for uncoloured cells; resets colour at the end.
    """
    out, last = [], ''
    for ch, rgb in cells:
        code = _fg(rgb, tier) if rgb else ''
        if code != last:
            out.append(code or Bar.COLOUR_RESET)
            last = code
        out.append(ch)
    if last:
        out.append(Bar.COLOUR_RESET)
    return ''.join(out)


def blend(rgb1, rgb2, t):
    """linear interpolation between two rgb colours, `t` in [0, 1]"""
    return tuple(int(a + (b - a) * t) for a, b in zip(rgb1, rgb2))


def wave01(x):
    """sine ease mapping phase `x` (cycles) to [0, 1] (seamless loop)"""
    return 0.5 - 0.5 * cos(2 * pi * x)


def sweep(elapsed, period):
    """
    Leftward, decelerating sweep position in [0, 1].

    Each cycle starts at the right (1) and decelerates towards the left
    (0): backwards-decelerating texture motion measurably shortens
    perceived waits (Harrison et al., CHI 2010).
    """
    u = elapsed % period / period
    return (1 - u) ** 2


def noise(i, step):
    """deterministic pseudo-random float in [0, 1) for cell `i` at `step`"""
    n = (i * 2654435761 + step * 40503) & 0xffffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0xffffffff
    return (n >> 8) / 16777216.0


def base_rgb(colour, default):
    """rgb tuple from a `Bar`-resolved ANSI `colour` code (or `default`)"""
    if colour:
        m = RE_RGB.match(colour)
        if m:
            return tuple(map(int, m.groups()))
        for name, code in Bar.COLOURS.items():
            if code == colour:
                return _NAMED_RGB[name]
    return default


class BarAnimation:
    """
    Base class for animated bar styles.

    Subclasses implement `__call__` returning a string exactly `width`
    cells wide (ANSI colour codes allowed) for progress `frac` in
    [0, 1] after `elapsed` wall-clock seconds. Frames must be a pure
    function of the arguments (`elapsed` drives all motion).
    """
    interval = 0.1   # suggested seconds between refreshes (~10 fps)
    tier = NOCOLOUR  # colour capability, set by `resolve`

    def __call__(self, frac, elapsed, width, ascii=False, colour=None):  # noqa: B042
        raise NotImplementedError


registry = {}


def register(name):
    """Class decorator adding a `BarAnimation` to `registry` under `name`."""
    def inner(cls):
        registry[name] = cls
        return cls
    return inner


def resolve(animation, fp=None):
    """
    Return a configured `BarAnimation` instance (`None` if unknown).

    Parameters
    ----------
    animation  : str or Animation or BarAnimation.
    fp  : file-like, optional. Output stream (for colour detection).
    """
    if isinstance(animation, BarAnimation):
        anim = animation
    else:
        name = str(getattr(animation, 'value', animation)).lower()
        try:
            anim = registry[name]()
        except KeyError:
            warn(f"Unknown animation ({animation}); valid choices:"
                 f" [{', '.join(registry)}]", TqdmWarning, stacklevel=3)
            return None
    anim.tier = colour_tier(fp)
    return anim


class AnimatedBar(Bar):
    """`Bar` whose painting is delegated to a `BarAnimation`."""
    def __init__(self, frac, default_len=10, charset=Bar.UTF, colour=None,
                 animation=None, elapsed=0.0):
        super().__init__(frac, default_len, charset, colour)
        self.animation = animation
        self.elapsed = elapsed

    def _paint(self, N_BARS, charset):
        if self.animation is None or charset == Bar.BLANK:
            return super()._paint(N_BARS, charset)
        return self.animation(self.frac, self.elapsed, N_BARS,
                              ascii=_is_ascii(charset), colour=self.colour)


Animation = Enum('Animation', [(name.upper(), name) for name in registry], type=str)
Animation.__doc__ = "Names of the available animated bar styles."
