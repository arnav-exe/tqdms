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
import atexit
import os
import re
import sys
from colorsys import hsv_to_rgb
from enum import Enum
from math import cos, pi
from threading import Event, Lock, Thread
from time import time
from warnings import warn
from weakref import WeakSet

from .std import Bar, TqdmWarning
from .utils import IS_WIN, _is_ascii

__all__ = ['Animation', 'AnimatedBar', 'BarAnimation', 'TAnimator', 'registry',
           'resolve', 'colour_tier', 'NOCOLOUR', 'C16', 'C256', 'TRUECOLOUR']

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


def fill_glyphs(frac, width, charset):
    """glyphs (list of `width` chars) of a `frac`-filled bar, and full-cell count"""
    nsyms = len(charset) - 1
    filled, part = divmod(int(frac * width * nsyms), nsyms)
    res = [charset[-1]] * filled
    if filled < width:
        res.append(charset[part])
        res.extend([charset[0]] * (width - filled - 1))
    return res, filled


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


@register('wave')
class Wave(BarAnimation):
    """
    Brightness wave rolling backwards along the filled region.

    Backwards texture motion makes waits feel measurably shorter
    (Harrison et al., CHI 2010). Monochrome terminals get a subtle
    shade-glyph wave instead; ascii output stays static.
    """
    period = 1.6      # seconds per wavelength
    wavelength = 14   # cells
    base = (16, 170, 120)

    def __call__(self, frac, elapsed, width, ascii=False, colour=None):  # noqa: B042
        charset = Bar.ASCII if ascii else Bar.UTF
        glyphs, filled = fill_glyphs(frac, width, charset)
        phase = elapsed / self.period
        if self.tier:
            base = base_rgb(colour, self.base)
            dim = blend(base, (0, 0, 0), 0.35)
            bright = blend(base, (255, 255, 255), 0.4)
            cells = []
            for i, ch in enumerate(glyphs):
                if ch == charset[0]:  # unfilled
                    cells.append((ch, None))
                else:
                    w = wave01(i / self.wavelength + phase)
                    cells.append((ch, blend(dim, bright, w)))
            return compose(cells, self.tier)
        if ascii:
            return ''.join(glyphs)  # static: no colour, no safe glyph motion
        shades = '▒▓█'  # monochrome: shade-glyph wave
        return ''.join(
            shades[int(wave01(i / self.wavelength + phase) * 2.999)]
            if i < filled else ch for i, ch in enumerate(glyphs))


@register('shimmer')
class Shimmer(BarAnimation):
    """
    Bright gleam sweeping backwards over the filled region.

    The skeleton-screen effect: a leftward, decelerating highlight
    band (the empirically preferred motion; Harrison et al., CHI
    2010). Monochrome terminals get a shade-glyph gleam; ascii output
    stays static.
    """
    period = 1.8  # seconds per sweep
    band = 3.0    # gleam half-width in cells
    base = (96, 126, 218)

    def __call__(self, frac, elapsed, width, ascii=False, colour=None):  # noqa: B042
        charset = Bar.ASCII if ascii else Bar.UTF
        glyphs, filled = fill_glyphs(frac, width, charset)
        span = max(filled, 1)
        # travel beyond both ends so the gleam enters and leaves smoothly
        x = sweep(elapsed, self.period) * (span + 2 * self.band) - self.band
        if self.tier:
            base = base_rgb(colour, self.base)
            gleam = blend(base, (255, 255, 255), 0.8)
            cells = []
            for i, ch in enumerate(glyphs):
                if ch == charset[0]:  # unfilled
                    cells.append((ch, None))
                else:
                    g = max(0.0, 1 - abs(i - x) / self.band)  # soft falloff
                    cells.append((ch, blend(base, gleam, g * g)))
            return compose(cells, self.tier)
        if ascii:
            return ''.join(glyphs)
        return ''.join(  # monochrome: shade-glyph gleam
            '▓' if i < filled and abs(i - x) < self.band * 0.7 else ch
            for i, ch in enumerate(glyphs))


@register('rainbow')
class Rainbow(BarAnimation):
    """
    Hue-cycling colours flowing backwards along the filled region.

    Needs colour support to mean anything: monochrome output is a
    plain bar.
    """
    period = 3.0   # seconds per full hue rotation
    stretch = 1.2  # hue rotations across the bar width

    def __call__(self, frac, elapsed, width, ascii=False, colour=None):  # noqa: B042
        charset = Bar.ASCII if ascii else Bar.UTF
        glyphs, _ = fill_glyphs(frac, width, charset)
        if not self.tier:
            return ''.join(glyphs)
        cells = []
        for i, ch in enumerate(glyphs):
            if ch == charset[0]:  # unfilled
                cells.append((ch, None))
            else:
                h = (i * self.stretch / width + elapsed / self.period) % 1
                cells.append(
                    (ch, tuple(int(c * 255) for c in hsv_to_rgb(h, 0.85, 1))))
        return compose(cells, self.tier)


@register('fire')
class Fire(BarAnimation):
    """
    Ember gradient with a flickering burning edge.

    Deep red at the start of the fill through orange to a flickering
    yellow tip (deterministic noise keyed to elapsed time). Monochrome
    unicode gets shade-glyph flicker at the edge; ascii stays static.
    """
    interval = 0.07  # flicker looks livelier at ~14 fps
    _RAMP = ((80, 8, 0), (178, 34, 0), (255, 106, 0), (255, 200, 40))

    def __call__(self, frac, elapsed, width, ascii=False, colour=None):  # noqa: B042
        charset = Bar.ASCII if ascii else Bar.UTF
        glyphs, filled = fill_glyphs(frac, width, charset)
        step = int(elapsed / self.interval)
        span = max(filled, 1)
        if self.tier:
            cells = []
            for i, ch in enumerate(glyphs):
                if ch == charset[0]:  # unfilled
                    cells.append((ch, None))
                else:
                    t = min(i / span, 1.0)
                    j = min(int(t * 3), 2)
                    rgb = blend(self._RAMP[j], self._RAMP[j + 1], t * 3 - j)
                    d = span - i  # cells from the burning edge
                    if d <= 3:  # flicker towards white-hot at the tip
                        rgb = blend(rgb, (255, 240, 140),
                                    noise(i, step) * (4 - d) / 4 * 0.7)
                    cells.append((ch, rgb))
            return compose(cells, self.tier)
        if ascii or not filled:
            return ''.join(glyphs)
        for i in range(max(0, filled - 3), filled):  # monochrome flicker
            if noise(i, step) > 0.5:
                glyphs[i] = '▓▒'[int(noise(i + 7, step) * 1.999)]
        return ''.join(glyphs)


@register('ripple')
class Ripple(BarAnimation):
    """
    A low wave rippling backwards through the unfilled region.

    Phase-offset oscillators per cell (a la alive-progress waves): the
    fill stays static and readable while the remaining space undulates
    along the baseline. Fully glyph-based with an ascii variant.
    """
    period = 1.5
    wavelength = 7
    RAMP = ' ▁▂'
    RAMP_ASCII = ' .:'

    def __call__(self, frac, elapsed, width, ascii=False, colour=None):  # noqa: B042
        charset = Bar.ASCII if ascii else Bar.UTF
        glyphs, filled = fill_glyphs(frac, width, charset)
        ramp = self.RAMP_ASCII if ascii else self.RAMP
        for i in range(filled + 1, width):
            w = wave01(i / self.wavelength + elapsed / self.period)
            glyphs[i] = ramp[int(w * 2.999)]
        res = ''.join(glyphs)
        if self.tier and colour:
            return colour + res + Bar.COLOUR_RESET
        return res


@register('pacman')
class Pacman(BarAnimation):
    """
    The classic ILoveCandy easter egg: progress eats a trail of candy.

    A chomping mouth rides the boundary with candy dots ahead and
    emptiness behind; the mouth position is the progress fraction
    (a la pacman's decade-old namesake option). Works everywhere:
    plain ascii glyphs, coloured when the terminal allows.
    """
    interval = 0.12

    def __call__(self, frac, elapsed, width, ascii=False, colour=None):  # noqa: B042
        pos = min(int(frac * width), width - 1)
        mouth = 'Cc'[int(elapsed / 0.25) % 2]
        dot = 'o' if ascii else '·'
        cells = []
        for i in range(width):
            if i == pos:
                cells.append((mouth, (255, 210, 0)))
            elif i > pos and i % 2 == 0:  # uneaten candy on a fixed grid
                cells.append((dot, (222, 222, 222)))
            else:
                cells.append((' ', None))
        if self.tier:
            return compose(cells, self.tier)
        return ''.join(ch for ch, _ in cells)


class TAnimator(Thread):
    """
    Daemon thread refreshing animated bars between iterations.

    Keeps animations moving even when the wrapped iterable is slow.
    Started on demand for bars created with `animation=` on a tty;
    exits once no such bars remain.
    """
    _lock = Lock()
    _animator = None  # singleton
    _test = {}  # internal vars for unit testing

    def __init__(self, tqdm_cls):
        Thread.__init__(self, name="tqdm_animator")
        self.daemon = True  # kill thread when main killed (KeyboardInterrupt)
        self.tqdm_cls = tqdm_cls
        self.instances = WeakSet()
        self._time = self._test.get("time", time)
        self.was_killed = self._test.get("Event", Event)()
        atexit.register(self.was_killed.set)
        self.start()

    @classmethod
    def register(cls, instance):
        """Ensure a running animator thread and register `instance`."""
        with cls._lock:
            animator = cls._animator
            if animator is None or animator.was_killed.is_set():
                try:
                    animator = cls._animator = cls(type(instance))
                except Exception:  # pragma: no cover
                    return  # no thread support: animate passively on updates
            animator.instances.add(instance)

    def run(self):
        interval = 0.1
        while True:
            self.was_killed.wait(interval)
            if self.was_killed.is_set():
                return
            interval = 0.1
            with self.tqdm_cls.get_lock():
                # copy to avoid set-changed-during-iteration races
                for instance in self.instances.copy():
                    if self.was_killed.is_set():
                        return
                    if getattr(instance, 'disable', False):
                        self.instances.discard(instance)  # closed
                    elif hasattr(instance, 'start_t'):
                        animation = getattr(instance, '_animation', None)
                        if animation is not None:
                            interval = min(interval, animation.interval)
                            if instance._time() >= instance.start_t + instance.delay:
                                instance.refresh(nolock=True)
                    del instance
            if not self.instances:
                with TAnimator._lock:
                    if not self.instances:  # nothing new registered: retire
                        if TAnimator._animator is self:
                            TAnimator._animator = None
                        self.was_killed.set()
                        return


Animation = Enum('Animation', [(name.upper(), name) for name in registry], type=str)
Animation.__doc__ = "Names of the available animated bar styles."
