"""Tests for `tqdm.animations` core plumbing."""
import re
from io import StringIO
from time import sleep

from pytest import warns

from tqdm import TqdmWarning, tqdm
from tqdm.animations import (
    C16, C256, NOCOLOUR, TRUECOLOUR, AnimatedBar, BarAnimation, TAnimator, base_rgb, blend,
    colour_tier, compose, noise, registry, resolve, sweep, wave01)
from tqdm.std import Bar

RE_RATE = re.compile(r'[\d.]+(it/s|s/it)')


class FakeTTY(StringIO):
    """StringIO pretending to be a terminal"""
    def isatty(self):
        return True


class Stars(BarAnimation):
    """minimal test animation: fill with `*`"""
    def __call__(self, frac, elapsed, width, ascii=False, colour=None):
        filled = int(frac * width)
        return '*' * filled + ' ' * (width - filled)


def _clean_env(monkeypatch):
    for var in ('NO_COLOR', 'FORCE_COLOR', 'CLICOLOR_FORCE', 'COLORTERM',
                'TERM', 'TERM_PROGRAM'):
        monkeypatch.delenv(var, raising=False)


def test_colour_tier_overrides(monkeypatch):
    """Test NO_COLOR/FORCE_COLOR precedence and the non-tty gate"""
    _clean_env(monkeypatch)
    tty = FakeTTY()
    monkeypatch.setenv('COLORTERM', 'truecolor')
    assert colour_tier(tty) == TRUECOLOUR
    monkeypatch.setenv('NO_COLOR', '1')
    assert colour_tier(tty) == NOCOLOUR
    monkeypatch.delenv('NO_COLOR')
    for force, tier in (('0', NOCOLOUR), ('1', C16), ('2', C256),
                        ('3', TRUECOLOUR), ('true', C16)):
        monkeypatch.setenv('FORCE_COLOR', force)
        assert colour_tier(StringIO()) == tier
    monkeypatch.delenv('FORCE_COLOR')
    assert colour_tier(StringIO()) == NOCOLOUR  # not a tty
    assert colour_tier(None) == NOCOLOUR


def test_colour_tier_term(monkeypatch):
    """Test TERM/TERM_PROGRAM probing ladder"""
    _clean_env(monkeypatch)
    monkeypatch.setattr('tqdm.animations.IS_WIN', False)
    tty = FakeTTY()
    assert colour_tier(tty) == NOCOLOUR  # no signals at all
    monkeypatch.setenv('TERM', 'xterm')
    assert colour_tier(tty) == C16
    monkeypatch.setenv('TERM', 'xterm-256color')
    assert colour_tier(tty) == C256
    monkeypatch.setenv('TERM', 'xterm-kitty')
    assert colour_tier(tty) == TRUECOLOUR
    monkeypatch.setenv('TERM', 'dumb')
    assert colour_tier(tty) == NOCOLOUR
    monkeypatch.setenv('TERM_PROGRAM', 'Apple_Terminal')
    assert colour_tier(tty) == C256
    monkeypatch.setenv('TERM_PROGRAM', 'iTerm.app')
    assert colour_tier(tty) == TRUECOLOUR


def test_compose():
    """Test colour run coalescing and trailing reset"""
    red = (255, 0, 0)
    res = compose([('a', red), ('b', red), ('c', None)], TRUECOLOUR)
    assert res == '\x1b[38;2;255;0;0mab\x1b[0mc'
    res = compose([('a', red), ('b', None)], NOCOLOUR)
    assert res == 'ab'  # no escapes at all without colour support
    res = compose([('a', red)], C256)
    assert res.startswith('\x1b[38;5;') and res.endswith('\x1b[0m')
    res = compose([('a', red)], C16)
    assert res == '\x1b[31ma\x1b[0m'  # nearest basic colour
    assert compose([], TRUECOLOUR) == ''


def test_helpers():
    """Test easing/noise/blend helpers"""
    assert wave01(0) == 0 and abs(wave01(0.5) - 1) < 1e-9
    assert sweep(0, 2) == 1  # cycle starts at the right
    assert abs(sweep(1.999999, 2)) < 1e-3  # decelerates into the left
    assert blend((0, 0, 0), (100, 200, 50), 0.5) == (50, 100, 25)
    vals = {noise(i, s) for i in range(9) for s in range(9)}
    assert all(0 <= v < 1 for v in vals) and len(vals) > 70  # deterministic spread
    assert noise(3, 7) == noise(3, 7)


def test_base_rgb():
    """Test recovering rgb from `Bar`-resolved colour codes"""
    assert base_rgb(None, (1, 2, 3)) == (1, 2, 3)
    assert base_rgb('\x1b[38;2;9;8;7m', (1, 2, 3)) == (9, 8, 7)
    assert base_rgb(Bar.COLOURS['GREEN'], (1, 2, 3)) == (0, 205, 0)


def test_resolve():
    """Test animation lookup, casing, passthrough and unknown names"""
    with warns(TqdmWarning, match="Unknown animation"):
        assert resolve('no-such-style') is None
    stars = Stars()
    assert resolve(stars) is stars  # instance passthrough
    registry['stars'] = Stars
    try:
        assert isinstance(resolve('STARS'), Stars)  # case-insensitive
    finally:
        del registry['stars']


def test_animated_bar():
    """Test `AnimatedBar` delegation and `Bar` format specs"""
    plain = AnimatedBar(0.5, 10)  # no animation: behaves like Bar
    assert format(plain) == format(Bar(0.5, 10))
    bar = AnimatedBar(0.5, 10, animation=Stars(), elapsed=1.0)
    assert format(bar) == '*****     '
    assert format(bar, '4') == '**  '  # width spec
    assert format(bar, 'b') == format(Bar(0.5, 10, charset=Bar.BLANK))  # blank


def test_tqdm_integration():
    """Test `animation=` end to end on a non-tty stream"""
    out = StringIO()
    with tqdm(total=10, file=out, animation=Stars(), mininterval=0) as t:
        for _ in range(10):
            t.update()
    res = out.getvalue()
    assert '**********' in res
    assert TAnimator._animator is None or not TAnimator._animator.is_alive()

    with warns(TqdmWarning, match="Unknown animation"):
        out = StringIO()
        with tqdm(total=10, file=out, animation='no-such-style',
                  mininterval=0) as t:
            t.update(5)
    assert '50%' in out.getvalue()  # falls back to a plain bar


def test_no_animation_output_unchanged():
    """Test `animation=None` output is identical to not passing it"""
    outs = []
    for kwargs in ({}, {'animation': None}):
        out = StringIO()
        with tqdm(total=10, file=out, mininterval=0, **kwargs) as t:
            t.update(5)
        outs.append(RE_RATE.sub('R', out.getvalue()))
    assert outs[0] == outs[1]


def test_animator_thread():
    """Test thread-driven refreshes and thread retirement"""
    out = FakeTTY()
    calls = []

    class Counting(BarAnimation):
        def __call__(self, frac, elapsed, width, ascii=False, colour=None):
            calls.append(elapsed)
            return ' ' * width

    t = tqdm(total=100, file=out, animation=Counting(), mininterval=0)
    animator = TAnimator._animator
    assert animator is not None and animator.is_alive()
    before = len(calls)
    sleep(0.45)  # no update() calls: frames must still advance
    assert len(calls) - before >= 2
    t.close()
    for _ in range(100):  # thread retires once no animated bars remain
        if not animator.is_alive():
            break
        sleep(0.05)
    assert not animator.is_alive()


def test_animator_respects_delay():
    """Test the animator does not display before `delay` elapses"""
    out = FakeTTY()
    t = tqdm(total=10, file=out, animation=Stars(), mininterval=0, delay=30)
    sleep(0.35)
    t.close()
    assert out.getvalue() == ''
