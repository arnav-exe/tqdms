"""Tests for `tqdm.animations` core plumbing and styles."""
import re
from io import StringIO
from time import sleep

from pytest import mark, warns

from tqdm import TqdmWarning, tqdm
from tqdm.animations import (
    C16, C256, NOCOLOUR, TRUECOLOUR, AnimatedBar, BarAnimation, TAnimator, base_rgb, blend,
    colour_tier, compose, noise, ramp, registry, resolve, sweep, swell, wave01)
from tqdm.std import Bar
from tqdm.utils import RE_ANSI, disp_len

RE_RATE = re.compile(r'[\d.?]+(it/s|s/it)')


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
    assert swell(0) < 1e-9 and abs(swell(0.35) - 1) < 1e-9  # peak at `rise`
    assert swell(0.9) < swell(0.5)  # fast attack, slow fall
    assert ramp(((0, 0, 0), (100, 200, 50)), 0.5) == (50, 100, 25)
    assert ramp(((0, 0, 0), (8, 8, 8), (100, 200, 50)), 1) == (100, 200, 50)


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
        # rates (and the padding overwriting them) are timing-dependent
        outs.append(re.sub(r' +(?=[\r\n]|$)', '', RE_RATE.sub('R', out.getvalue())))
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
    sleep(0.6)  # no update() calls: frames must still advance
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


@mark.parametrize("name", sorted(registry))
def test_animation_invariants(name):
    """Test every style renders exact width at all tiers/fracs/times"""
    anim = registry[name]()
    for tier in (NOCOLOUR, C16, C256, TRUECOLOUR):
        anim.tier = tier
        for is_ascii in (False, True):
            for frac in (0, 0.01, 1 / 3, 0.5, 0.97, 1):
                for elapsed in (0, 0.05, 0.13, 1.7, 33.3, 12345.6):
                    for width in (1, 2, 3, 10, 47):
                        res = anim(frac, elapsed, width, ascii=is_ascii)
                        ctx = (name, tier, is_ascii, frac, elapsed, width, res)
                        assert disp_len(res) == width, ctx
                        assert '\n' not in res and '\r' not in res, ctx
                        if is_ascii:
                            assert all(ord(c) < 256
                                       for c in RE_ANSI.sub('', res)), ctx
                        if tier == NOCOLOUR:
                            assert '\x1b' not in res, ctx
                        else:
                            escapes = RE_ANSI.findall(res)
                            if escapes:  # no colour state may leak out
                                assert escapes[-1] == Bar.COLOUR_RESET, ctx


@mark.parametrize("name", sorted(registry))
def test_animation_in_tqdm(name):
    """Test every style through a real bar without errors"""
    out = StringIO()
    with tqdm(total=10, file=out, animation=name, mininterval=0, ncols=60) as t:
        for _ in range(10):
            t.update()
    assert '10/10' in out.getvalue()


def test_shimmer():
    """Test the gleam sweeps backwards (leftward) over the fill"""
    anim = registry['shimmer']()
    anim.tier = TRUECOLOUR
    a = anim(1, 0.05, 30)
    b = anim(1, 0.6, 30)
    assert a != b
    anim.tier = NOCOLOUR  # monochrome gleam still animates
    frames = {anim(1, t / 10, 30) for t in range(12)}
    assert len(frames) > 3


def test_rainbow():
    """Test many hues at truecolour, plain bar when monochrome"""
    anim = registry['rainbow']()
    anim.tier = TRUECOLOUR
    res = anim(1, 0, 30)
    assert len(set(RE_ANSI.findall(res))) > 8  # many distinct colours
    assert anim(1, 0, 30) != anim(1, 0.9, 30)
    anim.tier = NOCOLOUR
    assert anim(0.5, 0, 10) == format(Bar(0.5, 10))


def test_fire():
    """Test flicker at the edge varies with time in colour and mono"""
    anim = registry['fire']()
    anim.tier = TRUECOLOUR
    assert anim(0.7, 0.0, 30) != anim(0.7, 0.2, 30)
    anim.tier = NOCOLOUR
    frames = {anim(0.7, t * 0.07, 30) for t in range(8)}
    assert len(frames) > 2


def test_pacman():
    """Test chomping mouth position tracks the fraction"""
    anim = registry['pacman']()
    anim.tier = NOCOLOUR
    res = anim(0.5, 0.0, 20, ascii=True)
    assert res.index('C') == 10
    assert 'o' in res[11:] and 'o' not in res[:10]  # candy only ahead
    assert 'c' in anim(0.5, 0.3, 20, ascii=True)  # mouth chomps
    assert anim(1, 0.0, 20, ascii=True).index('C') == 19


def test_wave():
    """Test the wave style animates over time (colour and monochrome)"""
    anim = registry['wave']()
    for tier in (TRUECOLOUR, C16):
        anim.tier = tier
        assert anim(0.5, 0.0, 30) != anim(0.5, 0.4, 30)
    anim.tier = NOCOLOUR
    assert anim(0.5, 0.0, 30) != anim(0.5, 0.4, 30)  # shade-glyph wave
    assert anim(0.5, 0.0, 30, ascii=True) == anim(0.5, 0.4, 30, ascii=True)
