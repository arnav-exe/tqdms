"""
Showcase of `tqdm` animated bar styles (`animation=`).

Usage:
  python examples/animations_demo.py [style ...]

With no arguments, cycles through every registered style.
"""
import sys
from time import sleep

from tqdm import tqdm
from tqdm.animations import registry


def main():
    names = sys.argv[1:] or list(registry)
    for name in names:
        for _ in tqdm(range(180), desc=name, animation=name, unit='step'):
            sleep(1 / 60)


if __name__ == '__main__':
    main()
