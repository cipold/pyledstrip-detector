import argparse
import os
import sys

import pyperclip

from ledworld import LedWorld


def main(options):
    world = LedWorld.from_json_file(options.infile)
    world.plot()
    world.fill_missing_leds()
    world.plot()
    world.smoothen()
    world.plot()
    world.to_json_file(options.outfile)


def read_arguments(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="""
        Postprocess Pyledstrip Heightmap
        """.strip()
    )
    parser.add_argument('--infile', help="input file", default=None, type=str)
    parser.add_argument('--outfile', help="output file", default=None, type=str)

    options = parser.parse_args(args)

    if options.infile is None:
        options.infile = pyperclip.paste()

    if not os.path.isfile(options.infile):
        raise Exception("no such file")

    if options.outfile is None:
        options.outfile = options.infile + ".processed.json"

    return options


if __name__ == '__main__':
    options = read_arguments(sys.argv[1:])
    main(options)
