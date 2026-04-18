from __future__ import annotations

import argparse
import json
import sys


def _print_json(obj: object) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False))
    sys.stdout.write("\n")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="ocr_corrector.py")
    p.add_argument("--text", type=str, required=True)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    out = {"corrected_text": args.text}
    if args.json:
        _print_json(out)
    else:
        sys.stdout.write(args.text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

