"""
Run every next-step test in order.

    python runAll.py --quick        # tiny sizes, a few minutes in total
    python runAll.py                # full sizes, about an hour
    python runAll.py --only 1,4,9   # just some of them

Any extra arguments (e.g. --sensor Landsat9) are passed to every test.
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = {int(v) for v in args[i + 1].split(",")}
        args = args[:i] + args[i + 2:]
    scripts = sorted(HERE.glob("test[0-9]_*.py"))
    failed = []
    for s in scripts:
        n = int(s.name[4])
        if only and n not in only:
            continue
        print(f"\n{'=' * 80}\n{s.name} {' '.join(args)}\n{'=' * 80}", flush=True)
        t = time.time()
        rc = subprocess.call([sys.executable, str(s), *args], cwd=HERE)
        print(f"--> {s.name} finished in {time.time() - t:.0f} s (exit code {rc})", flush=True)
        if rc != 0:
            failed.append(s.name)
    print("\nAll done." if not failed else f"\nFailed: {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
