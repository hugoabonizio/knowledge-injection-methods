import os
import sys
import json
from pathlib import Path
from statistics import mean
from lm_eval.utils import (
    get_latest_filename,
    get_results_filenames,
)


def read_lmeval(results_path):
    with open(results_path) as f:
        results = json.load(f)
        accs = []
        for result, metrics in results["results"].items():
            accs.append(metrics["acc,none"])
        return mean(accs)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: read-lm-eval-results.py [path-to-LM-Eval-results]")

    path = sys.argv[1]
    if os.path.isfile(path):
        print(read_lmeval(path))
    else:
        results_dir = Path(path)
        files = [f.as_posix() for f in results_dir.iterdir() if f.is_file()]
        results_filenames = get_results_filenames(files)
        latest_results = get_latest_filename(results_filenames)
        print(read_lmeval(latest_results))
