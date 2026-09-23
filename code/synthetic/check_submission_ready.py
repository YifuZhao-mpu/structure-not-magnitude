#!/usr/bin/env python3
"""
Submission gate. Fails while the manuscript still carries a placeholder, and names
exactly which fact is missing.

The failure mode this exists to prevent is specific: a manuscript that is finished in
every respect a script can check -- 269/269 numbers re-derived, 15/15 roadmap items
verified -- and is submitted with `[TODO: funder full name]` in the Funding declaration.
Numerical verification cannot catch that, because a placeholder is not a wrong number.
"""
import os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TARGETS = ["paper-stage/manuscript.md", "paper-stage/build/submission.md",
           "paper-stage/build/internal.md", "CITATION.cff"]
TOK = re.compile(r"\[TODO: ([^\]]*)\]")
# declarations a journal requires, which must exist and must not be a placeholder
REQUIRED = [("**Authors.**", "author list"),
            ("**Affiliations.**", "affiliations"),
            ("**Corresponding author.**", "corresponding author"),
            ("**Funding.**", "funding declaration"),
            ("**Authors' contributions.**", "contributions statement"),
            ("**Data availability.**", "data availability"),
            ("**Code availability.**", "code availability")]

def main():
    fails, todos = [], []
    ms_path = os.path.join(ROOT, TARGETS[0])
    if os.path.exists(ms_path):
        ms = open(ms_path).read()
        for marker, what in REQUIRED:
            if marker not in ms:
                fails.append(f"missing declaration: {what} ({marker})")
    else:
        print(f"note: {TARGETS[0]} not present; this is an authoring tool, not a "
              f"release artefact\n")
    for t in TARGETS:
        p = os.path.join(ROOT, t)
        if not os.path.exists(p):
            continue
        for i, line in enumerate(open(p).read().split("\n"), 1):
            for m in TOK.finditer(line):
                todos.append(f"{t}:{i}  {m.group(1)}")
    print("# Submission readiness\n")
    if todos:
        print(f"- placeholders remaining : {len(todos)}\n")
        for t in todos:
            print(f"  TODO {t}")
    else:
        print("- placeholders remaining : 0")
    if fails:
        print()
        for f in fails:
            print(f"  X {f}")
    print()
    if todos or fails:
        print("  NOT READY — the facts above are not derivable from the repository and "
              "must be supplied by the authors. Do not submit.")
        sys.exit(1)
    print("  READY — every required declaration is present and carries no placeholder.")

main()
