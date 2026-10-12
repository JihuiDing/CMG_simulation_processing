"""Read-only scan of all FINIT and F0000 files in pre_Petrel_realizations.

Per file: size, NUL runs (offset, length), md5; and for key sections the declared
size vs numeric tokens parsed before the first bad token (same regex as the
production parser). Files containing NULs are read a second time to test whether
the zeros are stable on disk or transient read errors.
Writes a CSV to the path given as argv[2].
"""
import csv
import hashlib
import re
import sys
from pathlib import Path

HDR = re.compile(rb"^\s*'([A-Z0-9_ ]{1,8})'\s+(\d+)\s+'([A-Z0-9]{4})'\s*$")
NUM = re.compile(rb"^[-+]?\d*\.?\d+([eE][-+]?\d+)?$")
NULRUN = re.compile(rb"\x00+")
KEYS = {"FINIT": ["PERMX", "PORO", "ACTID"], "F0000": ["TEMP"]}


def nul_runs(data):
    return [(m.start(), m.end() - m.start()) for m in NULRUN.finditer(data)]


def sections(data, keys):
    res = {k: None for k in keys}
    cur = None
    for line in data.split(b"\n"):
        m = HDR.match(line)
        if m:
            name = m.group(1).strip().decode()
            cur = name if name in keys else None
            if cur and res[cur] is None:
                res[cur] = [int(m.group(2)), 0, False]   # declared, parsed, stopped
            elif cur:
                res[cur].append("DUPLICATE")
            continue
        if cur is None or res[cur][2]:
            continue
        for tok in line.split():
            if NUM.match(tok):
                res[cur][1] += 1
            else:
                res[cur][2] = True
                break
    return res


def main(root, out_csv):
    rows = []
    for case in sorted(Path(root).iterdir()):
        if not case.is_dir():
            continue
        for ext in ("FINIT", "F0000"):
            f = case / f"{case.name}.{ext}"
            if not f.exists():
                rows.append({"case": case.name, "ext": ext, "status": "FILE MISSING"})
                continue
            data = f.read_bytes()
            runs = nul_runs(data)
            md5 = hashlib.md5(data).hexdigest()
            row = {"case": case.name, "ext": ext, "size": len(data), "md5": md5,
                   "nul_bytes": sum(n for _, n in runs), "n_nul_runs": len(runs),
                   "nul_runs": ";".join(f"{o}+{n}" for o, n in runs[:10])}
            if runs:
                data2 = f.read_bytes()
                row["reread_same_md5"] = hashlib.md5(data2).hexdigest() == md5
                del data2
            secs = sections(data, KEYS[ext])
            ok = True
            for k, v in secs.items():
                if v is None:
                    row[k] = "MISSING"
                    ok = False
                else:
                    row[k] = f"{v[1]}/{v[0]}" + (" DUP" if "DUPLICATE" in v else "")
                    ok &= v[1] == v[0]
            row["status"] = "OK" if ok and not runs else ("NUL" if runs else "PARSE")
            rows.append(row)
            print(case.name, ext, row["status"], row.get("nul_bytes"), {k: row[k] for k in KEYS[ext]}, flush=True)
            del data
    fields = ["case", "ext", "status", "size", "nul_bytes", "n_nul_runs", "nul_runs", "reread_same_md5",
              "PERMX", "PORO", "ACTID", "TEMP", "md5"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
