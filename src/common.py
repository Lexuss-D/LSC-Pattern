"""Paths, corpus column names and CSV/JSON helpers."""
from __future__ import annotations
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / '.local'

def output_path(path, mkdir=False):
    p = Path(path).expanduser().resolve()
    if mkdir:
        p.parent.mkdir(parents=True, exist_ok=True)
    return p

def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(rows[0])
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='raise')
        w.writeheader(); w.writerows(rows)

CORPORA = {
 'CHJ': {'surface':'長単位出現書字形','lemma':'長単位語彙素','reading':'長単位語彙素読み','year':'成立年','start':'開始文字位置','end':'終了文字位置'},
 'SHC': {'surface':'長単位出現書字形','lemma':'長単位語彙素','reading':'長単位語彙素読み','year':'成立年','start':'開始文字位置','end':'終了文字位置'},
 'BCCWJ': {'surface':'書字形出現形','lemma':'語彙素','reading':'語彙素読み','year':'出版年','start':'出現形開始位置','end':'出現形終了位置'},
}
