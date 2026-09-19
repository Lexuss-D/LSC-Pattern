"""Rebuild the frozen sense inventory from privately supplied dictionary JSONL."""
import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from common import ROOT,output_path,read_csv,write_json


def select_inventory(dictionary,out):
    targets=read_csv(ROOT/'data/targets.csv');refs=defaultdict(list)
    for r in read_csv(ROOT/'data/sense_ids.csv'):refs[r['entry_id']].append(r)
    wanted={r['sense_id'] for rs in refs.values() for r in rs};nodes={}
    def walk(value):
        if isinstance(value,dict):
            sid=str(value.get('@senseid',''))
            if sid in wanted:
                candidate=copy.deepcopy(value);candidate.pop('sense',None)
                nodes[sid]=candidate
            for x in value.values():walk(x)
        elif isinstance(value,list):
            for x in value:walk(x)
    with Path(dictionary).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():walk(json.loads(line))
    out=output_path(out,mkdir=True)
    with out.open('w',encoding='utf-8') as f:
        for t in targets:
            senses=[nodes[r['sense_id']] for r in sorted(refs[t['entry_id']],key=lambda r:int(r['sense_index']))]
            f.write(json.dumps({'data':{'entry':{'lemma':t['lemma'],'lemma_reading':t['lemma_reading'],'sense':senses}}},ensure_ascii=False)+'\n')
    print(f'Rebuilt {len(targets)} entries / {len(wanted)} senses')


def merge_generated(source,out):
    """Concatenate corpus-specific files into one file per frozen entry, preserving locators."""
    out=output_path(out);out.mkdir(parents=True,exist_ok=True)
    targets=read_csv(ROOT/'data/targets.csv')
    for t in targets:
        files=[Path(source)/f'{corpus}_all_{t["filename"]}' for corpus in ['BCCWJ','CHJ','SHC']]
        with (out/t['filename']).open('w',encoding='utf-8') as f:
            for p in files:
                with p.open(encoding='utf-8') as inp:
                    for line in inp:
                        if line.strip():f.write(line if line.endswith('\n') else line+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('inventory');q.add_argument('--dictionary',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('merge');q.add_argument('--input-dir',required=True);q.add_argument('--output-dir',required=True)
    a=p.parse_args()
    if a.command=='inventory':select_inventory(a.dictionary,a.output)
    else:merge_generated(a.input_dir,a.output_dir)
