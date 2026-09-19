"""Export anonymous labels with native corpus locators; restore text only locally.

Export reads the three workbooks and the original ordered corpus CSVs.
Restore uses only the public CSV and locally supplied enriched corpus CSVs.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict, deque
from pathlib import Path
from common import ROOT, CORPORA, output_path, read_csv, write_csv, write_json

ANNOTATORS = ['annotator_1','annotator_2','annotator_3']

def normalize_label(value):
    if value is None: return None
    text = unicodedata.normalize('NFKC', str(value)).strip()
    if text.startswith('[') and text.endswith(']'): text = text[1:-1]
    match = re.fullmatch(r'([0-9]+)(?:\.0+)?', text)
    return int(match.group(1)) if match else None

def majority(labels):
    counts = Counter(x for x in labels if x is not None)
    if not counts: return None, 0
    label, count = counts.most_common(1)[0]
    return (label if count >= 2 else None), count

def workbook_rows(path):
    from openpyxl import load_workbook
    out = {}; inventories = {}
    workbook = load_workbook(path, read_only=True, data_only=True)
    for sheet in workbook:
        header = None; senses = set()
        for values in sheet.iter_rows(values_only=True):
            if header is None:
                if 'source_id' in values and 'label' in values:
                    header = list(values); inventories[sheet.title] = sorted(senses)
                elif values and isinstance(values[0], str) and re.fullmatch(r'\[\d+\]', values[0].strip()):
                    senses.add(normalize_label(values[0]))
                continue
            row = dict(zip(header, values))
            if not row.get('source_id'): continue
            key = row['source_id']
            row['entry_id'] = sheet.title
            out[key] = row
    workbook.close()
    return out, inventories

def load_export_inputs(config):
    mapping = json.loads(Path(config['ANNOTATOR_MAP']).read_text())
    books = []; inv = None
    for ann in ANNOTATORS:
        rows, inventory = workbook_rows(mapping[ann])
        books.append(rows); inv = inventory
    keys = set(books[0])
    expected = {}
    for key in sorted(keys):
        r = books[0][key]
        lemma, reading = r['source_file'].rsplit('_', 2)[:2]
        labels = {}; valid = []
        for ann,b in zip(ANNOTATORS,books):
            v=b[key]['label']; raw='' if v is None else str(v); n=normalize_label(v)
            ok=n is not None and n in ({0}|set(inv[r['entry_id']]))
            labels[f'label_{ann}_raw']=raw
            labels[f'label_{ann}']='' if n is None else n
            labels[f'label_{ann}_valid']=int(ok)
            valid.append(n if ok else None)
        maj,votes=majority(valid)
        expected[key]={
            'source_id':key,'entry_id':r['entry_id'],'lemma':lemma,'lemma_reading':reading,
            'corpus':r['corpus'],'year':str(r['timestamp']),'source_usage_row':int(r['line_no']),
            'source_usage_file':r['source_file'],**labels,
            'majority_label':'' if maj is None else maj,'n_valid_labels':sum(x is not None for x in valid),
            'agreement_count':votes,'consensus_status':'majority' if maj is not None else 'no_majority',
        }
    return expected,inv

def locate_corpus(corpus, config, expected):
    """Locate each usage by its occurrence number in the original token stream."""
    spec=CORPORA[corpus]
    wanted={(e['lemma'],e['lemma_reading'],e['source_usage_row']):key
            for key,e in expected.items() if e['corpus']==corpus}
    entries={key[:2] for key in wanted};counts=Counter();located={}
    with Path(config[f'{corpus}_CSV']).open(encoding='utf-8-sig',newline='') as f:
        for record_no,row in enumerate(csv.DictReader(f),1):
            entry=(row[spec['lemma']],row[spec['reading']])
            if entry not in entries:continue
            counts[entry]+=1
            key=wanted.get((*entry,counts[entry]))
            if key:
                located[key]={'sample_id':row['サンプルID'],
                    'target_start':row[spec['start']],'target_end':row[spec['end']],
                    'position_system':f'{spec["start"]}/{spec["end"]}',
                    'corpus_record_no':record_no,'year':row[spec['year']]}
            if len(located)==len(wanted):break
    return located

def export(config_path):
    config=json.loads(Path(config_path).read_text());expected,inv=load_export_inputs(config)
    loc={};resources={}
    for corpus in CORPORA:
        p=Path(config[f'{corpus}_CSV'])
        loc.update(locate_corpus(corpus,config,expected))
        resources[corpus]={'resource_id':f'{corpus.lower()}_luw_paper_snapshot','filename':p.name,
            'bytes':p.stat().st_size,'official_version':'not recorded',
            'record_numbering':'1-based CSV data record, excluding header; not physical line number'}
    rows=[]
    for key,e in sorted(expected.items()):
        native=loc[key];year=int(e['year'])
        row={'usage_id':f"lscp_{e['corpus'].lower()}_{native['corpus_record_no']}",
            'entry_id':e['entry_id'],'lemma':e['lemma'],'lemma_reading':e['lemma_reading'],
            'corpus':e['corpus'],'resource_id':resources[e['corpus']]['resource_id'],
            'sample_id':native['sample_id'],'target_start':native['target_start'],'target_end':native['target_end'],
            'position_system':native['position_system'],'year':year,'decade':f'{year//10*10}s',
            'corpus_file_id':e['corpus']+'_CSV','corpus_record_no':native['corpus_record_no'],
            'source_id':key,'source_usage_row':e['source_usage_row']}
        for ann in ANNOTATORS:
            for suffix in ['_raw','','_valid']: row[f'label_{ann}{suffix}']=e[f'label_{ann}{suffix}']
        for field in ['majority_label','n_valid_labels','agreement_count','consensus_status']:row[field]=e[field]
        rows.append(row)
    write_csv(ROOT/'data/annotations.csv',rows)
    write_json(ROOT/'data/resources.json',resources)
    write_json(ROOT/'data/annotation_inventory.json',{k:sorted(set(v)|{0}) for k,v in sorted(inv.items())})
    summary={'rows':len(rows),'entries':len(inv),'corpora':dict(Counter(r['corpus'] for r in rows)),
             'majority_rows':sum(r['majority_label']!='' for r in rows),
             'invalid_labels':sum(r[f'label_{a}_valid']==0 for r in rows for a in ANNOTATORS),
             'native_locations':len(loc)}
    print(json.dumps(summary))

def restore(config_path,output,annotations=None):
    config=json.loads(Path(config_path).read_text());rows=read_csv(annotations or ROOT/'data/annotations.csv')
    output=output_path(output,mkdir=True);wanted=defaultdict(dict);found={}
    for r in rows:
        no=int(r['corpus_record_no'])
        wanted[r['corpus']][no]=r
    for corpus,w in wanted.items():
        spec=CORPORA[corpus];before=deque(maxlen=30);pending=[]
        def finish(item):
            r=item.pop('_annotation');tokens=item.pop('_tokens');left_n=item.pop('_left_n')
            found[r['usage_id']]={**r,'usage':tokens,'target_token_index':left_n,
                'target_surface':tokens[left_n],'left_context':''.join(tokens[:left_n]),
                'right_context':''.join(tokens[left_n+1:]),'usage_marked_text':''.join(tokens[:left_n])+'[['+tokens[left_n]+']]'+''.join(tokens[left_n+1:])}
        with Path(config[f'{corpus}_CSV']).open(encoding='utf-8-sig',newline='') as f:
            for i,v in enumerate(csv.DictReader(f),1):
                surface=v[spec['surface']]
                for item in pending:item['_tokens'].append(surface);item['_remaining']-=1
                done=[x for x in pending if x['_remaining']==0];pending=[x for x in pending if x['_remaining']>0]
                for item in done:item.pop('_remaining');finish(item)
                if i in w:
                    r=w[i]
                    pending.append({'_annotation':r,'_tokens':list(before)+[surface],'_left_n':len(before),'_remaining':30})
                before.append(surface)
                if i>max(w) and not pending:break
            for item in pending:item.pop('_remaining');finish(item)
        print(f'{corpus}: restored {len(w)} usages',flush=True)
    with output.open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(found[r['usage_id']],ensure_ascii=False)+'\n')
    print(f'Restored {len(found)} records to private output')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['export','restore'])
    p.add_argument('--config',required=True);p.add_argument('--output');p.add_argument('--annotations')
    a=p.parse_args()
    if a.command=='export':export(a.config)
    else:
        restore(a.config,a.output,a.annotations)
