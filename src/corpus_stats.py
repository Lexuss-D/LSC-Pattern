"""Count private enriched CSV token records and years without exporting text."""
import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from common import CORPORA,output_path,write_json

def count_corpus(item):
    corpus,path=item;years=Counter();missing=0;rows=0
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);field=CORPORA[corpus]['year']
        for row in reader:
            rows+=1
            try:year=int(row[field]);years[year]+=1
            except (TypeError,ValueError):missing+=1
    return corpus,{'token_records':rows,'missing_or_invalid_year':missing,'year_min':min(years) if years else None,'year_max':max(years) if years else None,'records_by_year':dict(sorted(years.items()))}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    config=json.loads(Path(a.config).read_text());inputs=[(c,config[c+'_CSV']) for c in CORPORA]
    with ThreadPoolExecutor(max_workers=3) as pool:result=dict(pool.map(count_corpus,inputs))
    write_json(output_path(a.output,mkdir=True),result)
    print(json.dumps({k:{n:v for n,v in d.items() if n!='records_by_year'} for k,d in result.items()}))
