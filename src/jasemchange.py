"""JSD and Spearman evaluation on frozen 7-entry and 20-word cohorts."""
import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from common import ROOT,output_path,read_csv,write_csv,write_json


def human_scores(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        head=f.readline();f.seek(0);reader=csv.DictReader(f,delimiter='\t' if '\t' in head else ',')
        key=reader.fieldnames[0];return {r[key].strip():float(r['Compare']) for r in reader if r.get('Compare')}

def evaluate(generated,chj_scores,shc_scores,out,overlap_generated):
    out=output_path(out);out.mkdir(parents=True,exist_ok=True)
    frozen=read_csv(ROOT/'data/jasemchange_targets.csv')
    main={r['target_lemma'] for r in frozen if r['overlap']=='1'}
    counts=defaultdict(Counter);support=defaultdict(set);excluded=Counter()
    for path in sorted(Path(generated).glob('*.jsonl')):
        with path.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():continue
                r=json.loads(line);word=r.get('target_lemma',r['lemma']);index=r.get('predicted_sense_index')
                if not isinstance(index,int) or index<0:excluded['unresolved']+=1;continue
                try:year=int(r.get('year',r.get('timestamp')))
                except (ValueError,TypeError):excluded['invalid_year']+=1;continue
                period='CHJ' if year<=1925 else 'SHC' if year<2000 else 'BCCWJ'
                counts[(word,period)][index]+=1;support[word].update(int(c['sense_index']) for c in r['wsd_candidate_synsets']);support[word].add(0)
    overlap_counts=defaultdict(Counter)
    for target in read_csv(ROOT/'data/targets.csv'):
        if target['lemma'] not in main:continue
        path=Path(overlap_generated)/target['filename']
        with path.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():continue
                r=json.loads(line);index=r.get('predicted_sense_index')
                if not isinstance(index,int) or index<0:continue
                corpus=str(r['corpus']).removesuffix('_all').removesuffix('_cherry')
                overlap_counts[(target['lemma'],corpus)][index]+=1
    report={'period_rule':'year <=1925: CHJ; 1926..1999: SHC; >=2000: BCCWJ (historical evaluation rule)','excluded':dict(excluded),'comparisons':{}}
    per_word=[]
    for period,path in [('CHJ',chj_scores),('SHC',shc_scores)]:
        human=human_scores(path);data=[]
        for word,score in sorted(human.items()):
            a=counts[(word,period)];b=counts[(word,'BCCWJ')]
            labels=sorted(support[word]);pa=np.array([a[x] for x in labels],float);pb=np.array([b[x] for x in labels],float)
            jsd=float(jensenshannon(pa/pa.sum(),pb/pb.sum(),base=2)**2)
            row={'word':word,'comparison':period+'-BCCWJ','relatedness':score,'change_degree':4-score,'jsd':jsd,'n_earlier':sum(a.values()),'n_later':sum(b.values()),'overlap':int(word in main)}
            data.append(row);per_word.append(row)
        overlap=[]
        for r in data:
            if not r['overlap']:continue
            aa=overlap_counts[(r['word'],period)];bb=overlap_counts[(r['word'],'BCCWJ')]
            ll=sorted(set(aa)|set(bb));pa=np.array([aa[x] for x in ll],float);pb=np.array([bb[x] for x in ll],float)
            overlap.append({**r,'jsd':float(jensenshannon(pa/pa.sum(),pb/pb.sum(),base=2)**2),'n_earlier':sum(aa.values()),'n_later':sum(bb.values())})
        for r in data:r['cohort']='full'
        for r in overlap:r['cohort']='overlap'
        per_word.extend(overlap)
        for cohort,subset in [('full',data),('overlap',overlap)]:
            raw=spearmanr([r['jsd'] for r in subset],[r['relatedness'] for r in subset]);change=spearmanr([r['jsd'] for r in subset],[r['change_degree'] for r in subset])
            report['comparisons'][f'{period}-BCCWJ/{cohort}']={'n':len(subset),'rho_relatedness':float(raw.statistic),'rho_change':float(change.statistic),'p':float(change.pvalue),'words':[r['word'] for r in subset]}
    write_csv(out/'correlations_by_word.csv',per_word);write_json(out/'correlations.json',report);print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generated-dir',required=True);p.add_argument('--overlap-generated-dir',required=True);p.add_argument('--chj-scores',required=True);p.add_argument('--shc-scores',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();evaluate(a.generated_dir,a.chj_scores,a.shc_scores,a.output_dir,a.overlap_generated_dir)
