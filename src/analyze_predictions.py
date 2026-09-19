"""Aggregate the frozen main cohort; export only numerical results for analysis."""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
from common import ROOT,output_path,read_csv,write_csv,write_json
from semantic_metrics import WordStats,add_record,format_optional_float


def analyze(input_dir,output_dir):
    out=output_path(output_dir);out.mkdir(parents=True,exist_ok=True)
    targets=read_csv(ROOT/'data/targets.csv');counts=Counter();times=Counter();entry_counts=[]
    metrics=[];distributions=[];sense_metrics=[];predictions=[];inventory=[]
    ann_file=ROOT/'data/annotations.csv'
    ann_ids={r['source_id'] for r in read_csv(ann_file)} if ann_file.exists() else set()
    for target in targets:
        path=Path(input_dir)/target['filename']
        stats=WordStats(target['lemma'],target['lemma_reading']);valid=0;unparsed=0;by_year=Counter()
        with path.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():continue
                r=json.loads(line);counts['total_records']+=1
                if stats.rows_read==0:
                    inventory.append({'entry_id':target['entry_id'],'candidates':r['wsd_candidate_synsets']})
                add_record(stats,r,path,'decade')
                idx=r.get('predicted_sense_index');parsed=isinstance(idx,int) and idx>=0
                counts['resolved_predictions' if parsed else 'unresolved_predictions']+=1
                valid+=parsed;unparsed+=not parsed
                year=str(r.get('timestamp'));times[year]+=1;by_year[year]+=1
                if idx==0:counts['sense0']+=1
                corpus=str(r.get('corpus','')).removesuffix('_all').removesuffix('_cherry')
                sid=f"{corpus}:{Path(r['source_usage_file']).name}:{r['source_usage_row']}"
                if not ann_ids or sid in ann_ids:
                    predictions.append({'source_id':sid,'predicted_sense_index':'' if not parsed else idx})
        entry_counts.append({'entry_id':target['entry_id'],'records':stats.rows_read,'resolved':valid,'unresolved':unparsed,'invalid_time':stats.skipped_bad_time_bin})
        counts['invalid_time']+=stats.skipped_bad_time_bin
        for dec,b in sorted(stats.bins.items()):
            m={'lemma':target['lemma'],'lemma_reading':target['lemma_reading'],'decade':f'{dec}s','n':b.n,
               'val_t':format_optional_float(b.valence_sum/b.valence_n if b.valence_n else None),
               'depth_t':format_optional_float(b.depth_sum/b.depth_n if b.depth_n else None),
               'meta_t':format_optional_float(b.metaphor_sum/b.metaphor_n if b.metaphor_n else None),
               'corpus_present':'|'.join(sorted(b.corpus_counts)),
               'n_valence':b.valence_n,'n_scope':b.depth_n,'n_metaphor':b.metaphor_n,'n_resolved':sum(b.sense_counts.values())}
            metrics.append(m)
            for sense,n in sorted(b.sense_counts.items()):
                distributions.append({'lemma':target['lemma'],'lemma_reading':target['lemma_reading'],'decade':f'{dec}s','sense_index':sense,'n':n,'ratio':n/sum(b.sense_counts.values())})
                s=b.sense_stats[sense]
                sense_metrics.append({'lemma':target['lemma'],'lemma_reading':target['lemma_reading'],'decade':f'{dec}s','sense_index':sense,'n':n,
                    'val_t':s.valence_sum/s.valence_n if s.valence_n else '', 'depth_t':s.depth_sum/s.depth_n if s.depth_n else '', 'meta_t':s.metaphor_sum/s.metaphor_n if s.metaphor_n else ''})
    for name,data in [('metrics_by_decade.csv',metrics),('sense_distribution_by_decade.csv',distributions),('sense_metrics_by_decade.csv',sense_metrics),('entry_counts.csv',entry_counts),('annotation_predictions.csv',predictions)]:write_csv(out/name,data)
    # Candidate text is explicitly a private intermediary, never a release artifact.
    with (out/'inventory.jsonl').open('w',encoding='utf-8') as f:
        for item in inventory:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    write_json(out/'counts.json',{'totals':dict(counts),'years':dict(sorted(times.items())),'target_entries':len(targets)})
    print(json.dumps(dict(counts)),flush=True)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input-dir',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();analyze(a.input_dir,a.output_dir)
