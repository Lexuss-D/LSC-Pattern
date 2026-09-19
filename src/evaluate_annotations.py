"""Text-free annotation agreement and WSD/baseline evaluation."""
from __future__ import annotations
import argparse
import json
import math
import re
from collections import Counter,defaultdict
from pathlib import Path
from annotations import ANNOTATORS,normalize_label,majority
from common import ROOT,output_path,read_csv,write_json,write_csv
from agreement_legacy import cohen_kappa,krippendorff_alpha_nominal


def paper_label(value):
    text=str(value).strip()
    match=re.fullmatch(r'(-?\d+)(?:\.0+)?',text)
    if match:return match.group(1)
    match=re.fullmatch(r'\[(-?\d+)\]',text)
    return match.group(1) if match else text

def gold_rows(rows,mode):
    result=[]
    for r in rows:
        if mode=='normalized':
            labels=[normalize_label(r[f'label_{a}']) if str(r[f'label_{a}_valid'])=='1' else None for a in ANNOTATORS]
        else:
            labels=[paper_label(r[f'label_{a}_raw']) or None for a in ANNOTATORS]
        m,v=majority(labels)
        result.append((r,labels,m))
    return result

def evaluation_gold_rows(rows,mode):
    gold=gold_rows(rows,mode)
    if mode=='paper':return gold
    mapping={(r['entry_id'],int(r['annotation_sense_index'])):normalize_label(r['model_sense_index']) for r in read_csv(ROOT/'data/annotation_sense_alignment.csv')}
    result=[]
    for r,labels,_ in gold:
        converted=[mapping.get((r['entry_id'],v)) if v is not None else None for v in labels]
        m,_=majority(converted)
        result.append((r,converted,m))
    return result


def agreement(rows,mode):
    grouped=defaultdict(list);pairs={}
    gr=gold_rows(rows,mode)
    for r,labels,_ in gr:grouped[r['entry_id']].append(['' if v is None else str(v) for v in labels])
    alphas=[krippendorff_alpha_nominal(v)[0] for v in grouped.values()]
    alphas=[v for v in alphas if math.isfinite(v)]
    for i,j in [(0,1),(0,2),(1,2)]:
        aa=[];bb=[]
        for r,labels,_ in gr:
            def enc(v):
                if v is None:return ''
                return (r['entry_id']+':'+str(v)) if mode=='normalized' else str(v)
            aa.append(enc(labels[i]));bb.append(enc(labels[j]))
        k,acc,n=cohen_kappa(aa,bb)
        pairs[f'{ANNOTATORS[i]}__{ANNOTATORS[j]}']={'kappa':k,'agreement':acc,'n':n}
    return {'mode':mode,'n_usages':len(rows),'n_entries':len(grouped),'majority_n':sum(m is not None for _,_,m in gr),
            'mean_word_alpha':sum(alphas)/len(alphas),'mean_pairwise_kappa':sum(v['kappa'] for v in pairs.values())/3,'pairs':pairs}

def score(rows,predictions,mode,method='qwen'):
    refs={a:[] for a in ANNOTATORS};refs['majority']=[]
    corpus=defaultdict(list)
    for r,labels,m in evaluation_gold_rows(rows,mode):
        pred=predictions.get(r['source_id'])
        if mode=='paper' and method not in ['qwen','mfs','random']:
            # Historical retrieval evaluation parsed integer labels and excluded invalid labels.
            labels=[normalize_label(r[f'label_{a}']) if str(r[f'label_{a}_valid'])=='1' else None for a in ANNOTATORS]
            m=normalize_label(m)
        def success(gold):
            if isinstance(pred,dict):return float(pred.get(str(gold),0))
            if pred is None:return 0.0
            return float(str(pred)==str(gold))
        for a,g in zip(ANNOTATORS,labels):
            if g is not None and not (mode=='paper' and method not in ['qwen','mfs','random'] and pred is None):refs[a].append(success(g))
        if m is not None:refs['majority'].append(success(m));corpus[r['corpus']].append(success(m))
    metrics={k:{'correct':sum(v),'n':len(v),'accuracy':sum(v)/len(v) if v else None} for k,v in refs.items()}
    metrics['macro_annotator_accuracy']=sum(metrics[a]['accuracy'] for a in ANNOTATORS)/3 if all(metrics[a]['accuracy'] is not None for a in ANNOTATORS) else None
    metrics['by_corpus']={k:{'correct':sum(v),'n':len(v),'accuracy':sum(v)/len(v)} for k,v in corpus.items()}
    metrics['covered_n']=sum(predictions.get(r['source_id']) is not None for r in rows)
    return metrics

def trivial_predictions(rows,inventory,mode):
    counts=defaultdict(Counter)
    for r,labels,_ in evaluation_gold_rows(rows,mode):counts[r['entry_id']].update(x for x in labels if x is not None)
    mfs={};random={}
    for r in rows:
        # Stable tie-break: smaller normalized sense index, then raw spelling.
        c=counts[r['entry_id']];label=sorted(c,key=lambda x:(-c[x],normalize_label(x) if normalize_label(x) is not None else 10**9,str(x)))[0] if c else None
        mfs[r['source_id']]=label
        candidates=list(range(1+int(next(t['n_senses'] for t in read_csv(ROOT/'data/targets.csv') if t['entry_id']==r['entry_id']))))
        # Historical random baseline samples all choices, including fallback 0.
        candidates=[int(x) for x in candidates]
        random[r['source_id']]={str(x):1/len(candidates) for x in candidates}
    return mfs,random

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--annotations',default=ROOT/'data/annotations.csv')
    p.add_argument('--predictions',required=True,help='CSV: source_id,predicted_sense_index')
    p.add_argument('--baseline-dir');p.add_argument('--mode',choices=['normalized','paper'],default='normalized');p.add_argument('--output-dir',required=True)
    a=p.parse_args();out=output_path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=read_csv(a.annotations);inventory=json.loads((ROOT/'data/annotation_inventory.json').read_text())
    preds={r['source_id']:normalize_label(r.get('predicted_sense_index')) for r in read_csv(a.predictions)}
    report={'alignment_policy':'exact definition alignment; unresolved gold labels excluded' if a.mode=='normalized' else 'historical numeric labels without inventory realignment','agreement':agreement(rows,a.mode),'methods':{'qwen':score(rows,preds,a.mode)}}
    mfs,rnd=trivial_predictions(rows,inventory,a.mode)
    original=gold_rows(rows,a.mode);aligned=evaluation_gold_rows(rows,a.mode)
    report['label_audit']={'raw_label_count':3*len(rows),'invalid_label_count':sum(r[f'label_{ann}_valid']!='1' for r in rows for ann in ANNOTATORS),'alignment_excluded_labels':sum(x is not None and y is None for (_,before,_),(_,after,_) in zip(original,aligned) for x,y in zip(before,after)),'original_majority_n':sum(m is not None for _,_,m in original),'evaluation_majority_n':sum(m is not None for _,_,m in aligned)}
    report['methods']['mfs']=score(rows,mfs,a.mode,'mfs')
    for index,ref in enumerate([*ANNOTATORS,'majority']):
        freq=defaultdict(Counter)
        for r,labels,m in evaluation_gold_rows(rows,a.mode):
            value=m if ref=='majority' else labels[index]
            if value is not None:freq[r['entry_id']][value]+=1
        fitted={r['source_id']:freq[r['entry_id']].most_common(1)[0][0] for r in rows if freq[r['entry_id']]}
        specific=score(rows,fitted,a.mode,'mfs')
        report['methods']['mfs'][ref]=specific[ref]
        if ref=='majority':report['methods']['mfs']['by_corpus']=specific['by_corpus']
    report['methods']['mfs']['macro_annotator_accuracy']=sum(report['methods']['mfs'][x]['accuracy'] for x in ANNOTATORS)/3
    report['methods']['random']=score(rows,rnd,a.mode,'random')
    if a.baseline_dir:
        files={'wordnet_bert':'experiment1_wordnet_predictions.csv','usage_gloss':'experiment2_gloss_retrieval_predictions.csv','target_gloss':'experiment2_tohoku_bert_predictions.csv'}
        for method,name in files.items():
            data=read_csv(Path(a.baseline_dir)/name)
            prediction={r['source_id']:normalize_label(r.get('pred_label',r.get('predicted_label',r.get('predicted_sense_index')))) for r in data}
            report['methods'][method]=score(rows,prediction,a.mode,method)
    write_json(out/'evaluation.json',report)
    print(json.dumps({'mode':a.mode,'agreement':report['agreement'],'methods':{m:v['majority']['accuracy'] for m,v in report['methods'].items()}},ensure_ascii=False))

if __name__=='__main__':main()
