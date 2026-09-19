"""Run the paper experiments in separate stages."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from common import ROOT,CORPORA,output_path,read_csv,write_csv,write_json

STAGES=['prepare','restore','wsd','dimensions','aggregate','baselines','evaluate','jasemchange','trends','figures']
INPUTS=['CHJ_CSV','SHC_CSV','BCCWJ_CSV','IWANAMI_TARGETS','IWANAMI_FULL','WORDNET_DB','PN_DICT','PROMPT_TEMPLATE','QWEN_CACHE','BERT_CHECKPOINT','METAPHOR_PACKAGE','METAPHOR_CHECKPOINT','BUNRUI_WORD','BUNRUI_DATABASE','JASEM_ROOT','JASEM_ANNOTATED','JASEM_CHJ_SCORES','JASEM_SHC_SCORES']
OPTIONAL=['EXISTING_GENERATED','EXISTING_BASELINES','EXISTING_JASEM_GENERATED','HISTORICAL_METRICS','EMBEDDING_MODEL','BERT_MODEL','GPU_IDS','GPU_WORKERS','BATCH_SIZE','MAX_TOKENS','GPU_MEMORY_UTILIZATION','DICTIONARY_ALREADY_ALIGNED','ANNOTATOR_MAP']

def env(name,default=None):
    return os.environ[name] if default is None else os.environ.get(name,default)


def command(script,*args,cwd=None):
    cmd=[sys.executable,str(ROOT/'src'/script),*[str(a) for a in args]]
    print('Running:',script,flush=True)
    subprocess.run(cmd,check=True,cwd=cwd or ROOT)

def run_inference(prompts,output):
    command('run_wsd_inference.py','--prompt-dir',prompts,'--output-dir',output,'--model-cache-dir',env('QWEN_CACHE'),
            '--gpu-ids',env('GPU_IDS','0'),'--gpu-workers',env('GPU_WORKERS','1'),'--batch-size',env('BATCH_SIZE','64'),
            '--max-tokens',env('MAX_TOKENS','16384'),
            '--temperature','0.0','--enable-thinking','--gpu-memory-utilization',env('GPU_MEMORY_UTILIZATION','0.92'))

def local_config(run):
    p=run/'inputs.json';write_json(p,{k:os.environ[k] for k in INPUTS+OPTIONAL if k in os.environ});return p

def prepare(run):
    wordlist=run/'wordlist.jsonl'
    command('prepare_inputs.py','inventory','--dictionary',env('IWANAMI_TARGETS'),'--output',wordlist)
    if env('DICTIONARY_ALREADY_ALIGNED','0')!='1':
        command('match_iwanami_to_wordnet_luw.py','--input',wordlist,'--wordnet-db',env('WORDNET_DB'),'--output',run/'matched.jsonl','--model',env('EMBEDDING_MODEL','sbintuitions/sarashina-embedding-v2-1b'))
    else:shutil.copyfile(wordlist,run/'matched.jsonl')
    command('extract_chj_shc_usages.py','--word-list-jsonl',run/'matched.jsonl','--chj-data-path',env('CHJ_CSV'),'--shc-data-path',env('SHC_CSV'),'--output-root',run/'usages')
    for c in ['CHJ','SHC']:
        source=run/'usages'/f'{c}_cherry';dest=run/'usages'/f'{c}_all'
        shutil.copytree(source,dest,dirs_exist_ok=True)
    command('extract_bccwj_usages.py','--data-path',env('BCCWJ_CSV'),'--word-list-jsonl',run/'matched.jsonl','--output-dir',run/'usages/BCCWJ_all')
    os.environ['LSCP_PROMPT_TEMPLATE']=env('PROMPT_TEMPLATE')
    command('build_lscp_wsd_prompts.py','--wordlist',run/'matched.jsonl','--usages-root',run/'usages','--output-root',run/'prompts','--summary-output',run/'prompt_summary.tsv')

def dimensions(run):
    met=run/'metaphor';met.mkdir(exist_ok=True)
    config={}
    for corpus,s in CORPORA.items():
        config[corpus+'_all']={'csv':env(corpus+'_CSV'),'surface':s['surface'],'lemma':s['lemma'],'reading':s['reading'],'timestamp':s['year'],
                              'pos':'品詞' if corpus=='BCCWJ' else '長単位品詞' if corpus=='CHJ' else 'corpus_col_32'}
    os.environ['LSCP_CORPUS_CONFIG']=json.dumps(config,ensure_ascii=False)
    command('metaphor_pipeline.py','build-manifest','--generated-dir',run/'generated','--manifest-output',met/'manifest.jsonl','--summary-output',met/'manifest_summary.json','--log-output',met/'manifest.log')
    command('metaphor_pipeline.py','build-raw','--manifest',met/'manifest.jsonl','--raw-dir',met/'raw','--metadata-output',met/'metadata.jsonl','--summary-output',met/'metadata_summary.json','--log-output',met/'metadata.log','--wordnet-matched',run/'matched.jsonl','--bunnrui-data',env('BUNRUI_WORD'))
    # Third-party code is used only from a private local copy, never from the source package.
    package=met/'external';source=Path(env('METAPHOR_PACKAGE'))
    if not package.exists():
        for p in source.rglob('*.py'):
            if '__pycache__' in p.parts:continue
            dest=package/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        for key,dest in [('BUNRUI_WORD','data/word/bunnrui_data.json'),('BUNRUI_DATABASE','data/bunnrui/bunnrui_database.json')]:
            target=package/dest;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(env(key),target)
    (met/'meta').mkdir(exist_ok=True);(met/'predictions').mkdir(exist_ok=True)
    for raw in sorted((met/'raw').glob('usage_targets_raw_shard_*.json')):
        shard=raw.stem.removeprefix('usage_targets_raw_shard_');meta=met/'meta'/f'meta_data_shard_{shard}.json';pred=met/'predictions'/f'predictions_shard_{shard}.json'
        subprocess.run([sys.executable,str(package/'preprocess_for_inference.py'),'--input-file',str(raw),'--output-file',str(meta),'--bunnrui-file',str(package/'data/word/bunnrui_data.json')],cwd=package,check=True)
        subprocess.run([sys.executable,str(package/'ModelPredictForData.py'),'--best-model-path',env('METAPHOR_CHECKPOINT'),'--bert-model-name',env('BERT_MODEL','tohoku-nlp/bert-base-japanese-v3'),'--input-file',str(meta),'--output-file',str(pred)],cwd=package,check=True)
    command('metaphor_pipeline.py','merge','--generated-dir',run/'generated','--predictions-dir',met/'predictions','--manifest',met/'manifest.jsonl','--metadata',met/'metadata.jsonl','--output-dir',run/'with_metaphor','--summary-output',met/'merge_summary.json','--log-output',met/'merge.log')
    command('add_luw_usage_valence.py','--input-dir',run/'with_metaphor','--output-dir',run/'with_dimensions','--pn-dict',env('PN_DICT'))
    command('prepare_inputs.py','merge','--input-dir',run/'with_dimensions','--output-dir',run/'merged')

def aggregate(run,mode):
    input_dir=Path(os.environ.get('EXISTING_GENERATED',str(run/'merged')))
    command('analyze_predictions.py','--input-dir',input_dir,'--output-dir',run/'analysis')
    if mode=='paper':
        paths=json.loads(env('HISTORICAL_METRICS'))
        target={(r['lemma'],r['lemma_reading']) for r in read_csv(ROOT/'data/targets.csv')};rows=[]
        for path in paths:
            for row in read_csv(path):
                if (row.get('lemma'),row.get('lemma_reading')) not in target:continue
                rows.append(row)
        write_csv(run/'analysis/metrics_by_decade.csv',rows)

def baselines(run,mode):
    restored=run/'restored_annotations.jsonl';rows=[]
    with restored.open() as f:
        for line in f:
            r=json.loads(line);row={**r,'word_sheet':r['entry_id'],'target':r['target_surface'],'timestamp':r['year'],'line_no':r['source_usage_row']}
            for i in range(1,4):
                col=f'label_annotator_{i}';row[col]=r[col] if r[col+'_valid']=='1' else ''
            rows.append(row)
    write_csv(run/'baseline_annotations.csv',rows)
    command('evaluate_iwanami_wsd_experiments.py','--annotations',run/'baseline_annotations.csv','--matched-jsonl',run/'matched.jsonl',
        '--generated-dir',os.environ.get('EXISTING_GENERATED',str(run/'merged')),'--wordnet-db',env('WORDNET_DB'),'--best-model',env('BERT_CHECKPOINT'),
        '--output-dir',run/'baselines','--embedding-model',env('EMBEDDING_MODEL','sbintuitions/sarashina-embedding-v2-1b'),'--tohoku-bert-model',env('BERT_MODEL','tohoku-nlp/bert-base-japanese-v3'))

def jasemchange(run):
    j=run/'jasemchange';j.mkdir(exist_ok=True)
    generated=os.environ.get('EXISTING_JASEM_GENERATED')
    if not generated:
        os.environ['LSCP_PROMPT_TEMPLATE']=env('PROMPT_TEMPLATE')
        command('build_jasemchange_iwanami_wordlist_from_xml.py','--iwanami-jsonl',env('IWANAMI_FULL'),'--samples-root',env('JASEM_ROOT'),'--target-list',ROOT/'data/jasemchange_targets.csv','--rawdata-root',Path(env('JASEM_ROOT'))/'rawdata','--output-jsonl',j/'wordlist.jsonl','--coverage-output',j/'coverage.tsv')
        command('build_jasemchange_wsd_prompts.py','--samples-root',env('JASEM_ROOT'),'--iwanami-wordlist',j/'wordlist.jsonl','--output-dir',j/'prompts','--manifest-output',j/'manifest.tsv')
        run_inference(j/'prompts',j/'generated');generated=j/'generated'
    command('jasemchange.py','--generated-dir',generated,'--overlap-generated-dir',os.environ.get('EXISTING_GENERATED',str(run/'merged')),
            '--chj-scores',env('JASEM_CHJ_SCORES'),'--shc-scores',env('JASEM_SHC_SCORES'),'--output-dir',j/'correlations')
    command('evaluate_jasemchange_pairwise_wsd.py','--annotated-dir',env('JASEM_ANNOTATED'),'--generated-dir',generated,'--output-dir',j/'pairwise')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=STAGES+['all','export-annotations','corpus-stats'],default='all')
    parser.add_argument('--mode',choices=['normalized','paper'],default='normalized')
    parser.add_argument('--run-dir')
    args=parser.parse_args()
    run=output_path(args.run_dir or ROOT/f'.local/runs/{args.mode}')
    run.mkdir(parents=True,exist_ok=True)
    config=local_config(run)
    for stage in STAGES if args.stage=='all' else [args.stage]:
        print(f'[{stage}]',flush=True)
        if stage=='export-annotations':command('annotations.py','export','--config',config)
        elif stage=='corpus-stats':command('corpus_stats.py','--config',config,'--output',run/'corpus_statistics.json')
        elif stage=='prepare':prepare(run)
        elif stage=='restore':command('annotations.py','restore','--config',config,'--output',run/'restored_annotations.jsonl')
        elif stage=='wsd':run_inference(run/'prompts',run/'generated')
        elif stage=='dimensions':dimensions(run)
        elif stage=='aggregate':aggregate(run,args.mode)
        elif stage=='baselines':baselines(run,args.mode)
        elif stage=='evaluate':command('evaluate_annotations.py','--predictions',run/'analysis/annotation_predictions.csv','--baseline-dir',os.environ.get('EXISTING_BASELINES',str(run/'baselines')),'--mode',args.mode,'--output-dir',run/'evaluation')
        elif stage=='jasemchange':jasemchange(run)
        elif stage=='trends':command('trends.py','--metrics',run/'analysis/metrics_by_decade.csv','--output-dir',run/'trends')
        elif stage=='figures':command('figures.py','--analysis-dir',run/'analysis','--output-dir',run/'figures')

if __name__=='__main__':main()
