"""Create corpus-text-free plots with numbered senses from numerical analysis tables."""
import argparse
from collections import defaultdict
from pathlib import Path
from common import read_csv,output_path


def draw(analysis_dir,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=output_path(out);out.mkdir(parents=True,exist_ok=True)
    metrics=read_csv(Path(analysis_dir)/'metrics_by_decade.csv');senses=read_csv(Path(analysis_dir)/'sense_distribution_by_decade.csv')
    cases=[('差別','サベツ','sabetsu','val_t','Context valence'),('飯','メシ','meshi','depth_t','WordNet depth'),('土台','ドダイ','dodai','meta_t','Model-detected metaphoricity')]
    fig,axes=plt.subplots(2,3,figsize=(13,6),constrained_layout=True)
    for col,(lemma,reading,title,metric,ylabel) in enumerate(cases):
        rr=[r for r in metrics if r['lemma']==lemma and r['lemma_reading']==reading]
        ss=[r for r in senses if r['lemma']==lemma and r['lemma_reading']==reading]
        decades=sorted({int(r['decade'][:-1]) for r in rr});lookup={(int(r['decade'][:-1]),int(r['sense_index'])):float(r['ratio']) for r in ss};base=[0.]*len(decades)
        for sense in sorted({int(r['sense_index']) for r in ss}):
            heights=[lookup.get((d,sense),0.) for d in decades];axes[0,col].bar(decades,heights,bottom=base,width=8,label=f'Sense {sense}');base=[a+b for a,b in zip(base,heights)]
        axes[0,col].set(title=title,ylabel='Sense proportion',ylim=(0,1));axes[0,col].legend(fontsize=7)
        points=sorted((int(r['decade'][:-1]),float(r[metric])) for r in rr if r.get(metric) not in ['',None])
        axes[1,col].plot([x for x,y in points],[y for x,y in points],marker='o',markersize=3)
        axes[1,col].set(xlabel='Decade',ylabel=ylabel)
        for ax in axes[:,col]:ax.tick_params(axis='x',labelrotation=45);ax.grid(axis='y',alpha=.2)
    for suffix in ['png','pdf']:fig.savefig(out/f'case_studies.{suffix}',dpi=200)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--analysis-dir',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();draw(a.analysis_dir,a.output_dir)
