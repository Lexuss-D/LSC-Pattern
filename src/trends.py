"""OLS, Theil-Sen, tie-corrected Mann-Kendall and global BH on decadal metrics."""
import argparse
import math
from collections import defaultdict
import numpy as np
from scipy import stats
from common import read_csv,write_csv,write_json,output_path
from trend_robustness import mann_kendall

METRICS=('val_t','depth_t','meta_t')

def bh(pvalues):
    p=np.asarray(pvalues,dtype=float);order=np.argsort(p);n=len(p)
    adjusted=np.minimum.accumulate((p[order]*n/np.arange(1,n+1))[::-1])[::-1]
    result=np.empty(n);result[order]=np.minimum(adjusted,1);return result

def analyze(path,out):
    out=output_path(out);out.mkdir(parents=True,exist_ok=True);series=defaultdict(lambda:defaultdict(list));decades=defaultdict(set)
    for r in read_csv(path):
        key=(r['lemma'],r['lemma_reading']);d=int(r['decade'].removesuffix('s'))/10
        decades[key].add(d)
        for m in METRICS:
            try:v=float(r[m])
            except (ValueError,TypeError):continue
            if math.isfinite(v):series[key][m].append((d,v))
    rows=[];robust=[]
    for key,by_metric in sorted(series.items()):
        for metric,points in by_metric.items():
            if len(points)<2:continue
            points.sort();x,y=np.array(points).T
            constant=np.all(y==y[0])
            fit=stats.linregress(x,y)
            row={'lemma':key[0],'lemma_reading':key[1],'metric':metric,'n_decades':len(decades[key]),'n_points':len(y),
                 'slope_per_decade':float(fit.slope),'intercept':float(fit.intercept),'r2':'' if constant else float(fit.rvalue**2),'constant':int(constant)}
            rows.append(row)
            if len(decades[key])>=10 and len(y)>=10:
                ts=stats.theilslopes(y,x);s,tau,z,p=mann_kendall(y)
                robust.append({**row,'theilsen_slope':float(ts.slope),'theilsen_low':float(ts.low_slope),'theilsen_high':float(ts.high_slope),
                               'mk_s':s,'mk_tau':tau,'mk_z':z,'mk_p':p})
    if robust:
        qs=bh([r['mk_p'] for r in robust])
        for r,q in zip(robust,qs):r['mk_q_global']=float(q);r['significant']=int(q<0.05)
    write_csv(out/'slopes.csv',rows)
    if robust:write_csv(out/'robustness.csv',robust)
    ranked=[]
    for metric in METRICS:
        selected=[r for r in rows if r['metric']==metric and r['n_decades']>=10]
        for direction in ['positive','negative']:
            ss=sorted([r for r in selected if r['slope_per_decade']*(1 if direction=='positive' else -1)>0],key=lambda r:abs(r['slope_per_decade']),reverse=True)[:10]
            ranked.extend({**r,'direction':direction,'rank':i} for i,r in enumerate(ss,1))
    if ranked:write_csv(out/'top_trends.csv',ranked)
    summary={'entries':len(series),'eligible_entries':sum(len(v)>=10 for v in decades.values()),'tested_series':len(robust),'significant_series':sum(r['significant'] for r in robust),'dimensions':{}}
    for metric in METRICS:
        rr=[r for r in robust if r['metric']==metric];rho=stats.spearmanr([r['slope_per_decade'] for r in rr],[r['theilsen_slope'] for r in rr]) if len(rr)>1 and len({r['slope_per_decade'] for r in rr})>1 and len({r['theilsen_slope'] for r in rr})>1 else None
        summary['dimensions'][metric]={'n':len(rr),'significant':sum(r['significant'] for r in rr),'ols_theilsen_rho':float(rho.statistic) if rho and math.isfinite(rho.statistic) else None,'p':float(rho.pvalue) if rho and math.isfinite(rho.pvalue) else None}
    write_json(out/'summary.json',summary);print(summary)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metrics',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();analyze(a.metrics,a.output_dir)
