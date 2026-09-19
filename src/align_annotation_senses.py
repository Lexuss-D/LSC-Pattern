"""Rebuild exact annotation-to-inference sense IDs without publishing definitions."""
import argparse
import json
import re
from pathlib import Path
from openpyxl import load_workbook
from annotations import normalize_label
from semantic_metrics import clean_definition_text
from common import output_path,write_csv


def align(workbook_paths,inventory_path):
    inventories=[]
    for path in workbook_paths:
        workbook=load_workbook(path,read_only=True,data_only=True);definitions={}
        for sheet in workbook:
            senses={}
            for values in sheet.iter_rows(values_only=True):
                if 'source_id' in values and 'label' in values:break
                if values and isinstance(values[0],str) and re.fullmatch(r'\[\d+\]',values[0].strip()):
                    index=normalize_label(values[0]);senses[index]=clean_definition_text(str(values[1] or ''))
            if senses:definitions[sheet.title]=senses
        workbook.close();inventories.append(definitions)
    model={}
    for line in Path(inventory_path).read_text().splitlines():
        item=json.loads(line)
        model[item['entry_id']]=item['candidates']
    output=[]
    for entry,senses in sorted(inventories[0].items()):
        for index,definition in sorted(senses.items()):
            matches=[c for c in model[entry] if c['sense_index']!=0 and clean_definition_text(str(c.get('definition') or ''))==definition and definition]
            if index==0:matched_index=0;senseid='';status='fallback'
            elif len(matches)==1:matched_index=matches[0]['sense_index'];senseid=matches[0].get('senseid','');status='exact_definition'
            else:matched_index='';senseid='';status='unresolved'
            output.append({'entry_id':entry,'annotation_sense_index':index,'model_sense_index':matched_index,'dictionary_sense_id':senseid,'status':status})
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--annotator-map',required=True,help='Private JSON mapping anonymous IDs to workbooks');p.add_argument('--model-inventory',required=True);p.add_argument('--output',default='.local/reports/annotation_sense_alignment.csv');a=p.parse_args()
    mapping=json.loads(Path(a.annotator_map).read_text());rows=align([mapping[f'annotator_{i}'] for i in [1,2,3]],a.model_inventory)
    output=output_path(a.output,mkdir=True)
    write_csv(output,rows);print('Wrote',len(rows),'sense correspondences;',sum(r['status']=='unresolved' for r in rows),'unresolved')
