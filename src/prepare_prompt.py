"""Reconstruct the historical prompt privately; never distribute example text."""
import argparse
import ast
import json
from pathlib import Path
from common import ROOT,output_path


def reconstruct(source_script=None,example_json=None,output=None):
    if source_script:
        tree=ast.parse(Path(source_script).read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROMPT_TEMPLATE' for t in n.targets))
        value=node.value
        if isinstance(value,ast.Call) and isinstance(value.func,ast.Attribute) and value.func.attr=='strip':value=value.func.value
        template=ast.literal_eval(value).strip()
    else:
        example=json.loads(Path(example_json).read_text(encoding='utf-8'))
        template=(ROOT/'config/prompt_skeleton.txt').read_text(encoding='utf-8').strip()
        for key in ['demo_usage','demo_senses']:template=template.replace('{'+key+'}',example[key])
    p=output_path(output,mkdir=True);p.write_text(template,encoding='utf-8')
    print('Wrote prompt template:', p)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True);g.add_argument('--source-script');g.add_argument('--example-json');p.add_argument('--output',required=True);a=p.parse_args();reconstruct(a.source_script,a.example_json,a.output)
