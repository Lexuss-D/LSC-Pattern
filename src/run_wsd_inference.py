"""Qwen3-8B word sense disambiguation with one model process per GPU."""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ANSWER_RE = re.compile(r"\[(\d+)\]")
THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def postprocess_generated_text(text):
    return THINK_BLOCK_RE.sub("", text, count=1).strip()


def parse_predicted_index(text):
    match = ANSWER_RE.search(text)
    return int(match.group(1)) if match else None


def annotate_record(record, prompt_path, response_text):
    index = parse_predicted_index(response_text)
    synset = next((c.get('synset_id') for c in record['wsd_candidate_synsets']
                   if index is not None and index > 0 and int(c['sense_index']) == index), None)
    return {**{key: value for key, value in record.items() if key != 'prompt'},
            'wsd_prompt_path': str(prompt_path.resolve()),
            'predicted_sense_index': index, 'predicted_synset_id': synset,
            'wsd_raw_response': response_text}


def run_worker(args):
    from vllm import LLM, SamplingParams

    files = sorted(args.prompt_dir.glob('*.jsonl'))[args.shard_id::args.num_shards]
    revision = (args.model_cache_dir / 'refs/main').read_text().strip()
    model = args.model_cache_dir / 'snapshots' / revision
    llm = LLM(model=str(model), tokenizer=str(model), trust_remote_code=True,
              tensor_parallel_size=1, dtype='bfloat16',
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=32768, disable_custom_all_reduce=True)
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=args.temperature, top_p=1.0,
                              max_tokens=args.max_tokens,
                              stop=['<|im_end|>', '<|endoftext|>'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in files:
        with path.open(encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]
        annotated = []
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            prompts = [tokenizer.apply_chat_template(
                [{'role': 'user', 'content': row['prompt']}], tokenize=False,
                add_generation_prompt=True, enable_thinking=True) for row in batch]
            responses = llm.generate(prompts, sampling, use_tqdm=False)
            for row, response in zip(batch, responses):
                text = response.outputs[0].text.strip() if response.outputs else ''
                annotated.append(annotate_record(row, path, postprocess_generated_text(text)))
        with (args.output_dir / path.name).open('w', encoding='utf-8') as f:
            for row in annotated:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        unresolved = sum(r['predicted_sense_index'] is None for r in annotated)
        print(f'{path.name}: {len(annotated)} usages, {unresolved} unresolved', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompt-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model-cache-dir', type=Path, required=True)
    p.add_argument('--gpu-ids', default='0')
    p.add_argument('--gpu-workers', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--max-tokens', type=int, default=16384)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.92)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--enable-thinking', action='store_true', default=True)
    p.add_argument('--num-shards', type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument('--shard-id', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--worker-mode', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker_mode:
        run_worker(args)
        return
    gpu_ids = args.gpu_ids.split(',')[:args.gpu_workers]
    processes = []
    for index, gpu in enumerate(gpu_ids):
        command = [sys.executable, str(Path(__file__).resolve()), '--worker-mode',
                   '--prompt-dir', str(args.prompt_dir.resolve()),
                   '--output-dir', str(args.output_dir.resolve()),
                   '--model-cache-dir', str(args.model_cache_dir.resolve()),
                   '--num-shards', str(len(gpu_ids)), '--shard-id', str(index),
                   '--batch-size', str(args.batch_size), '--max-tokens', str(args.max_tokens),
                   '--temperature', str(args.temperature),
                   '--gpu-memory-utilization', str(args.gpu_memory_utilization)]
        processes.append(subprocess.Popen(command, env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu.strip()}))
    codes = [process.wait() for process in processes]
    sys.exit(next((code for code in codes if code), 0))


if __name__ == '__main__':
    main()
