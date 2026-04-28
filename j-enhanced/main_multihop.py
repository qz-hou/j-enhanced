from vllm import AsyncLLMEngine, AsyncEngineArgs
from SEAKR.dataset import get_dataset
from SEAKR.reasoner import MultiHopReasoner

from transformers import AutoTokenizer

from SEAKR.retriever import BM25
import warnings
from elasticsearch.exceptions import ElasticsearchDeprecationWarning
warnings.simplefilter('ignore', ElasticsearchDeprecationWarning)

import asyncio
import aiofiles
from tqdm.asyncio import tqdm
import json
import os
import pickle
from dataclasses import dataclass

@dataclass
class HyperParams:
    eigen_threshold: float 
    jacobian_threshold: float
    prob_threshold: float
    max_reasoning_steps: int
    max_docs: int
    decision_mode: str
    compute_jacobian: bool
    jacobian_doc_topk: int
    jacobian_doc_token_topk: int


def validate_tp_jacobian(tensor_parallel_size: int, compute_jacobian: bool):
    """Jacobian runs only on driver_worker; TP>1 needs all ranks in model forward NCCL."""
    if compute_jacobian and tensor_parallel_size > 1:
        raise ValueError(
            "--compute_jacobian 仅在 driver 进程上调用 compute_greedy_jacobian_score；"
            "TP>1 时另一 rank 不参与该前向，会在 NCCL ALLREDUCE 上与 driver 死锁并触发 600s 超时。"
            "请改用 --tensor_parallel_size 1（例如 CUDA_VISIBLE_DEVICES=1）。"
        )


def validate_scaffold_controls(decision_mode: str, compute_jacobian: bool):
    if decision_mode == "eigen":
        return
    if decision_mode == "jacobian":
        return
    if decision_mode == "hybrid":
        raise NotImplementedError(
            "decision_mode='hybrid' is scaffolded but hybrid decision "
            "logic is not implemented yet.")
    raise ValueError(f"Unsupported decision_mode: {decision_mode}")

error_count = 0
async def run_one_question(semaphore, entry, dataset_obj, llm_engine, retriever, logger_dir, finished_file, failed_file, lock, progress_bar, hyperparams: HyperParams):
    global error_count
    async with semaphore: 
        reasoner = MultiHopReasoner(
            qid = entry['qid'],
            question=entry['question'],
            dataset=dataset_obj,
            llm_engine=llm_engine,
            retriever=retriever,
            logger_dir=logger_dir,
            eigen_threshold=hyperparams.eigen_threshold,
            jacobian_threshold=hyperparams.jacobian_threshold,
            prob_threshold=hyperparams.prob_threshold,
            decision_mode=hyperparams.decision_mode,
            compute_jacobian=hyperparams.compute_jacobian,
            jacobian_doc_topk=hyperparams.jacobian_doc_topk,
            jacobian_doc_token_topk=hyperparams.jacobian_doc_token_topk,
        )
        try:
            output_data = await asyncio.wait_for(
                reasoner.solve(
                    max_reasoning_steps=hyperparams.max_reasoning_steps,
                    max_docs=hyperparams.max_docs
                ),
                timeout=20*60  # 超时时间，单位为秒
            )
            output_data['ground_truth'] = entry['answer']
            reasoner.logger.info(f"\nGround Truth: {entry['answer']}")
            async with lock:
                await finished_file.write(json.dumps(output_data) + '\n')
                await finished_file.flush()
            progress_bar.update(1)
        except Exception as e:
            reasoner.logger.error(e)
            if len(reasoner.running_steps) > 0:
                current_state = reasoner.output_current_state()
                parent_dir = os.path.dirname(logger_dir)
                storage_dir = os.path.join(parent_dir, "reasoning_ckpt")
                os.makedirs(storage_dir, exist_ok=True)
                pickle_file_name = os.path.join(storage_dir, f"{entry['qid']}.pkl")
                with open(pickle_file_name, 'wb') as f:
                    pickle.dump(current_state, f)
                reasoner.logger.info(f"States Saved to {pickle_file_name}")
            progress_bar.update(1)
            async with lock:
                await failed_file.write(json.dumps(
                    {
                        "qid": entry['qid'],
                        "error": str(e)
                    }
                )+"\n")
                await failed_file.flush()
            async with lock:
                error_count += 1
                if error_count >= 10:
                    for task in asyncio.all_tasks():
                        task.cancel()
                    raise Exception("Error limit reached, stopping all tasks")


async def run_full(dataset_list, dataset_obj, llm_engine, retriever, save_dir, hyperparams: HyperParams, max_workers=1):
    logger_dir = os.path.join(save_dir, 'logs')
    os.makedirs(logger_dir, exist_ok=True)
    finished_filename = os.path.join(save_dir, "results.jsonl")
    failed_filename = os.path.join(save_dir, "failed.jsonl")
    semaphore = asyncio.Semaphore(max_workers)  # 控制最大并发数

    lock = asyncio.Lock()
    async with aiofiles.open(finished_filename, mode='a') as finished_file, \
               aiofiles.open(failed_filename, mode='a') as failed_file:
        progress_bar = tqdm(total=len(dataset_list), desc="Processing dataset")
        tasks = [run_one_question(semaphore, entry, dataset_obj, llm_engine, retriever, logger_dir, finished_file, failed_file, lock, progress_bar, hyperparams) for entry in dataset_list]
        await asyncio.gather(*tasks)
        progress_bar.close()

async def main(args):
    validate_scaffold_controls(args.decision_mode, args.compute_jacobian)
    validate_tp_jacobian(args.tensor_parallel_size, args.compute_jacobian)
    dataset_obj = get_dataset(args.dataset_name, args.n_shot)
    dataset_list = dataset_obj.load_data()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    retriever = BM25(
        tokenizer=tokenizer, 
        index_name="wiki", 
        engine="elasticsearch",
        port=args.retriever_port,
    )

    gpu_mu = args.gpu_memory_utilization
    if gpu_mu is None:
        # Jacobian runs a training-style forward+backward; leave VRAM headroom.
        gpu_mu = 0.72 if args.compute_jacobian else 0.9

    engine_args = AsyncEngineArgs(
        model=args.model_name_or_path,
        served_model_name=args.served_model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=gpu_mu,
        selected_intermediate_layer=args.selected_intermediate_layer, #default 15
        eigen_alpha=args.eigen_alpha, # default 1e-3,
        decision_mode=args.decision_mode,
        compute_jacobian=args.compute_jacobian,
        # Ray will try to package/upload the entire working_dir (including
        # large datasets under `data/`), which easily exceeds Ray's default
        # 512MiB limit even for local execution. Prefer the non-Ray executor
        # for single-node multi-GPU runs.
        worker_use_ray=False,
        distributed_executor_backend="mp",
        disable_custom_all_reduce=True,
        disable_log_requests=True,
        disable_log_stats=True,
        enable_prefix_caching=True,
        enforce_eager=True
    )
    
    hyperparams = HyperParams(
        eigen_threshold=args.eigen_threshold,
        jacobian_threshold=args.jacobian_threshold,
        prob_threshold=args.prob_threshold,
        max_reasoning_steps=args.max_reasoning_steps,
        max_docs=args.max_docs,
        decision_mode=args.decision_mode,
        compute_jacobian=args.compute_jacobian,
        jacobian_doc_topk=args.jacobian_doc_topk,
        jacobian_doc_token_topk=args.jacobian_doc_token_topk,
    )

    llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
    await run_full(
        dataset_list=dataset_list,
        dataset_obj=dataset_obj,
        llm_engine=llm_engine,
        retriever=retriever,
        save_dir=args.save_dir,
        hyperparams=hyperparams,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the model with provided arguments.")
    parser.add_argument("--dataset_name", required=True, help="Name of the dataset.")
    parser.add_argument("--retriever_port", required=True, help="Port of Elastic Search Service.")
    parser.add_argument("--n_shot", type=int, default=10, help="Number of examples per task.")
    parser.add_argument("--model_name_or_path", required=True, help="Pre-trained model name or path.")
    parser.add_argument("--served_model_name", required=True, help="Model name for serving.")
    parser.add_argument("--selected_intermediate_layer", type=int, default=15, help="Selected layer for processing.")
    parser.add_argument("--eigen_alpha", type=float, default=1e-3, help="eigen alpha to compute eigen score")
    parser.add_argument("--eigen_threshold", type=float, default=-5.0, help="Threshold for eigen score.")
    parser.add_argument("--jacobian_threshold", type=float, default=0.5, help="Threshold for jacobian scoe.")
    parser.add_argument("--prob_threshold", type=float, default=0.15, help="Log probability threshold to form query.")
    parser.add_argument("--max_reasoning_steps", type=int, default=16, help="Maximum reasoning steps.")
    parser.add_argument("--max_docs", type=int, default=9, help="Maximum documents to retrieve.")
    parser.add_argument("--jacobian_doc_topk", type=int, default=3, help="Number of docs selected in jacobian temperature-scope filtering.")
    parser.add_argument("--jacobian_doc_token_topk", type=int, default=8, help="Maximum tokens per doc when selecting in jacobian mode.")
    parser.add_argument("--decision_mode", type=str, default="eigen", choices=["eigen", "jacobian", "hybrid"],
                        help="Decision score mode. Defaults to eigen to preserve current behavior.")
    parser.add_argument("--compute_jacobian", action="store_true",
                        help="Reserved scaffold flag for future Jacobian computation.")
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="vLLM tensor parallel size. With --compute_jacobian must be 1 (see validate_tp_jacobian).",
    )
    parser.add_argument("--save_dir", required=True, help="Directory to save the results.")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Max concurrent questions (async semaphore). Default 1 avoids vLLM queueing; raise if throughput is too low.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=None,
        help="vLLM GPU memory fraction (0,1]. Default: 0.72 with --compute_jacobian, else 0.9. Lower if Jacobian OOMs.",
    )
    args = parser.parse_args()

    if os.path.exists(args.save_dir):
        import datetime
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
        args.save_dir = f"{args.save_dir}_{timestamp}"
        
    os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, "args.txt"), 'w') as file:
        for arg in vars(args):
            file.write(f"{arg}: {getattr(args, arg)}\n")
    asyncio.run(main(args))
