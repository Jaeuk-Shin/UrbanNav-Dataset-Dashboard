"""Multi-GPU parallel runner."""

import argparse


def _worker(stage_name, args_dict, gpu_id, seg_names):
    from stages import STAGES
    args = argparse.Namespace(**args_dict)
    args.device = f"cuda:{gpu_id}"
    args._segment_names = seg_names
    stage = STAGES[stage_name]()
    stage.run(args)


def run_parallel(stage_name, args):
    """Run a stage on one or more GPUs."""
    num_gpus = getattr(args, "num_gpus", 1) or 1

    if num_gpus <= 1:
        from stages import STAGES
        stage = STAGES[stage_name]()
        stage.run(args)
        return

    import multiprocessing as mp
    from pipeline.discovery import discover_segments

    all_segs = list(
        discover_segments(
            args.data_root,
            args.segments,
            subsample=getattr(args, "subsample", 1) or 1,
            fps=getattr(args, "fps", None),
            input_format=args.input_format,
        ).keys()
    )

    if not all_segs:
        print("No segments to process.")
        return

    chunks = [all_segs[i::num_gpus] for i in range(num_gpus)]
    args_dict = vars(args).copy()
    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id, chunk in enumerate(chunks):
        if chunk:
            p = ctx.Process(target=_worker, args=(stage_name, args_dict, gpu_id, chunk))
            p.start()
            procs.append((gpu_id, p))

    for gpu_id, p in procs:
        p.join()

    failed = [(gid, p) for gid, p in procs if p.exitcode != 0]
    if failed:
        gpus = ", ".join(str(gid) for gid, _ in failed)
        raise RuntimeError(f"Workers on GPU(s) {gpus} failed")
