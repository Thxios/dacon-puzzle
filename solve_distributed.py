
import os
import glob
import numpy as np
import json
import ray
import tqdm.auto as tqdm
import fire
from typing import Optional

from solver import Solver
from patch_util import *



@ray.remote
def solve_single(idx, logit_npy_path, max_branches=4):
    logit = np.load(logit_npy_path)
    n_true = int(np.sum(logit > 0))
    logit = logit.reshape((n_patches, n_patches, 2))


    if n_true >= 50:
        max_branches = max(max_branches - 1, 2)
    solver = Solver(logit, max_branches=max_branches)
    order, log_prob = solver.solve()

    order_ret = [order.index(i) + 1 for i in range(n_patches)]

    ret = dict(
        idx=idx,
        order=order_ret,
        log_prob=log_prob,
        n_true=n_true,
        cnt=solver.cnt,
    )

    return ret


def jobs_to_iterator(job_ids):
    while job_ids:
        done, job_ids = ray.wait(job_ids)
        yield ray.get(done[0])





def main(
        output_jsonl: str,
        logits_dir: str = 'logits',
        maximum: Optional[int] = None,
        num_cpus: int = 10,
):
    logit_list = glob.glob(os.path.join(logits_dir, '*.npy'))
    logit_list.sort()
    if maximum is not None:
        logit_list = logit_list[:maximum]
    print(len(logit_list))

    # ret = solve_single(3937, logit_list[3937])
    # print(ret)

    ray.init(num_cpus=num_cpus)

    jobs = []
    for i, logit_path in enumerate(logit_list):
        jobs.append(solve_single.remote(i, logit_path))

    iterator = jobs_to_iterator(jobs)
    with open(output_jsonl, 'w') as f:
        for ret in tqdm.tqdm(iterator, total=len(logit_list)):
            f.write(json.dumps(ret) + '\n')
            f.flush()



def solve_confusing(
        output_jsonl: str,
        compute_result: str,
        confuse_threshold: float = 12.,
        logits_dir: str = 'logits',
        num_cpus: int = 10,
):
    solved = []
    with open(compute_result, 'r') as f:
        for line in f.readlines():
            solved.append(json.loads(line))

    confusing_ind, confident = [], []
    for res in solved:
        if res['log_prob'] < confuse_threshold:
            confident.append(res)
        else:
            confusing_ind.append(res['idx'])

    print(len(confusing_ind))
    logit_list = glob.glob(os.path.join(logits_dir, '*.npy'))
    logit_list.sort()


    ray.init(num_cpus=num_cpus)

    jobs = []
    for idx in confusing_ind:
        jobs.append(solve_single.remote(idx, logit_list[idx], 5))

    iterator = jobs_to_iterator(jobs)
    with open(output_jsonl, 'w') as f:
        for conf in confident:
            f.write(json.dumps(conf) + '\n')
        f.flush()
        for ret in tqdm.tqdm(iterator, total=len(confusing_ind)):
            f.write(json.dumps(ret) + '\n')
            f.flush()



if __name__ == '__main__':
    fire.Fire(main)
