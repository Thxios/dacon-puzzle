
import os
import pandas as pd
import numpy as np
import json
import fire


def main(
        jsonl_path: str,
        save_path: str,
        data_dir: str = 'data',
):
    solved = []
    with open(jsonl_path, 'r') as f:
        for line in f.readlines():
            solved.append(json.loads(line))
    solved.sort(key=lambda x: x['idx'])
    orders_np = np.array([x['order'] for x in solved])

    submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
    submission.iloc[:, 1:] = orders_np
    submission.to_csv(save_path, index=False)
    print(f'saved in {save_path}')


if __name__ == '__main__':
    fire.Fire(main)


