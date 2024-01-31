
## dacon puzzle

train
```shell
python3 train.py [OUTPUT_DIR] [DATA_DIR]
```

inference
```shell
python3 test_infer.py [CKPT_PATH] [DATA_DIR]
python3 solve_distributed.py [OUTPUT_JSONL_PATH]
python3 make_submission.py [JSONL_PATH] [SUBMISSION_NAME]
```

