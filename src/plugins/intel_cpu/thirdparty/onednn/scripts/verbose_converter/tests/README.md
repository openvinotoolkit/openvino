# Description: testing via benchdnn

## Requirements
 - python 3.7

## Usage
``` sh
python3  benchdnn_test.py [-h] [-b BENCHDNN_PATH] [-d DATASET] [-i INPUTS_PATH]
```

## Arguments
  - `{-h,--help}` -- show this help message and exit.
  - `{-b,--benchdnn_path} STRING` -- path to benchdnn.
  - `{-d,--dataset} STRING` -- input with benchdnn batch files.
  - `{-i,--inputs_path} STRING` -- path to benchdnn batch files.

## Example

```sh
$ python3 benchdnn_test.py -d dataset_simple
BENCHDNN TEST: driver=binary, batch=shapes_ci: PASSED
BENCHDNN TEST: driver=bnorm, batch=shapes_ci: PASSED
BENCHDNN TEST: driver=concat, batch=test_concat_ci: PASSED
BENCHDNN TEST: driver=conv, batch=shapes_basic: PASSED
BENCHDNN TEST: driver=conv, batch=shapes_auto: PASSED
BENCHDNN TEST: driver=conv, batch=shapes_regression_small_spatial: PASSED
BENCHDNN TEST: driver=deconv, batch=shapes_ci: PASSED
BENCHDNN TEST: driver=eltwise, batch=shapes_eltwise: PASSED
BENCHDNN TEST: driver=ip, batch=shapes_ci: PASSED
BENCHDNN TEST: driver=lnorm, batch=shapes_ci: PASSED
BENCHDNN TEST: driver=lrn, batch=shapes_ci: PASSED
BENCHDNN TEST: driver=matmul, batch=shapes_2d_ci: PASSED
BENCHDNN TEST: driver=pooling, batch=shapes_basic: PASSED
BENCHDNN TEST: driver=prelu, batch=shapes_ci: PASSED
BENCHDNN TEST: driver=reduction, batch=shapes_ci: PASSED
BENCHDNN TEST: driver=resampling, batch=shapes_ci: PASSED
BENCHDNN TEST: driver=rnn, batch=shapes_small: PASSED
BENCHDNN TEST: driver=shuffle, batch=option_set_min: PASSED
BENCHDNN TEST: driver=softmax, batch=shapes_ci: PASSED
BENCHDNN TEST: driver=sum, batch=test_sum_ci: PASSED
```
