#!/usr/bin/env python
################################################################################
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys, os, subprocess

import argparse
from argparse import RawTextHelpFormatter

# add parent dir to sys.path to make verbose_converter visible for test
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import verbose_converter
from src import benchdnn_generator as benchdnn_gen

status = {'SUCCESS': 0, 'FAILED': 1}


def convert_dir_benchdnn2verbose(dir):
    return {
        'FWD_D': 'forward_training',
        'FWD_B': 'forward_training',
        'FWD_I': 'forward_inference',
        'BWD_D': 'backward_data',
        'BWD_W': 'backward_weights',
        'BWD_DW': 'backward'
    }.get(dir)


def generate_verbose(path_to_benchdnn, driver, batch):
    benchdnn_exe = path_to_benchdnn + '/benchdnn'
    sub_env = os.environ.copy()
    sub_env['DNNL_VERBOSE'] = '1'
    sub_env['DNNL_PRIMITIVE_CACHE_CAPACITY'] = '0'
    sub_args = [
        benchdnn_exe, f"--{driver}", f"--mode=R", f"-v1", f"--batch={batch}"
    ]
    try:
        sub = subprocess.run(sub_args,
                             capture_output=True,
                             text=True,
                             env=sub_env)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        return [
            status.get('FAILED'), f"subprocess.run() raised exception: " + \
                f"{e.stdout}"
        ], ""
    except BaseException as e:
        return [
            status.get('FAILED'), f"subprocess.run() raised exception: " + \
                f"{e.args}\n{e.stdout}"
        ], ""
    if sub.returncode != 0:
        # most likely converter generated incorrect batch file
        return [status.get('FAILED'), \
                f"subprocess.run() returned {sub.returncode},\n" + \
                f"args: {sub_args}\nstderr: {sub.stderr}"], ""
    v = ''
    # strip benchdnn and primitive execution time from verbose
    v_str = sub.stdout.splitlines()
    benchdnn_prop_kind = None
    benchdnn_with_rt_dims = False
    for l in v_str:
        if l.find("run: ") != -1:
            # detect prop kind in benchdnn log
            dir = '--dir='
            dir_start = l.find(dir)
            if dir_start != -1:
                dir_end = l.find(' ', dir_start)
                benchdnn_prop_kind = convert_dir_benchdnn2verbose(
                    l[dir_start + len(dir):dir_end])
            else:
                benchdnn_prop_kind = None
            # detect runtime dims
            rt_mask = '--runtime_dims_masks='
            rt_mask_start = l.find(rt_mask)
            if rt_mask_start != -1:
                rt_mask_end = l.find(' ', rt_mask_start)
                benchdnn_rt_dims = l[rt_mask_start + len(rt_mask):rt_mask_end]
                if benchdnn_rt_dims != '0:0':
                    benchdnn_with_rt_dims = True
            else:
                benchdnn_with_rt_dims = False

        # detect driver
        l_s = l.split(',')
        d = benchdnn_gen.convert_driver(l_s[3]) if len(l_s) > 3 else ''
        if len(l_s) > 3 and l_s[0] == 'dnnl_verbose' and d == driver:
            # filter out additional forward calls
            verbose_prop_kind = l_s[5]
            if benchdnn_prop_kind != None and verbose_prop_kind != benchdnn_prop_kind:
                continue
            # filter out cases with runtime dims
            if benchdnn_with_rt_dims:
                continue

            # remove time
            l_wo_time = "".join(f + ',' for f in l.split(',')[0:-1])[0:-1]

            v += l_wo_time + '\n'

    return [status.get('SUCCESS'), ''], v


def generate_batch(verbose, driver):
    verbose = verbose.splitlines()
    s, data = verbose_converter.convert(verbose_level=0,
                                        parser='oneDNN',
                                        input=verbose,
                                        action='generate',
                                        generator='benchdnn',
                                        split_output=True)
    if s != status.get('SUCCESS'):
        return [s, f"verbose_converter.convert() returned {s}"], ""

    filename = "test.generated"
    for key, value in data.items():
        # remove -- from driver name
        driver_filename = key + '.' + filename
        of = open(driver_filename, 'w')
        print(value, file=of)
    return [s, ''], driver + '.' + filename


def compare(driver, ref_v, comp_v):
    ref_lines = ref_v.splitlines()
    ref_lines = [l for l in ref_lines if driver in l]
    comp_lines = comp_v.splitlines()
    len(comp_lines)
    comp_lines = [l for l in comp_lines if driver in l]
    len(comp_lines)

    for r, c in zip(ref_lines, comp_lines):
        if r != c:
            ref_log_filename = f"{driver}.reference.log"
            com_log_filename = f"{driver}.computed.log"
            ref_log = open(ref_log_filename, 'w')
            com_log = open(com_log_filename, 'w')
            print(ref_v, file=ref_log)
            print(comp_v, file=com_log)
            return status.get('FAILED'), \
                    f"verboses do not match,\nref: {r}\ncom: {c}"

    return status.get('SUCCESS'), ''


def test(path_to_benchdnn, driver, batch):
    s, ref_verbose = generate_verbose(path_to_benchdnn, driver, batch)
    if s[0] != status.get('SUCCESS'):
        return s
    # XXX: Maybe generate batch and run becndhnn for each verbose line
    # separately to detect error on case level and not on batch level?
    # The reason behind testing on batch level is that ref_verbose generator
    # might introduce multiple verbose lines for single line in batch file
    s, gen_batch = generate_batch(ref_verbose, driver)
    if s[0] != status.get('SUCCESS'):
        return s
    s, verbose = generate_verbose(path_to_benchdnn, driver, gen_batch)
    if s[0] != status.get('SUCCESS'):
        return s

    return compare(driver, ref_verbose, verbose)


def main():
    realpath = os.path.dirname(os.path.realpath(__file__))
    print(realpath)
    realpath_benchdnn = realpath + '/../../../build/tests/benchdnn'
    args_parser = argparse.ArgumentParser(description='benchdnn test',
                                          formatter_class=RawTextHelpFormatter)
    args_parser.add_argument('-d',
                             '--dataset',
                             default=realpath + '/' + 'dataset_simple',
                             help='input with benchdnn batch files')
    args_parser.add_argument('-b',
                             '--benchdnn_path',
                             default=realpath_benchdnn,
                             help='Path to benchdnn executable')
    args_parser.add_argument('-i',
                             '--inputs_path',
                             default=realpath_benchdnn + '/' + 'inputs',
                             help='Path to benchdnn batch files')
    args = args_parser.parse_args()

    with open(args.dataset, 'r') as dataset:
        for case in dataset.readlines():
            if case[0] != '#' and case[0] != '\n':
                [driver, batch] = case.split(',')
                batch = batch.split('\n')[0]
                batch_file_path = args.inputs_path + '/' + driver + '/' + batch
                s = test(args.benchdnn_path, driver, batch_file_path)
                s_str = 'PASSED' if s[0] == status.get('SUCCESS') else 'FAILED'
                print(f"BENCHDNN TEST: {driver}, {batch}: {s_str} " + s[1])

    return status.get('SUCCESS')


if __name__ == "__main__":
    main()
