# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import errno
import numpy as np
import paddle

from paddle.base.framework import (
    Program,
    Parameter,
    default_main_program,
    default_startup_program,
    Variable,
    program_guard,
    dygraph_not_support,
    static_only,
)
from paddle.base import core


def prepend_feed_ops(
    inference_program, feed_target_names, feed_holder_name='feed'
):
    if len(feed_target_names) == 0:
        return

    global_block = inference_program.global_block()
    feed_var = global_block.create_var(
        name=feed_holder_name,
        type=core.VarDesc.VarType.FEED_MINIBATCH,
        persistable=True,
    )

    for i, name in enumerate(feed_target_names):
        if not global_block.has_var(name):
            raise ValueError(
                "The feeded_var_names[{i}]: '{name}' doesn't exist in pruned inference program. "
                "Please check whether '{name}' is a valid feed_var name, or remove it from feeded_var_names "
                "if '{name}' is not involved in the target_vars calculation.".format(
                    i=i, name=name
                )
            )
        out = global_block.var(name)
        global_block._prepend_op(
            type='feed',
            inputs={'X': [feed_var]},
            outputs={'Out': [out]},
            attrs={'col': i},
        )


def append_fetch_ops(
    inference_program, fetch_target_names, fetch_holder_name='fetch'
):
    global_block = inference_program.global_block()
    fetch_var = global_block.create_var(
        name=fetch_holder_name,
        type=core.VarDesc.VarType.FETCH_LIST,
        persistable=True,
    )

    for i, name in enumerate(fetch_target_names):
        global_block.append_op(
            type='fetch',
            inputs={'X': [name]},
            outputs={'Out': [fetch_var]},
            attrs={'col': i},
        )

@static_only
def save_inference_model(
    dirname,
    feeded_var_names,
    target_vars,
    executor,
    main_program=None,
    model_filename=None,
    params_filename=None,
    export_for_deployment=True,
    program_only=False,
    clip_extra=True,
    legacy_format=False,
):
    if isinstance(feeded_var_names, str):
        feeded_var_names = [feeded_var_names]
    elif export_for_deployment:
        if len(feeded_var_names) > 0:
            # TODO(paddle-dev): polish these code blocks
            if not (
                bool(feeded_var_names)
                and all(isinstance(name, str) for name in feeded_var_names)
            ):
                raise ValueError("'feed_var_names' should be a list of str.")

    if isinstance(target_vars, Variable):
        target_vars = [target_vars]
    elif export_for_deployment:
        if not (
            bool(target_vars)
            and all(isinstance(var, Variable) for var in target_vars)
        ):
            raise ValueError("'target_vars' should be a list of Variable.")

    main_program = paddle.static.io._get_valid_program(main_program)

    # remind user to set auc_states to zeros if the program contains auc op
    all_ops = main_program.global_block().ops
    for op in all_ops:
        # clear device of Op
        device_attr_name = core.op_proto_and_checker_maker.kOpDeviceAttrName()
        op._set_attr(device_attr_name, "")
        if op.type == 'auc':
            warnings.warn(
                "please ensure that you have set the auc states to zeros before saving inference model"
            )
            break

    with program_guard(main_program):
        uniq_target_vars = []
        for i, var in enumerate(target_vars):
            uniq_target_vars.append(var)
        target_vars = uniq_target_vars
    target_var_name_list = [var.name for var in target_vars]

    # when a pserver and a trainer running on the same machine, mkdir may conflict
    save_dirname = dirname
    try:
        save_dirname = os.path.normpath(dirname)
        os.makedirs(save_dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if model_filename is not None:
        model_basename = os.path.basename(model_filename)
    else:
        model_basename = "__model__"
    model_basename = os.path.join(save_dirname, model_basename)

    # When export_for_deployment is true, we modify the program online so that
    # it can only be loaded for inference directly. If it's false, the whole
    # original program and related meta are saved so that future usage can be
    # more flexible.

    origin_program = main_program.clone()

    if export_for_deployment:
        main_program = main_program.clone()
        global_block = main_program.global_block()
        need_to_remove_op_index = []
        for i, op in enumerate(global_block.ops):
            op.desc.set_is_target(False)
            if op.type == "feed" or op.type == "fetch":
                need_to_remove_op_index.append(i)

        for index in need_to_remove_op_index[::-1]:
            global_block._remove_op(index)

        main_program.desc.flush()

        main_program = main_program._prune_with_input(
            feeded_var_names=feeded_var_names, targets=target_vars
        )
        main_program = main_program._inference_optimize(prune_read_op=True)
        fetch_var_names = [v.name for v in target_vars]

        for target_v in target_vars:
            if not main_program.global_block().has_var(target_v.name):
                main_program.global_block().create_var(
                    name=target_v.name,
                    shape=target_v.shape,
                    dtype=target_v.dtype,
                    persistable=target_v.persistable,
                )

        prepend_feed_ops(main_program, feeded_var_names)
        append_fetch_ops(main_program, fetch_var_names)

        with open(model_basename, "wb") as f:
            f.write(
                main_program._remove_training_info(
                    clip_extra=clip_extra
                ).desc.serialize_to_string()
            )
    else:
        # TODO(panyx0718): Save more information so that it can also be used
        # for training and more flexible post-processing.
        with open(model_basename + ".main_program", "wb") as f:
            f.write(
                main_program._remove_training_info(
                    clip_extra=clip_extra
                ).desc.serialize_to_string()
            )

    if program_only:
        warnings.warn(
            "save_inference_model specified the param `program_only` to True, It will not save params of Program."
        )
        return target_var_name_list

    main_program._copy_dist_param_info_from(origin_program)

    if params_filename is not None:
        params_filename = os.path.basename(params_filename)

    paddle.distributed.io.save_persistables(
        executor, save_dirname, main_program, params_filename
    )
    return target_var_name_list

#print numpy array like C structure       
def print_alike(arr, seperator_begin='{', seperator_end='}', verbose=False):
    shape = arr.shape
    rank = len(shape)

    #print("shape: ", shape, "rank: %d" %(rank))

    # for idx, value in np.ndenumerate(arr):
    #    print(idx, value)

    def print_array(arr, end=' '):
        shape = arr.shape
        rank = len(arr.shape)
        if rank > 1:
            line = seperator_begin
            for i in range(arr.shape[0]):
                line += print_array(
                    arr[i, :],
                    end=seperator_end +
                    ",\n" if i < arr.shape[0] - 1 else seperator_end)
            line += end
            return line
        else:
            line = seperator_begin
            for i in range(arr.shape[0]):
                line += "{:.2f}".format(arr[i])  # str(arr[i])
                line += ", " if i < shape[0] - 1 else ' '
            line += end
            # print(line)
            return line

    if verbose:
        print(print_array(arr, seperator_end))        

def saveModel(name, exe, feedkeys:list, fetchlist:list, inputs:list, outputs:list, target_dir:str, use_static_api=False):
    model_dir = os.path.join(target_dir, name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # print("\n\n------------- %s -----------\n" % (name))
    for i, input in enumerate(inputs):
        if use_static_api == True:
            feedkey = feedkeys[i].name
        else:
            feedkey = feedkeys[i]
        # print("INPUT %s :" % (feedkey), input.shape, input.dtype, "\n")
        # print_alike(input)
        np.save(os.path.join(model_dir, "input{}".format(i)), input)
        np.save(os.path.join(model_dir, "input{}.{}.{}".format(i, feedkey, input.dtype)), input)
    # print("\n")

    for i, output in enumerate(outputs):
        # print("OUTPUT %s :" % (fetchlist[i]),output.shape, output.dtype, "\n")
        # print_alike(output)
        np.save(os.path.join(model_dir, "output{}".format(i)), output)

    # composited model + scattered model
    if use_static_api == True:
        model_name = os.path.join(model_dir, name)
        paddle.static.io.save_inference_model(model_name, feedkeys, fetchlist, exe)
    else:
        save_inference_model(model_dir, feedkeys, fetchlist, exe)
        save_inference_model(model_dir, feedkeys, fetchlist, exe, model_filename=name+".pdmodel", params_filename=name+".pdiparams")


'''
export dyn model, along with input and output for reference.
input_data: list of all inputs
'''
def exportModel(name, dyn_func, input_data:list, target_dir:str, dyn_shapes:list=[]):
    model_dir = os.path.join(target_dir, name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    save_path = '{}/{}'.format(model_dir, name)

    if len(dyn_shapes)>0:
        assert(len(dyn_shapes) == len(input_data))

    input_specs = []
    for idx, data in enumerate(input_data):
        input_name = 'input{}'.format(idx)
        input_shape = dyn_shapes[idx] if len(dyn_shapes)>0 and dyn_shapes[idx] is not None else data.shape

        input_specs.append(
            paddle.static.InputSpec(shape=input_shape, dtype=data.dtype, name=input_name)
        )

        # dump input
        np.save(os.path.join(model_dir, "input{}".format(idx)), data)        

    paddle.jit.save(dyn_func, save_path, input_specs)
    print('saved exported model to {}'.format(save_path))

    # infer
    model = paddle.jit.load(save_path)

    result = model(*[input[:] for input in input_data])
   
    # dump output for reference
    if isinstance(result, (tuple, list)):
        for idx, out in enumerate(result):
            np.save(os.path.join(model_dir, "output{}".format(idx)), out.numpy())
    else:       
        np.save(os.path.join(model_dir, "output{}".format(0)), result.numpy())
    
    if paddle.__version__ < "2.6.0": 
        paddle.fluid.core.clear_executor_cache()
    else:
        paddle.base.core.clear_executor_cache()
    return result


if __name__ == "__main__":
    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)  

    #x = np.random.randn(2,3).astype(np.float32)
    x = np.array([[[
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [1, 2, 3],
        [4, 5, 6]
    ]], 
    [[
        [1, 2, 3],
        [4, 5, 6]
    ],
    [
        [1, 2, 3],
        [4, 5, 6]
    ]]]).astype(np.float32)
