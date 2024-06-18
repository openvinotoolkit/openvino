# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# helper for multiclass/matrix_nms paddle model generator
#
import os
import numpy as np
import copy  # deepcopy
import sys
import paddle

from save_model import saveModel, exportModel, print_alike

if paddle.__version__ >= '2.6.0':
    from paddle.base import data_feeder
else:
    from paddle.fluid import data_feeder


# bboxes shape (N, M, 4) if shared else (M, C, 4)
# scores shape (N, C, M) if shared else (M, C)
def NMS(name: str, bboxes, scores, attrs: dict, rois_num=None, verbose=False):
    import paddle
    from ops import multiclass_nms as multiclass_nms
    from ops import matrix_nms as matrix_nms
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(),
                                   paddle.static.Program()):
        # make model with inputs of dynamic shape
        node_boxes = paddle.static.data(name='bboxes',
                                      shape=[-1, -1, 4],
                                      dtype=bboxes.dtype,
                                      lod_level=1)
        node_scores = paddle.static.data(name='scores',
                                       shape=[-1] * len(scores.shape),
                                       dtype=scores.dtype,
                                       lod_level=1)

        node_rois_num = None
        if rois_num is not None:
            node_rois_num = paddle.static.data(name='rois_num',
                                        shape=rois_num.shape,
                                        dtype=rois_num.dtype,
                                        lod_level=1)

        if attrs['nms_type'] is 'multiclass_nms3':
            nms_outputs = multiclass_nms(bboxes=node_boxes,
                                             scores=node_scores,
                                             background_label=attrs['background_label'],
                                             score_threshold=attrs['score_threshold'],
                                             nms_top_k=attrs['nms_top_k'],
                                             nms_threshold=attrs['nms_threshold'],
                                             keep_top_k=attrs['keep_top_k'],
                                             normalized=attrs['normalized'],
                                             nms_eta=attrs['nms_eta'],
                                             return_index=attrs['return_index'],
                                             return_rois_num=True,
                                             rois_num=node_rois_num)
        else:
            nms_outputs = matrix_nms(bboxes=node_boxes,
                                         scores=node_scores,
                                         score_threshold=attrs['score_threshold'],
                                         post_threshold=attrs['post_threshold'],
                                         nms_top_k=attrs['nms_top_k'],
                                         keep_top_k=attrs['keep_top_k'],
                                         use_gaussian=attrs['use_gaussian'],
                                         gaussian_sigma=attrs['gaussian_sigma'],
                                         background_label=attrs['background_label'],
                                         normalized=attrs['normalized'],
                                         return_index=attrs['return_index'],
                                         return_rois_num=attrs['return_rois_num'])
        # output of NMS is mix of int and float. To make it easy for op_fuzzy unittest, cast int output to float.
        output = []
        for x in nms_outputs:
            if x is not None:
                if x.dtype==paddle.int32 or x.dtype==paddle.int64:
                    x = paddle.cast(x, "float32")
            output.append(x)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        fetch_vars = [x for x in output if x is not None]
        feed_dict = {'bboxes': bboxes, 'scores': scores}
        if rois_num is not None:
             feed_dict['rois_num'] = rois_num

        output_lod = exe.run(feed=feed_dict,
                             fetch_list=fetch_vars,
                             return_numpy=False)

        # There is a bug in paddledet that dtype of model var mismatch its output LodTensor.
        # Specifically, it is 'Index' is 'int64', while its LodTensor of 'int32'.
        # This will lead to a failure in OpenVINO frontend op fuzzy test.
        # So here is an workaround to align the dtypes.
        out = np.array(output_lod.pop(0))
        nms_rois_num = np.array(
            output_lod.pop(0)) if output[1] is not None else None
        if paddle.__version__ >= '2.0.0':
            index = np.array(output_lod.pop(0)).astype(data_feeder.convert_dtype(
                output[2].dtype)) if output[2] is not None else None
        else:
            index = np.array(output_lod.pop(0)).astype(data_feeder.convert_dtype(
                output[2].dtype)) if output[2] is not None else None
            
        feed_vars = [node_boxes, node_scores]
        if node_rois_num is not None:
            feed_vars.append(node_rois_num)
        # Save inputs in order of OpenVINO model, to facilite Fuzzy test,
        # which accepts inputs and outputs in this order as well.
        output_np = [out, nms_rois_num, index]
        saveModel(name,
                  exe,
                  feed_vars=feed_vars,
                  fetchlist=fetch_vars,
                  inputs=list(feed_dict.values()),
                  outputs=[x for x in output_np if x is not None],
                  target_dir=sys.argv[1])

    if verbose:
        # input
        print('\033[94m' + 'bboxes: {}'.format(bboxes.shape) + '\033[0m')
        print_alike(bboxes, seperator_begin='', seperator_end='', verbose=True)
        print('\033[94m' + 'scores: {}'.format(scores.shape) + '\033[0m')
        print_alike(scores, seperator_begin='', seperator_end='', verbose=True)

        # output
        print('\033[91m' + 'out_np: {}'.format(out.shape) + '\033[0m')
        print_alike(out, seperator_begin='', seperator_end='', verbose=True)
        print('\033[91m' + 'nms_rois_num_np: {}'.format(nms_rois_num.shape) +
              '\033[0m')
        print_alike(nms_rois_num, seperator_begin='', seperator_end='', verbose=True)
        if index is not None:
            print('\033[91m' + 'index_np: {}'.format(index.shape) + '\033[0m')
            print_alike(index, seperator_begin='', seperator_end='', verbose=True)
