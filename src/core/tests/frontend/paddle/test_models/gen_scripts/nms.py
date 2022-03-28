#
# multiclass_nms paddle model generator
#
import os
import numpy as np
import copy  # deepcopy
import sys

from save_model import saveModel

# print numpy array like vector array
# this is to faciliate some unit test, e.g. ngraph op unit test.


def print_alike(arr, seperator_begin='{', seperator_end='}'):
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

    print(print_array(arr, seperator_end))


# bboxes shape (N, M, 4)
# scores shape (N, C, M)
def NMS(name: str, bboxes, scores, attrs: dict, rois_num=None, quite=True):
    import paddle as pdpd
    from ppdet.modeling import ops
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(),
                                   pdpd.static.Program()):
        node_boxes = pdpd.static.data(name='bboxes',
                                      shape=bboxes.shape,
                                      dtype=bboxes.dtype,
                                      lod_level=1)
        node_scores = pdpd.static.data(name='scores',
                                       shape=scores.shape,
                                       dtype=scores.dtype,
                                       lod_level=1)

        node_rois_num = None
        if rois_num is not None:
            node_rois_num = pdpd.static.data(name='rois_num',
                                        shape=rois_num.shape,
                                        dtype=rois_num.dtype,
                                        lod_level=1)

        if attrs['nms_type'] is 'multiclass_nms3':
            nms_outputs = ops.multiclass_nms(bboxes=node_boxes,
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
            nms_outputs = ops.matrix_nms(bboxes=node_boxes,
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
                if x.dtype==pdpd.int32 or x.dtype==pdpd.int64:
                    x = pdpd.cast(x, "float32")
            output.append(x)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        fetch_vars = [x for x in output if x is not None]
        feed_dict = {'bboxes': bboxes, 'scores': scores}
        if rois_num is not None:
             feed_dict['rois_num'] = rois_num

        output_lod = exe.run(feed=feed_dict,
                             fetch_list=fetch_vars,
                             return_numpy=False)

        # There is a bug in paddledet that dtype of model var mismatch its output LodTensor.
        # Specifically, it is 'Index' is 'int64', while its LodTensor of 'int32'.
        # This will lead to a failure in ngraph frontend op fuzzy test.
        # So here is an workaround to align the dtypes.
        out = np.array(output_lod.pop(0))
        nms_rois_num = np.array(
            output_lod.pop(0)) if output[1] is not None else None
        index = np.array(output_lod.pop(0)).astype(pdpd.fluid.data_feeder.convert_dtype(
            output[2].dtype)) if output[2] is not None else None

        # Save inputs in order of ngraph function, to facilite Fuzzy test,
        # which accepts inputs and outputs in this order as well.
        output_np = [out, nms_rois_num, index]
        saveModel(name,
                  exe,
                  feedkeys=list(feed_dict.keys()),
                  fetchlist=fetch_vars,
                  inputs=list(feed_dict.values()),
                  outputs=[x for x in output_np if x is not None],
                  target_dir=sys.argv[1])

    if quite is False:
        # input
        print('\033[94m' + 'bboxes: {}'.format(bboxes.shape) + '\033[0m')
        print_alike(bboxes, seperator_begin='', seperator_end='')
        print('\033[94m' + 'scores: {}'.format(scores.shape) + '\033[0m')
        print_alike(scores, seperator_begin='', seperator_end='')

        # output
        print('\033[91m' + 'out_np: {}'.format(out.shape) + '\033[0m')
        print_alike(out, seperator_begin='', seperator_end='')
        print('\033[91m' + 'nms_rois_num_np: {}'.format(nms_rois_num.shape) +
              '\033[0m')
        print_alike(nms_rois_num, seperator_begin='', seperator_end='')
        if index is not None:
            print('\033[91m' + 'index_np: {}'.format(index.shape) + '\033[0m')
            print_alike(index, seperator_begin='', seperator_end='')
