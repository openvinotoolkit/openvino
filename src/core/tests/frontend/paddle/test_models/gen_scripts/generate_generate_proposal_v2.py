#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel
import sys

def generate_proposals_v2(name: str, input_data: dict, attr: dict):
    scores_np = input_data["scores"]
    bbox_deltas_np = input_data["bbox_deltas"]
    im_shape_np = input_data["im_shape"]
    anchors_np = input_data["anchors"]
    variances_np = input_data["variances"]

    pre_nms_top_n = attr["pre_nms_top_n"]
    post_nms_top_n = attr["post_nms_top_n"]
    nms_thresh = attr["nms_thresh"]
    min_size = attr["min_size"]
    pixel_offset = attr["pixel_offset"]

    import paddle
    import ppdet.modeling.ops as ops
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        scores = paddle.static.data(
            name='scores', shape=scores_np.shape, dtype='float32')            # [N, A, H, W]
        bbox_deltas = paddle.static.data(
            name='bbox_deltas', shape=bbox_deltas_np.shape, dtype='float32')  # [N, 4 * A, H, W]
        im_shape = paddle.static.data(
            name='im_shape', shape=im_shape_np.shape, dtype='float32')        # [N, 2]
        anchors = paddle.static.data(
            name='anchors', shape=anchors_np.shape, dtype='float32')          # [H, W, A, 4]
        variances = paddle.static.data(
            name='var', shape=variances_np.shape, dtype='float32')            # [H, W, A, 4]
        rois, roi_probs, rois_num = ops.generate_proposals(
            scores,
            bbox_deltas,
            im_shape,
            anchors,
            variances,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=nms_thresh,
            min_size=min_size,
            pixel_offset=pixel_offset,
            return_rois_num=True)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={
                'scores': scores_np,
                'bbox_deltas': bbox_deltas_np,
                'im_shape': im_shape_np,
                'anchors': anchors_np,
                'var': variances_np
            },
            fetch_list=[rois, roi_probs, rois_num],
            return_numpy=False)

        #print("-----------------------------")
        #print(outs[0])
        print("-----------------------------")
        print(outs[1])
        print("-----------------------------")
        #print(output_data['roi_probs'])
        #print("-----------------------------")
        #print(outs[2])
        ##print(outs[0].__array__())
        ##print(sys.modules[outs[0].__module__])
        #print("-----------------------------")

        # Save inputs in order of ngraph function, to facilite Fuzzy test, 
        # which accepts inputs and outputs in this order as well. 
        saveModel(name, exe, feedkeys=['scores', 'bbox_deltas', 'im_shape', 'anchors', 'var'],
                  fetchlist=[rois, roi_probs, rois_num],
                  inputs=[scores_np, bbox_deltas_np, im_shape_np, anchors_np, variances_np],
                  outputs=outs, target_dir=sys.argv[1])
        #test_data = dict()
        #test_data['roi_probs'] = np.load("/tmp/generate_proposals_v2_0/output1.npy")
        #print(np.load("/tmp/generate_proposals_v2/output0.npy"))
        #print(test_data['roi_probs'].shape)
        #print("-----------------------------")
        #print(output_data['roi_probs'].shape)
        #print(np.load("/tmp/generate_proposals_v2/output2.npy"))

if __name__ == "__main__":
    input_data = dict()
    attr = dict()

    # test case 0
    input_name = "generate_proposals_v2_0"
    input_data["scores"] = np.random.rand(1, 3, 4, 4).astype('float32')
    input_data["bbox_deltas"] = np.random.rand(1, 12, 4, 4).astype('float32')
    input_data["im_shape"] = np.array([[200, 200]]).astype('float32')
    input_data["anchors"] = np.reshape(np.arange(4 * 4 * 3 * 4),
                                        [4, 4, 3, 4]).astype('float32')
    input_data["variances"] = np.ones((4, 4, 3, 4)).astype('float32')

    attr["pre_nms_top_n"] = 6000
    attr["post_nms_top_n"] = 1000
    attr["nms_thresh"] = 0.699999988079071
    attr["min_size"] = 0
    attr["pixel_offset"] = False

    load_from_local_1 = False
    load_from_local_2 = False
    if (load_from_local_1):
        test_data = np.load('/home/iot/Downloads/tmp/gen.npy', allow_pickle=True)
        output_data = dict()
        test_arr = dict()
        input_arr = test_data.item().get('input')
        output_arr = test_data.item().get('output')
        param_arr = test_data.item().get('param')
        print(param_arr)
        input_data["scores"] = input_arr['Scores']
        input_data["anchors"] = input_arr['Anchors']
        input_data["bbox_deltas"] = input_arr['BboxDeltas']
        input_data["variances"] = input_arr['Variances']
        input_data["im_shape"] = input_arr['ImShape']
        print(input_data['scores'].shape)
        print(input_data['anchors'].shape)
        print(input_data['bbox_deltas'].shape)
        print(input_data['variances'].shape)
        print(input_data['im_shape'].shape)
        output_data["rois"] = output_arr['rois']
        output_data["roi_probs"] = output_arr['roi_probs']
        output_data["rois_num"] = output_arr['rois_num']
        print(output_data['rois'].shape)
        print(output_data['roi_probs'].shape)
        print(output_data['rois_num'].shape)
        attr["pre_nms_top_n"] = param_arr['pre_nms_topN']
        attr["post_nms_top_n"] = param_arr['post_nms_topN']
        attr["nms_thresh"] = param_arr['nms_thresh']
        attr["min_size"] = param_arr['min_size']
        attr["pixel_offset"] = param_arr['pixel_offset']
    elif (load_from_local_2):
        test_data = np.load('/home/iot/Downloads/tmp/paddle.npy', allow_pickle=True)
        input_data["scores"] = test_data.item().get('conv2d_100.tmp_1_slice_0')
        input_data["bbox_deltas"] = test_data.item().get('conv2d_101.tmp_1_slice_0')
        #input_data["anchors"] = np.reshape(test_data.item().get('reshape2_4.tmp_0'), [38, 38, 15, 4])
        input_data["anchors"] = test_data.item().get('reshape2_4.tmp_0')
        input_data["im_shape"] = test_data.item().get('im_shape_slice_0')
        input_data["variances"] = test_data.item().get('full_like_0.tmp_0')
        #input_data["variances"] = np.ones((38, 38, 15, 4)).astype('float32')
        # A = 15, H = 38, W = 38
        print(input_data["scores"].shape)
        print(input_data["bbox_deltas"].shape)
        print(input_data["anchors"].shape)
        print(input_data["im_shape"].shape)
        print(input_data["variances"].shape)
        #print(input_data["bbox_deltas"])

    generate_proposals_v2(input_name, input_data, attr)

    # test case 1
    input_name = "generate_proposals_v2_1"
    input_data["variances"] = np.ones((4 * 4 * 3, 4)).astype('float32')
    input_data["anchors"] = np.reshape(np.arange(4 * 4 * 3 * 4),
                                        [4 * 4 * 3, 4]).astype('float32')
    attr["min_size"] = 4
    attr["pixel_offset"] = True

    generate_proposals_v2(input_name, input_data, attr)

    # test case 2
    input_name = "generate_proposals_v2_2"

    bbox_deltas0 = np.random.rand(1, 12, 1, 4).astype('float32')
    bbox_deltas1 = np.random.rand(1, 12, 2, 4).astype('float32')
    input_data["bbox_deltas"] = np.concatenate((bbox_deltas0, bbox_deltas0, bbox_deltas1), axis = 2)
    #print(bbox_deltas0)
    #print("----------------------------")
    #print(bbox_deltas1)
    #print("----------------------------")
    #print(input_data["bbox_deltas"])

    anchors0 = np.reshape(np.arange(1 * 4 * 3 * 4),
                                    [1, 4, 3, 4]).astype('float32')
    anchors1 = np.reshape(np.arange(3 * 4 * 3 * 4),
                                    [3, 4, 3, 4]).astype('float32')
    input_data["anchors"] = np.concatenate((anchors0, anchors1), axis = 0)
    #print(anchors0)
    #print("----------------------------")
    #print(anchors1)
    #print("----------------------------")
    #print(input_data["anchors"])

    attr["nms_thresh"] = 0.5

    generate_proposals_v2(input_name, input_data, attr)

    # test case 3
    input_name = "generate_proposals_v2_3"
    attr["nms_thresh"] = 0.7

    generate_proposals_v2(input_name, input_data, attr)

    # test case 4
    input_name = "generate_proposals_v2_4"
    variances_0 = np.ones((11, 4)).astype('float32') * 0.5
    variances_1 = np.ones((37, 4)).astype('float32')
    input_data["variances"] = np.concatenate((variances_0, variances_1), axis = 0)

    generate_proposals_v2(input_name, input_data, attr)

    # test case 5
    input_name = "generate_proposals_v2_5"
    input_data["scores"] = np.random.rand(1, 6, 10, 8).astype('float32')
    input_data["bbox_deltas"] = np.random.rand(1, 24, 10, 8).astype('float32')
    input_data["im_shape"] = np.array([[1000, 1000]]).astype('float32')
    input_data["anchors"] = np.reshape(np.arange(10 * 8 * 6 * 4),
                                        [10, 8, 6, 4]).astype('float32')
    input_data["variances"] = np.ones((10, 8, 6, 4)).astype('float32')

    attr["pre_nms_top_n"] = 100
    attr["post_nms_top_n"] = 60

    generate_proposals_v2(input_name, input_data, attr)

    # test case 6
    input_name = "generate_proposals_v2_6"
    input_data["scores"] = np.random.rand(2, 6, 10, 8).astype('float32')
    input_data["bbox_deltas"] = np.random.rand(2, 24, 10, 8).astype('float32')
    input_data["im_shape"] = np.array([[1000, 1000]] * 2).astype('float32')
    input_data["anchors"] = np.reshape(np.arange(10 * 8 * 6 * 4),
                                        [10, 8, 6, 4]).astype('float32')
    input_data["variances"] = np.ones((10, 8, 6, 4)).astype('float32')

    attr["pre_nms_top_n"] = 100
    attr["post_nms_top_n"] = 60

    generate_proposals_v2(input_name, input_data, attr)