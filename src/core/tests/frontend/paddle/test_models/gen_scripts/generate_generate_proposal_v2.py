#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel
import sys

def generate_proposals_v2(name: str, ctx: dict):
    scores_np = np.random.rand(1, 3, 4, 4).astype('float32')
    bbox_deltas_np = np.random.rand(1, 12, 4, 4).astype('float32')
    im_shape_np = np.array([[200, 200]]).astype('float32')
    anchors_np = np.reshape(np.arange(4 * 4 * 3 * 4),
                            [4, 4, 3, 4]).astype('float32')
    variances_np = np.ones((4, 4, 3, 4)).astype('float32')

    #scores_np = ctx.scores
    #bbox_deltas_np = ctx.bbox_deltas
    #im_shape_np = ctx.im_shape
    #anchors_np = ctx.anchors
    #variances_np = ctx.variances

    import paddle
    import ppdet.modeling.ops as ops
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        scores = paddle.static.data(
            name='scores', shape=[1, 3, 4, 4], dtype='float32')         # [N, A, H, W]
        bbox_deltas = paddle.static.data(
            name='bbox_deltas', shape=[1, 12, 4, 4], dtype='float32')   # [N, 4 * A, H, W]
        im_shape = paddle.static.data(
            name='im_shape', shape=[1, 2], dtype='float32')             # [N, 2]
        anchors = paddle.static.data(
            name='anchors', shape=[4, 4, 3, 4], dtype='float32')        # [H, W, A, 4]
        variances = paddle.static.data(
            name='var', shape=[4, 4, 3, 4], dtype='float32')            # [H, W, A, 4]
        rois, roi_probs, rois_num = ops.generate_proposals(
            scores,
            bbox_deltas,
            im_shape,
            anchors,
            variances,
            pre_nms_top_n=40,
            post_nms_top_n=35,
            nms_thresh=0.5,
            min_size=3,
            pixel_offset=False,
            return_rois_num=True)
            #min_size=3,

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
        #print(np.load("/tmp/generate_proposals_v2/output0.npy"))
        #print(np.load("/tmp/generate_proposals_v2/output1.npy"))
        #print(np.load("/tmp/generate_proposals_v2/output2.npy"))

if __name__ == "__main__":
    generate_proposals_v2("generate_proposals_v2", dict())
