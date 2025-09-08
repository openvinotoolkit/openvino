import sys

import numpy as np
import paddle
from ops import box_coder

from save_model import exportModel, saveModel


def test_box_coder(name: str, prior_box, prior_box_var, target_box, code_type, box_normalized, axis, dtype):
    paddle.enable_static()
    is_tensor = not isinstance(prior_box_var, list)
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):

        prior_box_decode = paddle.static.data(name='prior_box',
                                              shape=prior_box.shape,
                                              dtype=dtype)
        target_box_decode = paddle.static.data(name='target_box',
                                               shape=target_box.shape,
                                               dtype=dtype)
        if is_tensor:
            prior_box_var_decode = paddle.static.data(name='prior_box_var',
                                                      shape=prior_box_var.shape,
                                                      dtype=dtype)
        else:
            prior_box_var_decode = prior_box_var

        out = box_coder(prior_box=prior_box_decode,
                        prior_box_var=prior_box_var_decode,
                        target_box=target_box_decode,
                        code_type=code_type,
                        box_normalized=box_normalized,
                        axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        feed_dict = {'prior_box': prior_box, 'target_box': target_box}
        if is_tensor:
            feed_dict['prior_box_var'] = prior_box_var

        outs = exe.run(
            feed=feed_dict,
            fetch_list=[out])
        
        feed_vars = [prior_box_decode, target_box_decode]
        if is_tensor:
            feed_vars.append(prior_box_var_decode)

        saveModel(name, exe, feed_vars=feed_vars, fetchlist=[out], inputs=[*feed_dict.values()],
                  outputs=[outs[0]],
                  target_dir=sys.argv[1])

    return outs[0]


def main():
    # For decode
    datatype = "float32"
    prior_box = np.random.random([8, 4]).astype(datatype)
    target_box = np.random.random([8, 4, 4]).astype(datatype)
    prior_box_var = [0.1, 0.1, 0.1, 0.1]
    code_type = "decode_center_size"
    box_normalized = True
    axis = 1
    paddle_out = test_box_coder("box_coder_1", prior_box, prior_box_var, target_box, code_type, box_normalized, axis,
                                datatype)

    axis = 0
    prior_box = np.random.random([4, 4]).astype(datatype)
    box_normalized = False
    prior_box_var = np.repeat(
        np.array([[0.1, 0.2, 0.1, 0.1]], dtype=np.float32), prior_box.shape[0], axis=0)
    paddle_out = test_box_coder("box_coder_2", prior_box, prior_box_var, target_box, code_type, box_normalized, axis,
                                datatype)

    box_normalized = True
    paddle_out = test_box_coder("box_coder_3", prior_box, prior_box_var, target_box, code_type, box_normalized, axis,
                                datatype)


def box_coder_dygraph():
    paddle.disable_static()

    @paddle.jit.to_static
    def test_model_1(prior_box, target_box):
        prior_box_var = [0.1, 0.1, 0.1, 0.1]
        code_type = "decode_center_size"
        box_normalized = True
        axis = 1
        out = box_coder(prior_box, prior_box_var, target_box,
                        code_type, box_normalized, axis=axis)
        return out

    datatype = "float32"
    prior_box = paddle.rand(shape=[2, 4], dtype=datatype)
    target_box = paddle.rand(shape=[2, 4, 4], dtype=datatype)
    exportModel("box_coder_dygraph_1", test_model_1, [
                prior_box, target_box], target_dir=sys.argv[1])

    @paddle.jit.to_static
    def test_model_2(prior_box, target_box, prior_box_var):
        code_type = "decode_center_size"
        box_normalized = True
        axis = 0
        out = box_coder(prior_box, prior_box_var, target_box,
                        code_type, box_normalized, axis=axis)
        return out

    datatype = "float32"
    prior_box = paddle.rand(shape=[4, 4], dtype=datatype)
    target_box = paddle.rand(shape=[8, 4, 4], dtype=datatype)
    prior_box_var = paddle.tile(paddle.to_tensor([[0.1, 0.2, 0.1, 0.1]], dtype=datatype), [prior_box.shape[0], 1])

    exportModel("box_coder_dygraph_2", test_model_2, [
                prior_box, target_box, prior_box_var], target_dir=sys.argv[1])


if __name__ == "__main__":
    main()
    # box_coder_dygraph()
