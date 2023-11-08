# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# paddle model generator
# for lookup_table_v2
# https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Embedding_cn.html#embedding
# equal to "gather"
#
import numpy as np
import sys

from save_model import saveModel


def ngraph_embedding(ids, vocab_embeddings, vocab_size, embedding_dim, padding_idx, sparse):
    """
    decomposing embedding with OpenVINO ops.
    """
    import ngraph as ng
    from ngraph import opset8 as opset
    from openvino.inference_engine import IECore

    if vocab_embeddings is None:
        #
        vocab_embeddings = np.zeros(
            (vocab_size, embedding_dim)).astype("float32")

    node_ids = ng.parameter(shape=ids.shape, name='ids', dtype=ids.dtype)
    node_w = ng.parameter(shape=vocab_embeddings.shape,
                          name='w', dtype=vocab_embeddings.dtype)

    if padding_idx == -1:
        padding_idx += vocab_size

    if padding_idx is not None:
        '''
        mask W
        '''
        masked_embeddings = np.ones(vocab_embeddings.shape, dtype='int64')
        masked_embeddings[padding_idx, :] = 0  # mask

        node_mask = ng.constant(
            masked_embeddings, name='mask', dtype=vocab_embeddings.dtype)
        node_masked_w = ng.multiply(node_w, node_mask)

    node_axis = ng.constant([0], name='const0', dtype=np.int64)
    node_gather = opset.gather(data=node_masked_w if padding_idx else node_w,
                               indices=node_ids, axis=node_axis, batch_dims=0)

    graph = ng.result(node_gather, name='y')

    parameters = [node_ids, node_w]
    inputs_dict = {'ids': ids, "w": vocab_embeddings}

    #
    function = ng.Function(graph, parameters, "embedding")

    ie_network = ng.function_to_cnn(function)
    ie = IECore()
    executable_network = ie.load_network(ie_network, 'CPU')
    output = executable_network.infer(inputs_dict)

    return output


def embedding(name: str, ids, vocab_size, embedding_dim, padding_idx=None, sparse=False, vocab_embeddings=None, compare=False):
    """
    padding_idx (int|long|None) 
    """
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_ids = paddle.static.data(
            name='Ids', shape=ids.shape, dtype=ids.dtype)

        pretrained_attr = paddle.ParamAttr(name='W',
                                           initializer=paddle.nn.initializer.Assign(
                                               vocab_embeddings),
                                           trainable=False) if vocab_embeddings is not None else None

        node_embedding = paddle.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim,
                                             padding_idx=padding_idx, sparse=sparse, weight_attr=pretrained_attr, name=name)
        node_out = node_embedding(node_ids)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        input_dict = {'Ids': ids}
        output_vars_list = [node_out]

        infer_results = exe.run(
            feed=input_dict,
            fetch_list=output_vars_list)

        saveModel(name, exe, feedkeys=[node_ids], fetchlist=output_vars_list, inputs=list(
            input_dict.values()), outputs=infer_results, target_dir=sys.argv[1], use_static_api=True)

        #
        outputs = dict()
        for i in range(len(infer_results)):
            outputs[output_vars_list[i].name] = infer_results[i]

    #
    if compare:
        ng_result = ngraph_embedding(
            ids, vocab_embeddings, vocab_size, embedding_dim, padding_idx, sparse)

        ng_result = list(ng_result.values())[0]
        paddle_result = list(outputs.values())[0]

        match = np.all(np.isclose(
            paddle_result, ng_result, rtol=1e-4, atol=1e-5))

        prefix_color = '\n\033[92m' if match else '\n\033[91m'
        print(prefix_color
              + 'TestCase {} Result {} '.format(name, match) + '\033[0m\n')

        if not match:
            np.set_printoptions(precision=2)
            np.set_printoptions(suppress=True)

            print(prefix_color
                  + 'paddle_result: {}'.format(paddle_result) + '\033[0m\n')
            print(prefix_color
                  + 'ng_result: {}'.format(ng_result) + '\033[0m\n')

            raise ValueError(name + ': OV result does not match paddle!')

    return outputs


if __name__ == "__main__":
    vocab_size = 17
    embedding_dim = 31

    table = np.random.random((vocab_size, embedding_dim)).astype("float32")

    #
    ids = np.random.randint(0, vocab_size, 4).astype("int32")
    embedding("embedding_0", ids, vocab_size, embedding_dim,
              vocab_embeddings=table, compare=False)

    #
    ids = np.random.randint(0, vocab_size, 4).astype("int32")
    embedding("embedding_sparse", ids, vocab_size, embedding_dim,
              sparse=True, vocab_embeddings=table, compare=False)

    # # compare fail
    ids = np.random.randint(0, vocab_size, 4).astype("int32")
    embedding("embedding_none_weight", ids,
              vocab_size, embedding_dim, compare=False)

    #
    ids = np.random.randint(0, vocab_size, 4).astype("int32")
    ids = np.squeeze(ids)
    padding_idx = np.random.choice(ids, 1)[0]
    # print('padding_idx {}, ids {}'.format(padding_idx, ids))
    outputs = embedding("embedding_paddings", ids, vocab_size, embedding_dim, padding_idx=int(
        padding_idx), vocab_embeddings=table, compare=False)
    # print('outputs {}'.format(outputs))

    # corner case
    ids = np.random.randint(0, vocab_size, 4).astype("int32")
    pick = np.random.choice(4, 1)[0]  # pick randomly to be max vacab_size -1
    ids[pick] = vocab_size - 1
    padding_idx = -1
    # print('padding_idx {}, ids {}'.format(padding_idx, ids))
    outputs = embedding("embedding_paddings_neg1", ids, vocab_size, embedding_dim,
                        padding_idx=int(padding_idx), vocab_embeddings=table, compare=False)
    # print('outputs {}'.format(outputs))

    #
    ids = np.random.randint(low=0, high=vocab_size,
                            size=(2, 4, 5)).astype("int32")
    embedding("embedding_tensorIds", ids, vocab_size,
              embedding_dim, vocab_embeddings=table, compare=False)

    #
    ids = np.random.randint(low=0, high=vocab_size,
                            size=(2, 4, 5)).astype("int32")
    flatten_idx = ids.flatten()
    padding_idx = np.random.choice(flatten_idx, 1)[0]
    # print('padding_idx {}'.format(padding_idx))

    if paddle.__version__ >= '2.0.0':
        outputs = embedding("embedding_tensorIds_paddings", ids, vocab_size, embedding_dim,
                            padding_idx=np.compat.long(padding_idx), vocab_embeddings=table, compare=False)
    else:
        import paddle.compat as cpt
        outputs = embedding("embedding_tensorIds_paddings", ids, vocab_size, embedding_dim,
                            padding_idx=cpt.long_type(padding_idx), vocab_embeddings=table, compare=False)
