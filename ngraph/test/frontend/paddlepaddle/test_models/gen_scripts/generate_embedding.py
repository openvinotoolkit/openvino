#
# paddle model generator
# for lookup_table_v2
# https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/Embedding_cn.html#embedding
# equal to "gather"
# 
import numpy as np
import sys

from save_model import saveModel


def embedding(name : str, ids, vocab_size, embedding_dim, padding_idx=None, sparse=False, vocab_embeddings=None):
    """
    padding_idx (int|long|None) 
    """
    import paddle
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_ids = paddle.static.data(name = 'Ids', shape = ids.shape, dtype = ids.dtype)

        pretrained_attr = paddle.ParamAttr(name='W',
                                   initializer=paddle.nn.initializer.Assign(vocab_embeddings),
                                   trainable=False) if vocab_embeddings is not None else None

        node_embedding = paddle.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx, sparse=sparse, weight_attr=pretrained_attr, name=name)
        node_out = node_embedding(node_ids)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        
        input_dict = {'Ids': ids}
        output_vars_list = [node_out]

        infer_results = exe.run(
                    feed=input_dict,
                    fetch_list=output_vars_list )

        print(node_out, type(infer_results), type(infer_results[0]))

        saveModel(name, exe, feedkeys=list(input_dict.keys()), fetchlist=output_vars_list, inputs=list(input_dict.values()), outputs=infer_results, target_dir=sys.argv[1])

        #
        outputs = dict()
        for i in range(len(infer_results)):
            outputs[output_vars_list[i].name] = infer_results[i]
    
    return outputs
    

if __name__ == "__main__":
    import paddle.compat as cpt
    vocab_size = 17
    embedding_dim = 31

    table = np.random.random((vocab_size, embedding_dim)).astype("float64")

    #
    ids = np.random.randint(0, vocab_size, 4).astype("int32")
    embedding("embedding_0", ids, vocab_size, embedding_dim, vocab_embeddings=table)

    #
    ids = np.random.randint(0, vocab_size, 4).astype("int32")
    embedding("embedding_sparse", ids, vocab_size, embedding_dim, sparse=True, vocab_embeddings=table)    

    #
    ids = np.random.randint(0, vocab_size, 4).astype("int32")
    embedding("embedding_none_weight", ids, vocab_size, embedding_dim)

    #
    ids = np.random.randint(0, vocab_size, 4).astype("int32")
    ids = np.squeeze(ids)
    padding_idx = np.random.choice(ids, 1)[0]
    print('padding_idx {}, ids {}'.format(padding_idx, ids))
    outputs = embedding("embedding_paddings", ids, vocab_size, embedding_dim, padding_idx=int(padding_idx), vocab_embeddings=table)
    print('outputs {}'.format(outputs))

    # corner case
    ids = np.random.randint(0, vocab_size, 4).astype("int32")
    pick = np.random.choice(4, 1)[0] # pick randomly to be max vacab_size -1
    ids[pick] = vocab_size-1
    padding_idx = -1
    print('padding_idx {}, ids {}'.format(padding_idx, ids))
    outputs = embedding("embedding_paddings_neg1", ids, vocab_size, embedding_dim, padding_idx=int(padding_idx), vocab_embeddings=table)
    print('outputs {}'.format(outputs))    

    #
    ids = np.random.randint(low=0, high=vocab_size, size=(2, 4, 5)).astype("int32")
    embedding("embedding_tensorIds", ids, vocab_size, embedding_dim, vocab_embeddings=table)
    
    #
    ids = np.random.randint(low=0, high=vocab_size, size=(2, 4, 5)).astype("int32")
    flatten_idx = ids.flatten()
    padding_idx = np.random.choice(flatten_idx, 1)[0]
    print('padding_idx {}'.format(padding_idx))
    outputs = embedding("embedding_tensorIds_paddings", ids, vocab_size, embedding_dim, padding_idx=cpt.long_type(padding_idx), vocab_embeddings=table)
    print('outputs {}'.format(outputs))
   