#
# gather_tree paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd
import sys

def gather_tree(name:str, ids, parents, dtype):
    pdpd.enable_static()
    
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_ids = pdpd.static.data(name='ids', shape=ids.shape, dtype = dtype)
        node_parents = pdpd.static.data(name='parents', shape=parents.shape, dtype = dtype)
        out = pdpd.nn.functional.gather_tree(node_ids, node_parents)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'ids': ids, 'parents': parents},
            fetch_list=[out])
        saveModel(name, exe, feedkeys=['ids', 'parents'], fetchlist=[out], inputs=[ids, parents], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    ids_int32 = np.array([[[2, 2], [6, 1]], [[3, 9], [6, 1]], [[0, 1], [9, 0]]]).astype("int32")
    parents_int32 = np.array([[[0, 0], [1, 1]], [[1, 0], [1, 0]], [[0, 0], [0, 1]]]).astype("int32")
    gather_tree("gather_tree_i32", ids_int32, parents_int32, dtype='int32')

    ids_int64 = np.array([[[2, 2], [6, 1]], [[3, 9], [6, 1]], [[0, 1], [9, 0]]]).astype("int64")
    parents_int64 = np.array([[[0, 0], [1, 1]], [[1, 0], [1, 0]], [[0, 0], [0, 1]]]).astype("int64")
    gather_tree("gather_tree_i64", ids_int64, parents_int64, dtype='int64')

if __name__ == "__main__":
    main()