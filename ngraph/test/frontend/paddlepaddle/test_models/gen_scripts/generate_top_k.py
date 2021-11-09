#
# top_k paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd
import paddle.fluid as fluid
import sys

data_type = 'float32'

def top_k(name : str, x, k:int):
    
    pdpd.enable_static()
    
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype='float32')
        value, indices = fluid.layers.topk(node_x, k=k, name="top_k")
        indices = pdpd.cast(indices, np.float32)
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
            feed={'x': x},
            fetch_list=[value,indices])
            
        saveModel(name, exe, feedkeys=['x'], fetchlist=[value,indices], inputs=[x], outputs=outs, target_dir=sys.argv[1])

    return outs[0]

def main():
    data = np.random.random([2,3,8]).astype("float32")
    k = 3
    top_k("top_k", data, k=k)


if __name__ == "__main__":
    main()     
