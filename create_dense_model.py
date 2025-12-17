import openvino as ov
import numpy as np

def create_dense_model():
    input_dim = 1024
    output_dim = 1024
    
    param = ov.opset10.parameter([1, input_dim], np.float32, name="input")
    weights = np.random.rand(input_dim, output_dim).astype(np.float32)
    const_weights = ov.opset10.constant(weights)
    
    # MatMul
    matmul = ov.opset10.matmul(param, const_weights, transpose_a=False, transpose_b=False)
    matmul.set_friendly_name("dense_layer")
    
    result = ov.opset10.result(matmul)
    model = ov.Model([result], [param], "dense_model")
    
    ov.save_model(model, "dense_model.xml")
    print("Saved dense_model.xml")

if __name__ == "__main__":
    create_dense_model()
