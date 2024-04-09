import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.framework import ops
from translate_matrix_set_diag_v3_op import translate_matrix_set_diag_v3_op


def create_node_context(input_shape, input_type, diagonal):
    node_context = ops.NodeDef()
    node_context.name = "MatrixSetDiagV3Test"
    node_context.op = "MatrixSetDiagV3"
    node_context.input.extend(["input:0", "diagonal:0"])
    node_context.attr["diagonal_shape"].list.shape.extend([
        ops.TensorShapeProto(dim=[ops.TensorShapeProto.Dim(size=d) for d in diagonal.shape])
    ])
    return node_context


@pytest.mark.parametrize("input_shape", [(5, 5), (3, 3), (2, 4)])
@pytest.mark.parametrize("input_type", [np.float32, np.float64])
@pytest.mark.parametrize("diagonal_length", [3, 5, 7])
def test_matrix_set_diag_v3_op(input_shape, input_type, diagonal_length):
    
    diagonal = np.arange(diagonal_length, dtype=input_type)

    
    node_context = create_node_context(input_shape, input_type, diagonal)

 
    output_nodes = translate_matrix_set_diag_v3_op(node_context)

   
    assert len(output_nodes) == 1 


if __name__ == "__main__":
    pytest.main()
