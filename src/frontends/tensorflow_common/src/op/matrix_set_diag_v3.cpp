#include "openvino/op/matmul.hpp"
#include "openvino/op/matmul.hpp"  
#include "openvino/op/convert.hpp"  
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace tf {
namespace op {
using namespace ov::op;

OutputVector translate_matrix_set_diag_v3_op(const NodeContext& context) {
    // Validate operation type
    auto op_type = context.get_op_type();
    TENSORFLOW_OP_VALIDATION(context, op_type == "MatrixSetDiagV3",
                             "Invalid operation type encountered in translate_matrix_set_diag_v3_op.");

    // Extract attributes and inputs
    auto diagonal = context.get_input(0);
    auto diagonal_shape = diagonal.get_shape();
    auto matrix = context.get_input(1);
    auto matrix_shape = matrix.get_shape();

    // Validate input shapes
    TENSORFLOW_OP_VALIDATION(context, diagonal_shape.size() == 1 && matrix_shape.size() >= 2,
                             "Invalid input shapes for MatrixSetDiagV3 operation.");

    // Perform MatrixSetDiagV3 operation
    auto result = context.mark_node(std::make_shared<opset::MatMul>(diagonal, matrix));

    return {result};
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
