#include "common_op_table.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "Eigen/Dense"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_matrix_band_part_op(const NodeContext& node) {
    default_op_checks(node, 3, {"MatrixBandPart"});
    auto input_matrix = node.get_input(0);
    auto num_lower = node.get_input(1);
    auto num_upper = node.get_input(2);
    auto node_name = node.get_name();

    // Convert num_lower and num_upper to integers (assuming they are scalar constants)
    int num_lower_value = as_scalar<int>(num_lower);
    int num_upper_value = as_scalar<int>(num_upper);

    // Create a mask to zero out elements outside the band
    MatrixXd mask = MatrixXd::Ones(input_matrix.get_shape());
    for (int i = 0; i < input_matrix.get_shape().at(0); ++i) {
        for (int j = 0; j < input_matrix.get_shape().at(1); ++j) {
            if (j < i - num_lower_value || j > i + num_upper_value) {
                mask(i, j) = 0.0;
            }
        }
    }

    // Apply the mask to the input matrix
    auto masked_matrix = make_shared<op::v1::Multiply>(input_matrix, make_constant(mask));

    set_node_name(node_name, masked_matrix.get_node_shared_ptr());

    return {masked_matrix};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
