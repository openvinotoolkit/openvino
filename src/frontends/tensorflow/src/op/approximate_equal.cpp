#include "common_op_table.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"

using namespace std;
using namespace ov;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_approximate_equal_op(const NodeContext& node) {
    auto op_type = node.get_op_type();
    TENSORFLOW_OP_VALIDATION(node, op_type == "ApproximateEqual", "Incorrect usage of translate_approximate_equal_op.");

    // Extract necessary attributes and inputs
    auto tolerance = node.get_attribute<float>("tolerance");
    OutputVector inputs;
    for (size_t input_ind = 0; input_ind < node.get_input_size(); ++input_ind) {
        inputs.push_back(node.get_input(input_ind));
    }

    // Implement the logic for ApproximateEqual
    auto difference = make_shared<Subtract>(inputs[0], inputs[1]);
    auto absolute = make_shared<Abs>(difference);
    auto less = make_shared<Less>(absolute, Constant::create(element::f32, Shape{}, {tolerance}));

    // Create and return the corresponding OpenVINO operation
    set_node_name(node.get_name(), less);
    return {less};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
