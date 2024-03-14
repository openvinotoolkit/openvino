#include "common_op_table.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"
#include "openvino/frontend/tensorflow/variable.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_approximate_equal_op(const NodeContext& node) {
    default_op_checks(node, 2, {"ApproximateEqual"});
    auto x = node.get_input(0);
    auto y = node.get_input(1);
    auto tolerance = make_shared<v0::Constant>(element::i32, Shape{}, node.get_attribute<float>("tolerance"));
    // Implement the logic for ApproximateEqual
    auto difference = make_shared<v1::Subtract>(x, y);
    auto absolute = make_shared<v0::Abs>(difference);
    auto is_less = make_shared<v1::Less>(absolute, tolerance);

    // Create and return the corresponding OpenVINO operation
    set_node_name(node.get_name(), is_less);
    return {is_less};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov