#include "common_op_table.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_tensor_scatter_update_op(const NodeContext& node) {
    default_op_checks(node, 3, {"TensorScatterUpdate"});
    auto tensor = node.get_input(0);
    auto indices = node.get_input(1);
    auto updates = node.get_input(2);
   

    auto indices_rank = indices.get_partial_shape().rank().get_length();
    auto tensor_rank = indices.get_partial_shape().rank().get_length();

    if(indices_rank!= tensor_rank){
        throw runtime_error("Indices rank must match Tensor rank.");
    }

    auto ones = create_constant_of_shape(indices, 1);
    auto scatter_nd = make_shared<v3::ScatterElementsUpdate>(tensor, indices, updates, ones);
    set_node_name(node.get_name(), scatter_nd);
    return {scatter_nd};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
