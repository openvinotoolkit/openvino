#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/core/type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

ov::OutputVector translate_get(const ov::frontend::pytorch::NodeContext& context) {
    num_inputs_check(context, 2, 2);

    auto data = context.get_input(0);
    auto index_node = context.get_input(1).get_node_shared_ptr();

    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(index_node);
    OPENVINO_ASSERT(constant, "aten::get expects the second input to be a constant index.");

    auto index_vec = constant->cast_vector<int64_t>();  // OR: get_vector<int64_t>()
    OPENVINO_ASSERT(index_vec.size() == 1, "aten::get expects a single integer index.");
    int64_t index = index_vec[0];

    auto source_node = data.get_node_shared_ptr();
    OPENVINO_ASSERT(index >= 0 && index < static_cast<int64_t>(source_node->get_output_size()),
                    "Index for aten::get is out of bounds.");

    return {source_node->output(index)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
