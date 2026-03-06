#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace frontend {
namespace pytorch {

OutputVector translate_isinstance(const NodeContext& node) {
    // prim::isinstance is a TorchScript type-check op.
    // It does not require runtime computation in OpenVINO.
    return {ov::op::v0::Constant::create(ov::element::boolean, {}, {false})};
}

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
