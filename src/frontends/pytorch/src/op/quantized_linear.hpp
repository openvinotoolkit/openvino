#pragma once

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_quantized_linear_dynamic(const NodeContext& context);

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov 