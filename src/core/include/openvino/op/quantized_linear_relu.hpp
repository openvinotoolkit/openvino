#pragma once

#include "openvino/op/op.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

// Function declaration for translating quantized::linear_relu
ov::OutputVector translate_quantized_relu(const NodeContext& context);

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
