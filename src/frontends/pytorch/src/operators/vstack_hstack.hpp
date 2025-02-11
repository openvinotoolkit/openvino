#pragma once

#include "openvino/frontend/pytorch/operator_converter.hpp"
#include "openvino/frontend/pytorch/visibility.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

// Conversion pattern for aten::vstack operator
class VStackConverter : public OperatorConverter {
public:
    PYTORCH_API VStackConverter();
    PYTORCH_API void convert(const torch::jit::Node* node,
                           std::shared_ptr<ov::Model>& model,
                           const std::vector<ov::Output<ov::Node>>& inputs) override;
};

// Conversion pattern for aten::hstack operator
class HStackConverter : public OperatorConverter {
public:
    PYTORCH_API HStackConverter();
    PYTORCH_API void convert(const torch::jit::Node* node,
                           std::shared_ptr<ov::Model>& model,
                           const std::vector<ov::Output<ov::Node>>& inputs) override;
};

} // namespace op
} // namespace pytorch
} // namespace frontend
} // namespace ov 