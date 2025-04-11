#pragma once

#include <openvino/frontend/onnx/extension/op.hpp>

namespace ov {
namespace frontend {
namespace onnx {
namespace op {

class DeformConv : public ov::frontend::onnx::Op {
public:
    DeformConv(const std::string& name, int version);
    
    void operator()(const std::vector<ov::frontend::onnx::Tensor>& inputs,
                    const std::vector<ov::frontend::onnx::Attribute>& attributes,
                    std::vector<ov::frontend::onnx::Tensor>& outputs) const override;

private:
    int m_version;  // For storing version (19 or 22) to handle datatype differences
};

}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
