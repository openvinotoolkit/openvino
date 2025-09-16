#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::intel_cpu {

// See comments in the original base for semantics of values.
enum class NodeFusingType : int8_t {
    NotSet,
    FusedTerminator,
    FusedWithConvolution,
    FusedWithBinaryConvolution,
    FusedWithConvolutionSumActivation,
    FusedWithMatMul,
    FusedWithFC,
    FusedWithReduce,
    FusedWithGather,
    FusedWithMisc
};

struct MarkSkippedConfig {
    std::vector<ov::DiscreteTypeInfo> fusable_op_types;
    std::function<bool(const std::shared_ptr<const ov::Node>&)> is_convolution;
    std::function<bool(const std::shared_ptr<const ov::Node>&)> is_binary_convolution;
    std::function<bool(const std::shared_ptr<const ov::Node>&)> is_matmul;
};

class SnippetsMarkSkipped : public ov::pass::ModelPass {
public:
    enum class Profile : uint8_t { X64, ARM64 };

    OPENVINO_MODEL_PASS_RTTI("SnippetsMarkSkipped");
    explicit SnippetsMarkSkipped(Profile profile, bool enableBF16 = false);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;

    static void SetNodeFusingType(const std::shared_ptr<ov::Node>& node, NodeFusingType nodeType);
    static NodeFusingType GetNodeFusingType(const std::shared_ptr<const ov::Node>& node);

private:
    MarkSkippedConfig m_config;
    bool m_enableBF16 = false;
};

}  // namespace ov::intel_cpu
