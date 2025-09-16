#include "snippets_mark_skipped.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/binary_convolution.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/mish.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/tanh.hpp"
#include "snippets/pass/tokenization.hpp"

namespace ov::intel_cpu {

SnippetsMarkSkipped::SnippetsMarkSkipped(Profile profile, bool enableBF16)
    : m_config{profile == Profile::X64 ? std::vector<ov::DiscreteTypeInfo>{ov::op::v0::Relu::get_type_info_static(),
                                                                           ov::op::v1::Add::get_type_info_static(),
                                                                           ov::op::v0::Sigmoid::get_type_info_static()}
                                       : std::vector<ov::DiscreteTypeInfo>{ov::op::v0::Relu::get_type_info_static(),
                                                                           ov::op::v0::Sigmoid::get_type_info_static(),
                                                                           ov::op::v0::Tanh::get_type_info_static()},
               [](const std::shared_ptr<const ov::Node>& n) {
                   return (std::dynamic_pointer_cast<const ov::op::v1::Convolution>(n) ||
                           std::dynamic_pointer_cast<const ov::op::v1::GroupConvolution>(n));
               },
               [](const std::shared_ptr<const ov::Node>& n) {
                   return static_cast<bool>(std::dynamic_pointer_cast<const ov::op::v1::BinaryConvolution>(n));
               },
               [](const std::shared_ptr<const ov::Node>& n) {
                   return static_cast<bool>(std::dynamic_pointer_cast<const ov::op::v0::MatMul>(n));
               }},
      m_enableBF16(enableBF16) {}

void SnippetsMarkSkipped::SetNodeFusingType(const std::shared_ptr<ov::Node>& node, NodeFusingType nodeType) {
    auto& rt = node->get_rt_info();
    rt["MayBeFusedInPlugin"] = nodeType;
}

NodeFusingType SnippetsMarkSkipped::GetNodeFusingType(const std::shared_ptr<const ov::Node>& node) {
    const auto& rt = node->get_rt_info();
    const auto it = rt.find("MayBeFusedInPlugin");
    if (it == rt.end()) {
        return NodeFusingType::NotSet;
    }
    return it->second.as<NodeFusingType>();
}

bool SnippetsMarkSkipped::run_on_model(const std::shared_ptr<ov::Model>& m) {
    auto is_single_output_single_child = [](const std::shared_ptr<const ov::Node>& n) {
        const auto outs = n->outputs();
        if (outs.size() != 1) {
            return false;
        }
        return outs[0].get_target_inputs().size() == 1;
    };

    auto mark_skipped = [](const std::shared_ptr<ov::Node>& n) {
        ov::snippets::pass::SetSnippetsNodeType(n, ov::snippets::pass::SnippetsNodeType::SkippedByPlugin);
    };

    auto is_activation = [&](const std::shared_ptr<const ov::Node>& n) -> bool {
        // A conservative set sufficient for current x64 tests; safe for ARM64 too
        return (std::dynamic_pointer_cast<const ov::op::v0::Relu>(n) ||
                std::dynamic_pointer_cast<const ov::op::v0::Abs>(n) ||
                std::dynamic_pointer_cast<const ov::op::v0::Elu>(n) ||
                std::dynamic_pointer_cast<const ov::op::v0::Sigmoid>(n) ||
                std::dynamic_pointer_cast<const ov::op::v5::HSigmoid>(n) ||
                std::dynamic_pointer_cast<const ov::op::v4::HSwish>(n) ||
                std::dynamic_pointer_cast<const ov::op::v4::Mish>(n) ||
                std::dynamic_pointer_cast<const ov::op::v5::Round>(n) ||
                std::dynamic_pointer_cast<const ov::op::v0::Gelu>(n) ||
                std::dynamic_pointer_cast<const ov::op::v7::Gelu>(n) ||
                std::dynamic_pointer_cast<const ov::op::v0::Clamp>(n) ||
                std::dynamic_pointer_cast<const ov::op::v0::Tanh>(n) ||
                std::dynamic_pointer_cast<const ov::op::v0::Sqrt>(n));
    };

    bool modified = false;
    for (const auto& node : m->get_ops()) {
        // Basic node-type tagging (kept from the initial common pass)
        if (m_config.is_convolution && m_config.is_convolution(node)) {
            SetNodeFusingType(node, NodeFusingType::FusedWithConvolution);
            modified = true;
        } else if (m_config.is_binary_convolution && m_config.is_binary_convolution(node)) {
            SetNodeFusingType(node, NodeFusingType::FusedWithBinaryConvolution);
            modified = true;
        } else if (m_config.is_matmul && m_config.is_matmul(node)) {
            SetNodeFusingType(node, NodeFusingType::FusedWithMatMul);
            modified = true;
        } else if (!m_config.fusable_op_types.empty() && std::any_of(m_config.fusable_op_types.begin(),
                                                                     m_config.fusable_op_types.end(),
                                                                     [&](const ov::DiscreteTypeInfo& ti) {
                                                                         return node->get_type_info() == ti;
                                                                     })) {
            SetNodeFusingType(node, NodeFusingType::FusedTerminator);
            modified = true;
        }

        // Skip FakeQuantize placed directly after Convolution/GroupConvolution/BinaryConvolution
        if (std::dynamic_pointer_cast<const ov::op::v0::FakeQuantize>(node)) {
            if (node->get_input_size() >= 1) {
                const auto parent = node->input_value(0).get_node_shared_ptr();
                if ((m_config.is_convolution && m_config.is_convolution(parent)) ||
                    (m_config.is_binary_convolution && m_config.is_binary_convolution(parent))) {
                    mark_skipped(std::const_pointer_cast<ov::Node>(node));
                    modified = true;
                }
            }
        }

        // Port a subset of old x64 logic: Conv -> Add -> Activation chain should be skipped by Snippets
        // Keep the logic architecture-agnostic; limited to simple single-user chains
        if (m_config.is_convolution && m_config.is_convolution(node) && is_single_output_single_child(node)) {
            // Convolution parent
            const auto child_input = *node->output(0).get_target_inputs().begin();
            auto child = child_input.get_node()->shared_from_this();
            // Sum path
            if (ov::is_type<ov::op::v1::Add>(child) && is_single_output_single_child(child)) {
                // Mark Conv as part of fusing chain (do not skip Conv itself)
                SetNodeFusingType(std::const_pointer_cast<ov::Node>(node), NodeFusingType::FusedWithConvolution);
                SetNodeFusingType(child, NodeFusingType::FusedWithConvolutionSumActivation);
                mark_skipped(child);

                // Propagate through a chain of activations (e.g., Tanh -> Sqrt)
                auto cur = child;
                while (is_single_output_single_child(cur)) {
                    const auto next_input = *cur->output(0).get_target_inputs().begin();
                    auto next = next_input.get_node()->shared_from_this();
                    // Skip over trivial data movement like Convert
                    while (is_single_output_single_child(next) &&
                           std::dynamic_pointer_cast<const ov::op::v0::Convert>(next)) {
                        const auto after_convert_input = *next->output(0).get_target_inputs().begin();
                        next = after_convert_input.get_node()->shared_from_this();
                    }
                    if (!is_activation(next)) {
                        break;
                    }
                    // Mark activation as part of the fusion chain and skipped
                    SetNodeFusingType(next, NodeFusingType::FusedWithConvolution);
                    mark_skipped(next);
                    cur = next;
                    modified = true;
                }
                modified = true;
            }
        }
    }

    return modified;
}

}  // namespace ov::intel_cpu
