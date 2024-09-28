// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cassert>
#include <climits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/opsets/opset1.hpp"
#include "snippets/itt.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/pass/fq_decomposition.hpp"
#include "snippets/pass/fuse_transpose_brgemm.hpp"
#include "snippets/pass/tokenization.hpp"
#include "snippets/pass/transpose_decomposition.hpp"
#include "snippets/remarks.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/utils/tokenization_utils.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace snippets {
namespace pass {


namespace {
auto is_supported_op(const std::shared_ptr<const Node> &n) -> bool {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::is_supported_op")
    auto is_supported_matmul = [](const std::shared_ptr<const Node>& n) -> bool {
        const auto& matmul = ov::as_type_ptr<const opset1::MatMul>(n);
        const auto& out_rank = n->get_output_partial_shape(0).rank();
        if (!matmul || out_rank.is_dynamic() || out_rank.get_length() != 4)
            return false;
        const auto intype_0 = matmul->get_input_element_type(0);
        const auto intype_1 = matmul->get_input_element_type(1);
        const bool is_f32 = intype_0 == element::f32 && intype_1 == element::f32;
        const bool is_int8 = (intype_0 == element::i8 || intype_0 == element::u8) && (intype_1 == element::i8);
        const bool is_bf16 = intype_0 == element::bf16 && intype_1 == element::bf16;
        return is_f32 || is_bf16 || is_int8;
    };
    auto is_supported_transpose = [](const std::shared_ptr<const Node>& n) -> bool {
        const auto& transpose = as_type_ptr<const opset1::Transpose>(n);
        if (transpose) {
            const auto parent = transpose->get_input_node_shared_ptr(0);
            const auto child = transpose->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
            auto is_brgemm_case = ov::is_type<opset1::MatMul>(parent) || ov::is_type<opset1::MatMul>(child);
            // Check for Transpose parent is MatMul inside Subgraph
            if (const auto subgraph = ov::as_type_ptr<const op::Subgraph>(parent)) {
                if (GetSnippetsSubgraphType(subgraph) != SnippetsSubgraphType::Completed) {
                    const auto body = subgraph->body_ptr();
                    const auto subgraph_output = body->get_results()[transpose->input_value(0).get_index()]->get_input_node_shared_ptr(0);
                    is_brgemm_case = is_brgemm_case || ov::is_type<opset1::MatMul>(subgraph_output);
                }
            }

            const auto& order = as_type_ptr<const opset1::Constant>(n->get_input_node_shared_ptr(1));
            if (order) {
                const auto order_value = order->cast_vector<int>();
                return (TransposeDecomposition::is_supported_transpose_order(order_value)) ||
                       (is_brgemm_case && FuseTransposeBrgemm::is_supported_transpose_order(order_value));
            }
        }
        return false;
    };

    auto is_supported_fq_op = [](const std::shared_ptr<const Node>& n) -> bool {
        return CommonFakeQuantizeDecomposition::is_supported_fq(ov::as_type_ptr<const opset1::FakeQuantize>(n));
    };

    auto is_supported_ternary_eltwise_op = [](const std::shared_ptr<const Node> &n) -> bool {
        return ov::is_type<ov::op::v1::Select>(n);
    };

    auto is_supported_binary_eltwise_op = [](const std::shared_ptr<const Node> &n) -> bool {
        return ov::is_type<ov::op::v1::Add>(n)
            || ov::is_type<ov::op::v1::Divide>(n)
            || ov::is_type<ov::op::v1::Equal>(n)
            || ov::is_type<ov::op::v1::FloorMod>(n)
            || ov::is_type<ov::op::v1::Greater>(n)
            || ov::is_type<ov::op::v1::GreaterEqual>(n)
            || ov::is_type<ov::op::v1::Less>(n)
            || ov::is_type<ov::op::v1::LessEqual>(n)
            || ov::is_type<ov::op::v1::LogicalAnd>(n)
            || ov::is_type<ov::op::v1::LogicalOr>(n)
            || ov::is_type<ov::op::v1::LogicalXor>(n)
            || ov::is_type<ov::op::v1::Maximum>(n)
            || ov::is_type<ov::op::v1::Minimum>(n)
            || ov::is_type<ov::op::v1::Mod>(n)
            || ov::is_type<ov::op::v1::Multiply>(n)
            || ov::is_type<ov::op::v1::NotEqual>(n)
            || ov::is_type<ov::op::v0::PRelu>(n)
            || ov::is_type<ov::op::v1::Power>(n)
            || ov::is_type<ov::op::v0::SquaredDifference>(n)
            || ov::is_type<ov::op::v1::Subtract>(n)
            || ov::is_type<ov::op::v0::Xor>(n)
            || ov::is_type<ov::op::v0::Convert>(n);
    };

    auto is_supported_unary_eltwise_op = [](const std::shared_ptr<const Node> &n) -> bool {
        return ov::is_type<ov::op::v0::Abs>(n)
            || ov::is_type<ov::op::v0::Clamp>(n)
            || ov::is_type<ov::op::v0::Floor>(n)
            || ov::is_type<ov::op::v0::Ceiling>(n)
            || ov::is_type<ov::op::v0::Elu>(n)
            || ov::is_type<ov::op::v0::Erf>(n)
            || ov::is_type<ov::op::v0::Exp>(n)
            || ov::is_type<ov::op::v1::LogicalNot>(n)
            || ov::is_type<ov::op::v4::Mish>(n)
            || ov::is_type<ov::op::v0::Negative>(n)
            || ov::is_type<ov::op::v0::Relu>(n)
            || ov::is_type<ov::op::v5::Round>(n)
            || ov::is_type<ov::op::v0::Sigmoid>(n)
            || ov::is_type<ov::op::v0::Sqrt>(n)
            || ov::is_type<ov::op::v0::Tanh>(n)
            || ov::is_type<ov::op::v0::Gelu>(n)
            || ov::is_type<ov::op::v7::Gelu>(n)
            || ov::is_type<ov::op::v4::Swish>(n)
            || ov::is_type<ov::op::v4::HSwish>(n);
    };

    auto is_supported_softmax = [](const std::shared_ptr<const Node> &n) -> bool {
        if (n->get_input_size() != 1 || n->get_input_partial_shape(0).rank().is_dynamic())
            return false;
        int64_t axis = -1;
        const auto rank = n->get_input_partial_shape(0).rank();
        if (const auto softmax_v8 = ov::as_type_ptr<const ov::op::v8::Softmax>(n)) {
            if (rank.is_static()) {
                axis = ov::util::try_normalize_axis(softmax_v8->get_axis(), rank, *n);
            }
        } else if (const auto softmax_v1 = ov::as_type_ptr<const ov::op::v1::Softmax>(n)) {
            axis = softmax_v1->get_axis();
        } else {
            return false;
        }
        return axis >= 0 && axis == (rank.get_length() - 1);
    };

    auto is_supported_broadcast_op = [](const std::shared_ptr<const Node> &n) -> bool {
        // Broadcast is supported only for MHA tokenization where there are needed and special checks
        if (auto broadcast_v1 = ov::as_type_ptr<const ov::op::v1::Broadcast>(n)) {
            return broadcast_v1->get_broadcast_spec().m_type == ov::op::AutoBroadcastType::NUMPY;
        } else if (auto broadcast_v3 = ov::as_type_ptr<const ov::op::v3::Broadcast>(n)) {
            return broadcast_v3->get_broadcast_spec().m_type == ov::op::BroadcastType::NUMPY;
        }
        return false;
    };

    auto is_supported_reduce_op = [](const std::shared_ptr<const Node> &n) -> bool {
        if (ov::is_type<const ov::op::v1::ReduceMax>(n) || ov::is_type<const ov::op::v1::ReduceSum>(n)) {
            const auto& reduce_base = ov::as_type_ptr<const ov::op::util::ArithmeticReductionKeepDims>(n);
            const auto& axis_constant = ov::as_type_ptr<const ov::op::v0::Constant>(n->get_input_node_shared_ptr(1));
            const auto rank = n->get_input_partial_shape(0).rank();
            if (rank.is_dynamic() || !reduce_base->get_keep_dims() || !axis_constant || shape_size(axis_constant->get_shape()) != 1)
                return false;

            const auto axis_value = axis_constant->cast_vector<int32_t>(1)[0];
            const auto normalized_axis = util::normalize(axis_value, rank.get_length());
            // Note: Reduction only over the last dimension is currently supported
            return normalized_axis == rank.get_length() - 1;
        }
        return false;
    };

    return is_supported_fq_op(n) ||
           is_supported_unary_eltwise_op(n) ||
           is_supported_binary_eltwise_op(n) ||
           is_supported_ternary_eltwise_op(n) ||
           is_supported_transpose(n) ||
           is_supported_softmax(n) ||
           is_supported_matmul(n) ||
           is_supported_broadcast_op(n) ||
           is_supported_reduce_op(n);
}

auto has_supported_in_out(const std::shared_ptr<const Node> &n) -> bool {
    auto supported = [](descriptor::Tensor& t) -> bool {
        // TODO [122585] Need to add dynamic rank support
        return t.get_partial_shape().rank().is_static();
    };
    const auto&  inputs = n->inputs();
    const auto&  outputs = n->outputs();
    // todo: Is this check necessary? Remove if not
    for (const auto& out : outputs) {
        for (const auto& in_out : out.get_target_inputs()) {
            if (ov::is_type<ov::op::v5::Loop>(in_out.get_node()->shared_from_this())) {
                return false;
            }
        }
    }
    return std::all_of(inputs.begin(), inputs.end(), [&](const Input<const Node>& in) {return  supported(in.get_tensor());}) &&
           std::all_of(outputs.begin(), outputs.end(), [&](const Output<const Node>& out) {return  supported(out.get_tensor());});
}
} // namespace

const std::set<ov::element::Type>& ov::snippets::pass::TokenizeSnippets::get_supported_element_types() {
    static const std::set<ov::element::Type> supported_element_types = {ov::element::f32,
                                                                        ov::element::bf16,
                                                                        ov::element::f16,
                                                                        ov::element::i8,
                                                                        ov::element::u8};
    return supported_element_types;
}

bool TokenizeSnippets::AppropriateForSubgraph(const std::shared_ptr<const Node> &node) {
    return
        is_supported_op(node) &&
        has_supported_in_out(node) &&
        node->get_control_dependencies().empty() &&
        snippets::op::Subgraph::check_broadcast(node);
}

TokenizeSnippets::TokenizeSnippets(const SnippetsTokenization::Config& config) {
    MATCHER_SCOPE(TokenizeSnippets);

    auto label = ov::pass::pattern::any_input(
        [](ov::Output<ov::Node> out) {
            const auto n = out.get_node_shared_ptr();
            // todo: MatMul and Transpose ops are always skipped by the SnippetsMarkSkipped pass.
            //  This is a temporary solution. Either modify SnippetsMarkSkipped
            //  or align this with the custom MHA tokenization pass.
            return (GetSnippetsNodeType(n) != SnippetsNodeType::SkippedByPlugin ||
                    ov::is_type<ov::op::v0::MatMul>(n) || ov::is_type<ov::op::v1::Transpose>(n))
                    && AppropriateForSubgraph(n);
        });
    ov::graph_rewrite_callback callback = [=](ov::pass::pattern::Matcher &m) -> bool {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::CreateSubgraph_callback")
        auto node = m.get_match_root();
        if (transformation_callback(node)) {
            return false;
        }
        remark(1) << "Match root: " << node->get_friendly_name() << " " << node << std::endl;
        return ov::snippets::utils::tokenize_node(node, config);
    };
    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(label, matcher_name);
    register_matcher(matcher, callback);
}
} // namespace pass
} // namespace snippets
} // namespace ov
