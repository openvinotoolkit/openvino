// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_topology.hpp"

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <unordered_set>

#include "moe_transformation_utils.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::npuw::moe {
namespace {

// Preserve the original limit: at most seven score views between Transpose
// and Multiply. TopK index conversions are checked separately without a limit.
constexpr size_t max_score_views = 7;

std::optional<std::vector<int64_t>> integers(const ov::Output<ov::Node>& value) {
    const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(value.get_node_shared_ptr());
    if (!constant || !constant->get_element_type().is_integral_number())
        return std::nullopt;
    return constant->cast_vector<int64_t>();
}

bool is_axis(const ov::Output<ov::Node>& value, int64_t axis, int64_t rank) {
    const auto data = integers(value);
    return data && data->size() == 1 && (data->front() == axis || data->front() == axis - rank);
}

bool is_zero(const ov::Output<ov::Node>& value) {
    const auto node = value.get_node_shared_ptr();
    if (auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node)) {
        const auto data = constant->cast_vector<float>();
        return !data.empty() && std::all_of(data.begin(), data.end(), [](float x) {
            return x == 0.0f;
        });
    }
    if (ov::is_type<ov::op::v0::Convert>(node) || ov::is_type<ov::op::v3::Broadcast>(node) ||
        ov::is_type<ov::op::v1::Broadcast>(node))
        return is_zero(node->input_value(0));
    return false;
}

std::shared_ptr<ov::Node> only_consumer(const ov::Output<ov::Node>& value) {
    const auto users = value.get_target_inputs();
    return users.size() == 1 ? users.begin()->get_node()->shared_from_this() : nullptr;
}

bool has_expert_axis(const ov::PartialShape& shape, size_t experts) {
    return shape.rank().is_static() && shape.rank().get_length() > 0 && shape[0].is_static() &&
           shape[0].get_length() == static_cast<int64_t>(experts);
}

bool has_token_layout(const ov::PartialShape& shape, const BatchedMoE& moe) {
    if (!has_expert_axis(shape, moe.num_experts))
        return false;
    const auto tokens = moe.indices.get_partial_shape()[0];
    if (shape.rank() == ov::Rank(3))
        return shape[1].compatible(tokens);
    if (shape.rank() == ov::Rank(4))
        return (shape[1] == ov::Dimension(1) && shape[2].compatible(tokens)) ||
               (shape[2] == ov::Dimension(1) && shape[1].compatible(tokens));
    return false;
}

bool has_supported_broadcast(const std::shared_ptr<ov::Node>& node) {
    const auto arithmetic = ov::as_type_ptr<ov::op::util::BinaryElementwiseArithmetic>(node);
    if (!arithmetic)
        return true;
    // Selection below assumes right-aligned NumPy axes or equal-rank operands.
    // PDPD can put a lower-rank scale on the expert axis; leaving it ungathered
    // silently changes expert order even when K == E and shapes still validate.
    const auto mode = arithmetic->get_autob().m_type;
    return mode == ov::op::AutoBroadcastType::NUMPY || mode == ov::op::AutoBroadcastType::NONE;
}

// Gathering must commute with the dequantization/view chain. In particular,
// never treat a feature dimension that happens to equal E as an expert axis.
bool can_gather_weight(const ov::Output<ov::Node>& value, size_t experts) {
    const auto node = value.get_node_shared_ptr();
    const auto shape = value.get_partial_shape();
    if (!shape.is_static() || !has_expert_axis(shape, experts) || !has_supported_broadcast(node))
        return false;
    if (ov::is_type<ov::op::v0::Constant>(node))
        return true;
    if (ov::is_type<ov::op::v0::Convert>(node))
        return can_gather_weight(node->input_value(0), experts);
    if (ov::is_type<ov::op::v1::Reshape>(node))
        return integers(node->input_value(1)).has_value() && can_gather_weight(node->input_value(0), experts);
    if (ov::is_type<ov::op::v1::Multiply>(node) || ov::is_type<ov::op::v1::Subtract>(node) ||
        ov::is_type<ov::op::v1::Add>(node)) {
        for (const auto& input : node->input_values()) {
            if (!moe_utils::is_constant_derived(input.get_node_shared_ptr()))
                return false;
            if (input.get_partial_shape().rank() == shape.rank() &&
                has_expert_axis(input.get_partial_shape(), experts) && !can_gather_weight(input, experts))
                return false;
        }
        return true;
    }
    return false;
}

// Walk only the expert arm, stopping at Tile and constant leaves. Routing and
// the rest of the model are outside this validation boundary.
bool collect_experts(BatchedMoE& moe) {
    std::unordered_set<std::shared_ptr<ov::Node>> seen;
    size_t matmuls = 0;
    std::function<bool(const ov::Output<ov::Node>&)> visit = [&](const ov::Output<ov::Node>& value) {
        const auto node = value.get_node_shared_ptr();
        if (!seen.insert(node).second)
            return true;
        if (ov::is_type<ov::op::v0::Constant>(node))
            return true;
        if (!has_supported_broadcast(node))
            return false;
        if (moe_utils::is_constant_derived(node)) {
            for (const auto& input : node->input_values()) {
                if (!visit(input))
                    return false;
            }
            moe.expert_nodes.push_back(node);
            return true;
        }
        if (auto tile = ov::as_type_ptr<ov::op::v0::Tile>(node)) {
            const auto repeats = integers(tile->input_value(1));
            const auto input_shape = tile->input_value(0).get_partial_shape();
            if (!repeats || *repeats != std::vector<int64_t>{static_cast<int64_t>(moe.num_experts), 1} ||
                input_shape.rank() != ov::Rank(2) || !input_shape[0].compatible(moe.indices.get_partial_shape()[0]) ||
                (moe.tile && moe.tile != tile))
                return false;
            moe.tile = tile;
            moe.expert_nodes.push_back(node);
            return true;
        }
        const auto shape = value.get_partial_shape();
        if (!has_token_layout(shape, moe))
            return false;
        if (auto matmul = ov::as_type_ptr<ov::op::v0::MatMul>(node)) {
            if (matmul->get_transpose_a() || !can_gather_weight(matmul->input_value(1), moe.num_experts))
                return false;
            ++matmuls;
        } else if (ov::is_type<ov::op::v1::Reshape>(node)) {
            // Never merge token and feature axes: host prefill dispatches tokens
            // independently. Only the initial Tile separates the expert axis;
            // later activation views may insert/move singleton dimensions.
            const auto input_shape = node->input_value(0).get_partial_shape();
            if (!input_shape.rank().is_static() || input_shape.rank().get_length() < 2 ||
                !input_shape[input_shape.rank().get_length() - 1].compatible(shape[shape.rank().get_length() - 1]) ||
                (!ov::is_type<ov::op::v0::Tile>(node->input_value(0).get_node()) &&
                 !has_token_layout(input_shape, moe)))
                return false;
        } else if (ov::is_type<ov::op::v8::Slice>(node)) {
            if (node->get_input_size() != 5 ||
                !is_axis(node->input_value(4), shape.rank().get_length() - 1, shape.rank().get_length()))
                return false;
        } else if (ov::is_type<ov::op::v1::Split>(node) || ov::is_type<ov::op::v1::VariadicSplit>(node)) {
            if (!is_axis(node->input_value(1), shape.rank().get_length() - 1, shape.rank().get_length()))
                return false;
        } else if (!ov::is_type<ov::op::v0::Convert>(node) && !ov::is_type<ov::op::v4::Swish>(node) &&
                   !ov::is_type<ov::op::util::UnaryElementwiseArithmetic>(node) &&
                   !ov::is_type<ov::op::util::BinaryElementwiseArithmetic>(node)) {
            return false;
        }
        // Shape inputs are metadata, not part of the expert dataflow.
        const size_t count = ov::is_type<ov::op::v1::Reshape>(node) || ov::is_type<ov::op::v8::Slice>(node) ||
                                     ov::is_type<ov::op::v1::Split>(node) ||
                                     ov::is_type<ov::op::v1::VariadicSplit>(node)
                                 ? 1
                                 : node->get_input_size();
        for (size_t i = 0; i < count; ++i) {
            if (!visit(node->input_value(i)))
                return false;
        }
        moe.expert_nodes.push_back(node);
        return true;
    };
    if (!visit(moe.expert_output) || !moe.tile || matmuls < 2)
        return false;
    // Host partitioning must not steal an intermediate used by another branch.
    for (const auto& node : moe.expert_nodes) {
        for (const auto& output : node->outputs()) {
            for (const auto& input : output.get_target_inputs()) {
                if (!seen.count(input.get_node()->shared_from_this()) && input.get_node() != moe.weighted_output.get())
                    return false;
            }
        }
    }
    return true;
}

}  // namespace

BatchedMoEPattern::BatchedMoEPattern() {
    namespace opp = ov::pass::pattern;

    auto k = opp::any_input([](const ov::Output<ov::Node>& value) {
        const auto data = integers(value);
        return data && data->size() == 1 && data->front() > 0;
    });
    m_topk = opp::wrap_type<ov::op::v11::TopK>(
        {opp::any_input(), k},
        opp::output_index_matches(1) && [](const ov::Output<ov::Node>& value) {
            const auto topk = ov::as_type_ptr<ov::op::v11::TopK>(value.get_node_shared_ptr());
            return topk->get_mode() == ov::op::v11::TopK::Mode::MAX &&
                   (topk->get_axis() == 1 || topk->get_axis() == -1);
        });

    // Indices may have arbitrary i32/i64 Convert chains. Match their unwrapped
    // source against m_topk in extract(); the score expression stays independent.
    auto zero = opp::any_input(is_zero);
    auto expert_axis = opp::any_input([](const ov::Output<ov::Node>& value) {
        return is_axis(value, 1, 2);
    });
    m_scatter = opp::wrap_type<ov::op::v3::ScatterElementsUpdate, ov::op::v12::ScatterElementsUpdate>(
        {zero, opp::any_input(), opp::any_input(), expert_axis},
        opp::consumers_count(1) && [](const ov::Output<ov::Node>& value) {
            const auto scatter = ov::as_type_ptr<ov::op::v12::ScatterElementsUpdate>(value.get_node_shared_ptr());
            return !scatter || scatter->get_reduction() == ov::op::v12::ScatterElementsUpdate::Reduction::NONE;
        });
    auto permutation = opp::any_input([](const ov::Output<ov::Node>& value) {
        return integers(value) == std::optional<std::vector<int64_t>>{{1, 0}};
    });
    m_score_transpose =
        opp::wrap_type<ov::op::v1::Transpose>({m_scatter, permutation}, opp::consumers_count(1));
    auto scores = m_score_transpose;
    for (size_t i = 0; i < max_score_views; ++i) {
        scores = opp::optional<ov::op::v1::Reshape, ov::op::v0::Unsqueeze>(
            {scores, opp::any_input()}, opp::consumers_count(1));
    }
    m_expert_output = opp::any_input();
    m_weighted_output = opp::wrap_type<ov::op::v1::Multiply>(
        {m_expert_output, scores}, opp::consumers_count(1) && has_supported_broadcast);
    m_reduction = opp::wrap_type<ov::op::v1::ReduceSum>({m_weighted_output, opp::any_input()});
}

std::optional<BatchedMoE> BatchedMoEPattern::extract(ov::pass::pattern::Matcher& matcher) const {
    const auto& matched = matcher.get_pattern_value_map();
    const auto scatter = matched.at(m_scatter).get_node_shared_ptr();
    auto indices = scatter->input_value(1);
    while (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(indices.get_node_shared_ptr())) {
        if (convert->get_destination_type() != ov::element::i32 && convert->get_destination_type() != ov::element::i64)
            return std::nullopt;
        indices = convert->input_value(0);
    }
    ov::pass::pattern::Matcher selection(m_topk, "BatchedMoESelection");
    if (!selection.match(indices))
        return std::nullopt;
    const auto topk = ov::as_type_ptr<ov::op::v11::TopK>(indices.get_node_shared_ptr());
    const auto data_shape = scatter->input_value(0).get_partial_shape();
    const auto indices_shape = indices.get_partial_shape();
    const auto selection_shape = topk->input_value(0).get_partial_shape();
    const auto k = integers(topk->input_value(1));
    if (data_shape.rank() != ov::Rank(2) || indices_shape.rank() != ov::Rank(2) ||
        selection_shape.rank() != ov::Rank(2) || !selection_shape.compatible(data_shape) ||
        !selection_shape[1].is_static() || selection_shape[1] != data_shape[1] || !data_shape[1].is_static() || !k ||
        k->front() > data_shape[1].get_length() ||
        !scatter->input_value(2).get_partial_shape().compatible(indices_shape))
        return std::nullopt;
    BatchedMoE moe;
    moe.topk = topk;
    moe.indices = indices;
    moe.scores = scatter->input_value(2);
    moe.num_experts = static_cast<size_t>(data_shape[1].get_length());
    moe.num_selected = static_cast<size_t>(k->front());
    moe.score_transpose = ov::as_type_ptr<ov::op::v1::Transpose>(matched.at(m_score_transpose).get_node_shared_ptr());
    moe.expert_output = matched.at(m_expert_output);
    moe.weighted_output = ov::as_type_ptr<ov::op::v1::Multiply>(matched.at(m_weighted_output).get_node_shared_ptr());
    moe.broadcast_scores =
        moe.weighted_output->input_value(moe.weighted_output->input_value(0) == moe.expert_output ? 1 : 0);
    moe.reduction = ov::as_type_ptr<ov::op::v1::ReduceSum>(matched.at(m_reduction).get_node_shared_ptr());
    const auto expert_shape = moe.expert_output.get_partial_shape();
    const auto score_shape = moe.broadcast_scores.get_partial_shape();
    if (!moe.reduction || !has_expert_axis(expert_shape, moe.num_experts) ||
        !has_expert_axis(score_shape, moe.num_experts) || expert_shape.rank() != score_shape.rank() ||
        score_shape.rank().get_length() < 3 || score_shape.rank().get_length() > 4 ||
        score_shape[score_shape.rank().get_length() - 1] != ov::Dimension(1) ||
        !is_axis(moe.reduction->input_value(1), 0, expert_shape.rank().get_length()))
        return std::nullopt;
    const auto rank = score_shape.rank().get_length();
    if (rank == 4 && score_shape[1] != ov::Dimension(1) && score_shape[2] != ov::Dimension(1))
        return std::nullopt;
    const size_t token_axis = rank == 4 && score_shape[1] == ov::Dimension(1) ? 2 : 1;
    if (!score_shape[token_axis].compatible(indices_shape[0]))
        return std::nullopt;
    for (int64_t axis = 1; axis < rank - 1; ++axis) {
        if (!expert_shape[axis].compatible(score_shape[axis]))
            return std::nullopt;
    }
    if (!collect_experts(moe))
        return std::nullopt;
    return moe;
}

std::optional<BatchedMoE> BatchedMoEPattern::match(const std::shared_ptr<ov::Node>& reduction) const {
    if (!ov::is_type<ov::op::v1::ReduceSum>(reduction))
        return std::nullopt;
    ov::pass::pattern::Matcher matcher(m_reduction, "BatchedMoEBoundary");
    return matcher.match(reduction) ? extract(matcher) : std::nullopt;
}

std::optional<BatchedMoE> match_batched_moe(const std::shared_ptr<ov::Node>& scatter) {
    if (!ov::is_type<ov::op::v3::ScatterElementsUpdate>(scatter) &&
        !ov::is_type<ov::op::v12::ScatterElementsUpdate>(scatter))
        return std::nullopt;
    // Legacy router callbacks are anchored upstream of the shared pattern.
    // Locate a candidate only; all topology checks belong to BatchedMoEPattern.
    auto node = scatter;
    for (size_t hop = 0; hop < max_score_views + 3; ++hop) {
        node = only_consumer(node->output(0));
        if (!node)
            return std::nullopt;
        if (ov::is_type<ov::op::v1::ReduceSum>(node)) {
            const auto moe = BatchedMoEPattern().match(node);
            return moe && moe->score_transpose->input_value(0).get_node_shared_ptr() == scatter ? moe : std::nullopt;
        }
    }
    return std::nullopt;
}

bool can_device_route(const BatchedMoE& moe) {
    const auto index_shape = moe.indices.get_partial_shape();
    if (!index_shape.is_static() || index_shape[0] != ov::Dimension(1) ||
        !moe.broadcast_scores.get_partial_shape().is_static())
        return false;
    for (const auto& node : moe.expert_nodes) {
        for (const auto& output : node->outputs()) {
            if (!output.get_partial_shape().is_static())
                return false;
        }
    }
    return true;
}

std::shared_ptr<ov::Node> build_device_routed_moe(const BatchedMoE& moe) {
    if (!can_device_route(moe))
        return nullptr;
    auto indices = std::make_shared<ov::op::v1::Reshape>(
        moe.indices,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {moe.num_selected}),
        false);
    indices->set_friendly_name(moe.topk->get_friendly_name() + "/indices_reshaped");
    const auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    std::unordered_map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> gathered;
    std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&)> select_weight =
        [&](const ov::Output<ov::Node>& value) -> ov::Output<ov::Node> {
        const auto node = value.get_node_shared_ptr();
        if (const auto it = gathered.find(node); it != gathered.end())
            return it->second->output(value.get_index());
        std::shared_ptr<ov::Node> selected;
        if (ov::is_type<ov::op::v0::Constant>(node)) {
            selected = std::make_shared<ov::op::v8::Gather>(value, indices, axis);
        } else if (ov::is_type<ov::op::v1::Reshape>(node)) {
            auto shape = value.get_shape();
            shape[0] = moe.num_selected;
            selected = std::make_shared<ov::op::v1::Reshape>(
                select_weight(node->input_value(0)),
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{shape.size()}, shape),
                false);
        } else {
            auto inputs = node->input_values();
            for (auto& input : inputs) {
                if (input.get_partial_shape().rank() == value.get_partial_shape().rank() &&
                    has_expert_axis(input.get_partial_shape(), moe.num_experts))
                    input = select_weight(input);
            }
            selected = node->clone_with_new_inputs(inputs);
        }
        selected->set_friendly_name(node->get_friendly_name() + "/gathered");
        ov::copy_runtime_info(node, selected);
        gathered.emplace(node, selected);
        return selected->output(value.get_index());
    };
    std::unordered_map<std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>> clones;
    for (const auto& node : moe.expert_nodes) {
        if (moe_utils::is_constant_derived(node))
            continue;
        auto inputs = node->input_values();
        if (node == moe.tile) {
            inputs[1] =
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<size_t>{moe.num_selected, 1});
        } else {
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto input = inputs[i];
                if (const auto it = clones.find(input.get_node_shared_ptr()); it != clones.end()) {
                    inputs[i] = it->second->output(input.get_index());
                } else if ((ov::is_type<ov::op::v0::MatMul>(node) && i == 1) ||
                           (input.get_partial_shape().rank() == node->get_output_partial_shape(0).rank() &&
                            has_expert_axis(input.get_partial_shape(), moe.num_experts))) {
                    if (!can_gather_weight(input, moe.num_experts))
                        return nullptr;
                    inputs[i] = select_weight(input);
                }
            }
            if (ov::is_type<ov::op::v1::Reshape>(node)) {
                auto shape = node->get_output_shape(0);
                shape[0] = moe.num_selected;
                inputs[1] = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{shape.size()}, shape);
            }
        }
        auto clone = node->clone_with_new_inputs(inputs);
        clone->set_friendly_name(node->get_friendly_name());
        ov::copy_runtime_info(node, clone);
        clones.emplace(node, clone);
    }
    auto scores = std::make_shared<ov::op::v1::Transpose>(moe.scores, moe.score_transpose->input_value(1));
    auto score_shape = moe.broadcast_scores.get_shape();
    score_shape[0] = moe.num_selected;
    auto broadcast_scores = std::make_shared<ov::op::v1::Reshape>(
        scores,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{score_shape.size()}, score_shape),
        false);
    auto expert_output = clones.at(moe.expert_output.get_node_shared_ptr())->output(moe.expert_output.get_index());
    auto weighted =
        std::make_shared<ov::op::v1::Multiply>(expert_output, broadcast_scores, moe.weighted_output->get_autob());
    auto result = moe.reduction->clone_with_new_inputs({weighted, moe.reduction->input_value(1)});
    result->set_friendly_name(moe.reduction->get_friendly_name());
    ov::copy_runtime_info(moe.reduction, result);
    return result;
}

}  // namespace ov::npuw::moe