// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_utils.hpp"

#include "itt.hpp"
#include "openvino/op/batch_to_space.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/elu.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/is_finite.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/reverse_sequence.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/space_to_batch.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace pass {
namespace transpose_sinking {
namespace utils {

using namespace ov;

using NodePtr = std::shared_ptr<Node>;
using InsertBroadcastUnsqueezeT = std::function<NodePtr(const Output<Node>& node, size_t n_dims)>;

Output<Node> ChangeValuesOrder(const Output<Node>& input,
                               const AxisVector& transpose_axis_order,
                               const std::shared_ptr<ov::op::v0::Constant>& axis) {
    auto indices =
        std::make_shared<ov::op::v0::Constant>(element::i32, Shape{transpose_axis_order.size()}, transpose_axis_order);
    auto gather = std::make_shared<ov::op::v8::Gather>(input, indices, axis);
    copy_runtime_info(input.get_node_shared_ptr(), gather);
    return gather;
}

Output<Node> ChangeAxes(const Output<Node>& indices,
                        const std::shared_ptr<ov::op::v0::Constant>& data,
                        const std::shared_ptr<ov::op::v0::Constant>& axis) {
    auto gather = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
    copy_runtime_info(indices.get_node_shared_ptr(), gather);
    return gather;
}

Output<Node> ChangeAxes(const Output<Node>& indices,
                        const AxisVector& transpose_axis_order,
                        const std::shared_ptr<ov::op::v0::Constant>& axis) {
    auto data =
        std::make_shared<ov::op::v0::Constant>(element::i32, Shape{transpose_axis_order.size()}, transpose_axis_order);
    return ChangeAxes(indices, data, axis);
}

bool if_transpose_sinkable_default(const std::shared_ptr<ov::op::v1::Transpose>& transpose,
                                   const std::shared_ptr<ov::op::v0::Constant>& transpose_order) {
    if (!transpose || !transpose_order)
        return false;
    const auto partial_shape_rank = transpose->get_input_partial_shape(0).rank();
    const auto order = transpose_order->get_axis_vector_val();
    if (partial_shape_rank.is_dynamic() && order.empty())
        return false;
    return true;
}

TransposeInputsInfo GetFirstTransposeInput(
    const NodePtr& node,
    const std::vector<size_t>& indices,
    const std::function<bool(const std::shared_ptr<ov::op::v1::Transpose>& transpose,
                             const std::shared_ptr<ov::op::v0::Constant>& transpose_order)>& if_transpose_sinkable) {
    auto indices_to_check = indices;
    if (indices.empty()) {
        indices_to_check.resize(node->get_input_size());
        std::iota(indices_to_check.begin(), indices_to_check.end(), 0);
    }

    for (const auto& input_idx : indices_to_check) {
        NodePtr input_node = node->get_input_node_shared_ptr(input_idx);
        auto transpose_node = as_type_ptr<ov::op::v1::Transpose>(input_node);
        if (!transpose_node)
            continue;
        auto constant_node = as_type_ptr<ov::op::v0::Constant>(transpose_node->input_value(1).get_node_shared_ptr());
        if (!if_transpose_sinkable(transpose_node, constant_node))
            continue;
        {
            TransposeInputsInfo input_info;
            input_info.transpose = transpose_node;
            input_info.transpose_const = constant_node;
            input_info.input_idx = input_idx;
            return input_info;
        }
    }

    return {};
}

AxisVector ReverseTransposeOrder(const AxisVector& axis_order) {
    AxisVector out(axis_order.size());
    for (size_t i = 0; i < axis_order.size(); i++) {
        out.at(axis_order[i]) = i;
    }
    return out;
}

void SwapOutputNames(Output<Node> output1, Output<Node> output2) {
    const auto node2_output_names = output2.get_names();
    output2.set_names(output1.get_names());
    output1.set_names(node2_output_names);
}

void SwapFriendlyNames(const NodePtr& node1, const NodePtr& node2) {
    const std::string node2_name = node2->get_friendly_name();
    node2->set_friendly_name(node1->get_friendly_name());
    node1->set_friendly_name(node2_name);
}

NodePtr InsertBroadcastUnsqueeze(const Output<Node>& node, size_t n_dims) {
    if (!n_dims)
        return node.get_node_shared_ptr();

    std::vector<size_t> dims(n_dims);
    std::iota(dims.begin(), dims.end(), 0);

    auto unsqueeze_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, Shape{dims.size()}, dims);
    auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(node, unsqueeze_const);
    copy_runtime_info(node.get_node_shared_ptr(), {unsqueeze, unsqueeze_const});
    return unsqueeze;
}

namespace {

bool HasDynamicRankInput(const NodePtr& node) {
    for (auto& input_node : node->input_values()) {
        const ov::Rank output_rank = input_node.get_partial_shape().rank();
        if (output_rank.is_dynamic())
            return true;
    }
    return false;
}

ov::Rank::value_type GetMaxInputRank(const NodePtr& node, const std::vector<size_t>& input_indexes) {
    ov::Rank::value_type max_input_rank = 0;

    for (const auto& idx : input_indexes) {
        const auto& input_node = node->get_input_source_output(idx);
        const ov::Rank output_rank = input_node.get_partial_shape().rank();
        if (output_rank.is_dynamic())
            return -1;
        const ov::Rank::value_type output_rank_len = output_rank.get_length();
        if (output_rank_len > max_input_rank)
            max_input_rank = output_rank_len;
    }
    return max_input_rank;
}

ov::Output<ov::Node> FixInputNodeRank(ov::Output<ov::Node> input_node,
                                      ov::Rank::value_type required_rank,
                                      InsertBroadcastUnsqueezeT InsertUnsqueeze = InsertBroadcastUnsqueeze) {
    auto rank = input_node.get_partial_shape().rank();
    if (rank.is_dynamic()) {
        return input_node;
    }
    const auto output_rank = rank.get_length();
    if (output_rank >= required_rank)
        return input_node;
    return InsertUnsqueeze(input_node, required_rank - output_rank)->output(0);
}

}  // namespace

namespace sink_forward {

namespace {

AxisVector AlignTransposeOrder(const Output<Node>& output, const TransposeInputsInfo& transpose_input_info) {
    if (transpose_input_info.isEmpty()) {
        return {};
    }
    auto num_of_val = static_cast<int64_t>(shape_size(transpose_input_info.transpose_const->get_shape()));
    const auto rank = output.get_partial_shape().rank();
    if (rank.is_dynamic()) {
        return {};
    }
    const auto rank_val = rank.get_length();
    AxisVector new_transpose_order;
    if (rank_val > num_of_val) {
        const auto diff = rank_val - num_of_val;
        new_transpose_order.resize(rank_val);
        std::iota(new_transpose_order.begin(), new_transpose_order.end(), 0);
        auto transpose_axis_order = transpose_input_info.transpose_const->get_axis_vector_val();
        for (int64_t i = diff; i < rank_val; ++i) {
            new_transpose_order[i] = transpose_axis_order[i - diff] + diff;
        }
    } else {
        new_transpose_order = transpose_input_info.transpose_const->get_axis_vector_val();
    }
    return new_transpose_order;
}

}  // namespace

bool UpdateInputTransposes(const NodePtr& main_node,
                           const TransposeInputsInfo& transpose_input_info,
                           std::vector<size_t> input_indexes) {
    if (input_indexes.empty()) {
        input_indexes.resize(main_node->get_input_size());
        std::iota(input_indexes.begin(), input_indexes.end(), 0);
    }
    if (transpose_input_info.isEmpty() || HasDynamicRankInput(main_node))
        return false;

    const auto max_input_rank = GetMaxInputRank(main_node, input_indexes);
    if (max_input_rank < 0)
        return false;

    const size_t transpose_input_index = transpose_input_info.input_idx;
    const auto transpose_element_type = transpose_input_info.transpose_const->get_element_type();

    for (const auto& i : input_indexes) {
        auto input_node = main_node->input_value(i);
        if (i == transpose_input_index) {
            auto transpose_parent = input_node.get_node()->input_value(0);
            main_node->input(i).replace_source_output(transpose_parent);
        } else {
            input_node = FixInputNodeRank(input_node, max_input_rank);
            auto transpose_order = AlignTransposeOrder(input_node, transpose_input_info);
            if (transpose_order.empty()) {
                return false;
            }
            const auto reversed_transpose_axis_order = ReverseTransposeOrder(transpose_order);
            auto new_transpose_const =
                std::make_shared<ov::op::v0::Constant>(transpose_element_type,
                                                       Shape{reversed_transpose_axis_order.size()},
                                                       reversed_transpose_axis_order);
            auto new_transpose = std::make_shared<ov::op::v1::Transpose>(input_node, new_transpose_const);

            main_node->input(i).replace_source_output(new_transpose->output(0));

            copy_runtime_info(input_node.get_node_shared_ptr(), {new_transpose, new_transpose_const});
        }
    }
    return true;
}

void RemoveInputNode(const NodePtr& main_node, size_t input_idx) {
    auto input_node = main_node->input_value(input_idx);
    if (input_node.get_node()->get_input_size() < (input_idx + 1))
        return;
    auto parent_node = input_node.get_node()->input_value(input_idx);
    main_node->input(input_idx).replace_source_output(parent_node);
}

NodeVector InsertOutputTransposes(const NodePtr& main_node, const TransposeInputsInfo& transpose_input_info) {
    if (transpose_input_info.isEmpty())
        return {};
    const auto transpose_axis_order = transpose_input_info.transpose_const->get_axis_vector_val();
    const auto transpose_element_type = transpose_input_info.transpose_const->get_element_type();

    NodeVector new_nodes;

    for (size_t i = 0; i < main_node->get_output_size(); ++i) {
        auto aligned_order = AlignTransposeOrder(main_node->output(i), transpose_input_info);
        auto new_transpose_const =
            std::make_shared<ov::op::v0::Constant>(transpose_element_type, Shape{aligned_order.size()}, aligned_order);
        auto main_node_consumers = main_node->output(i).get_target_inputs();
        auto new_transpose = std::make_shared<ov::op::v1::Transpose>(main_node->output(i), new_transpose_const);
        for (auto& consumer : main_node_consumers) {
            consumer.replace_source_output(new_transpose);
        }
        copy_runtime_info(main_node, {new_transpose, new_transpose_const});
        SwapOutputNames(main_node->output(i), new_transpose->output(0));
        if (main_node->get_output_size() > 1)
            new_transpose->set_friendly_name(main_node->get_friendly_name() + "." + std::to_string(i));
        else
            SwapFriendlyNames(new_transpose, main_node);

        new_nodes.push_back(new_transpose);
    }

    return new_nodes;
}

}  // namespace sink_forward

namespace sink_backward {
NodeVector InsertTransposeBeforeNode(const NodePtr& main_node,
                                     const std::shared_ptr<ov::op::v0::Constant>& transpose_const,
                                     std::vector<size_t> input_indexes,
                                     InsertBroadcastUnsqueezeT InsertUnsqueeze) {
    if (input_indexes.empty()) {
        input_indexes.resize(main_node->get_input_size());
        std::iota(input_indexes.begin(), input_indexes.end(), 0);
    }
    const auto transpose_axis_order = transpose_const->get_axis_vector_val();
    const auto transpose_element_type = transpose_const->get_element_type();

    if (HasDynamicRankInput(main_node))
        return {};

    NodeVector new_nodes;

    const auto max_input_rank = GetMaxInputRank(main_node, input_indexes);
    if (max_input_rank < 0)
        return {};

    for (const auto& i : input_indexes) {
        auto input_node = FixInputNodeRank(main_node->input_value(i), max_input_rank, InsertUnsqueeze);

        auto new_transpose_const = std::make_shared<ov::op::v0::Constant>(transpose_element_type,
                                                                          Shape{transpose_axis_order.size()},
                                                                          transpose_axis_order);
        auto new_transpose = std::make_shared<ov::op::v1::Transpose>(input_node, new_transpose_const);

        main_node->input(i).replace_source_output(new_transpose->output(0));

        copy_runtime_info(input_node.get_node_shared_ptr(), {new_transpose, new_transpose_const});

        new_nodes.push_back(new_transpose);
    }

    return new_nodes;
}
}  // namespace sink_backward

namespace {

std::shared_ptr<ov::op::v0::Constant> GetTransposeConstant(Node* node) {
    auto transpose_node = ov::as_type<ov::op::v1::Transpose>(node);
    if (!transpose_node)
        return {};

    auto constant_node = as_type_ptr<ov::op::v0::Constant>(transpose_node->input_value(1).get_node_shared_ptr());
    if (!constant_node)
        return {};

    return constant_node;
}

Node* FindFirstConsumer(const NodePtr& node) {
    for (size_t output_idx = 0; output_idx < node->get_output_size(); ++output_idx) {
        auto inputs = node->get_output_target_inputs(output_idx);
        if (inputs.empty())
            continue;
        return inputs.begin()->get_node();
    }
    return nullptr;
}

bool CheckTransposeConsumers(const NodePtr& main_node) {
    AxisVector first_transpose_axis_order;
    {
        Node* first_consumer = FindFirstConsumer(main_node);
        if (!first_consumer)
            return false;
        auto constant_node = GetTransposeConstant(first_consumer);
        if (!constant_node)
            return false;
        first_transpose_axis_order = constant_node->get_axis_vector_val();
    }

    for (size_t output_idx = 0; output_idx < main_node->get_output_size(); ++output_idx) {
        for (auto& input : main_node->get_output_target_inputs(output_idx)) {
            auto constant_node = GetTransposeConstant(input.get_node());
            if (!constant_node)
                return false;

            AxisVector transpose_axis_order = constant_node->get_axis_vector_val();
            if (transpose_axis_order.size() != first_transpose_axis_order.size())
                return false;
            if (!std::equal(transpose_axis_order.begin(),
                            transpose_axis_order.end(),
                            first_transpose_axis_order.begin()))
                return false;
            if (!is_sinking_node(input.get_node())) {
                return false;
            }
        }
    }

    return true;
}

}  // namespace

bool CheckTransposeConsumers(const Output<Node>& output) {
    return CheckTransposeConsumers(output.get_node_shared_ptr());
}

bool RemoveTransposeConsumers(const NodePtr& node) {
    std::unordered_map<size_t, std::vector<ov::op::v1::Transpose*>> out_idx_to_redundant_transposes;

    // in case of multiple Transposes connected directly to Result ops,
    // we can't guarantee that friendly names are copied correctly,
    // we preserve only one of possible variants.
    // This note related to friendly names only, not to tensor names.
    ov::op::v1::Transpose* transpose_connected_to_result = nullptr;
    for (size_t output_idx = 0; output_idx < node->get_output_size(); ++output_idx) {
        for (auto& consumer_input : node->get_output_target_inputs(output_idx)) {
            auto transpose = ov::as_type<ov::op::v1::Transpose>(consumer_input.get_node());
            if (!transpose) {
                // should never happen
                // the check that all consumers of the main node are Transposes is added
                // to the pattern of the transformations
                OPENVINO_ASSERT(false, "TransposeSinking error: attempted to remove not Transpose consumer.");
            }
            out_idx_to_redundant_transposes[output_idx].push_back(transpose);

            for (const auto& transpose_consumer_input : transpose->output(0).get_target_inputs()) {
                if (ov::as_type<ov::op::v0::Result>(transpose_consumer_input.get_node())) {
                    transpose_connected_to_result = transpose;
                }
            }
        }
    }

    if (transpose_connected_to_result) {
        node->set_friendly_name(transpose_connected_to_result->get_friendly_name());
    } else if (out_idx_to_redundant_transposes.count(0) && !out_idx_to_redundant_transposes[0].empty()) {
        // if no transpose connected to result op found
        // we save any friendly name
        node->set_friendly_name((*out_idx_to_redundant_transposes[0].begin())->get_friendly_name());
    }

    for (const auto& key_value : out_idx_to_redundant_transposes) {
        for (const auto& transpose : key_value.second) {
            transpose->output(0).replace(node->output(key_value.first));
        }
    }

    return true;
}

std::vector<size_t> GetOrderAfterReduction(const std::vector<size_t>& axes_values,
                                           const std::vector<size_t>& order_values) {
    size_t buffer_size = order_values.size() - axes_values.size();
    std::vector<size_t> aligned_order(buffer_size, 0);
    std::vector<size_t> values_to_reduce(axes_values);
    for (size_t i = 0; i < values_to_reduce.size(); ++i) {
        values_to_reduce[i] = order_values[axes_values[i]];
    }
    std::sort(values_to_reduce.begin(), values_to_reduce.end());
    for (size_t i = 0, j = 0; i < order_values.size(); ++i) {
        if (std::find(axes_values.begin(), axes_values.end(), i) != axes_values.end()) {
            continue;
        }

        auto lb = std::lower_bound(values_to_reduce.begin(), values_to_reduce.end(), order_values[i]);
        aligned_order[j] = order_values[i] - (lb - values_to_reduce.begin());
        ++j;
    }
    return aligned_order;
}

std::vector<size_t> GetOrderBeforeReduction(const std::vector<size_t>& axes_values,
                                            const std::vector<size_t>& order_values) {
    size_t buffer_size = order_values.size() + axes_values.size();
    std::vector<size_t> aligned_order(buffer_size);

    std::vector<int64_t> cnt_deleted(buffer_size);
    int64_t cnt = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(cnt_deleted.size()); ++i) {
        if (std::find(axes_values.begin(), axes_values.end(), i) != axes_values.end()) {
            cnt++;
        }
        cnt_deleted[i] = i - cnt;
    }

    for (size_t i = 0, j = 0; i < aligned_order.size(); ++i) {
        if (std::find(axes_values.begin(), axes_values.end(), i) != axes_values.end()) {
            aligned_order[i] = i;
            continue;
        }

        aligned_order[i] = std::find(cnt_deleted.begin(), cnt_deleted.end(), order_values[j]) - cnt_deleted.begin();
        ++j;
    }
    return aligned_order;
}

}  // namespace utils
}  // namespace transpose_sinking
}  // namespace pass
}  // namespace ov
