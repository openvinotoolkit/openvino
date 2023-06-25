// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>
#include <utility>

#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"

namespace gather_sinking {

struct GatherInputsInfo {
    std::shared_ptr<ov::opset12::Gather> gather;
    std::shared_ptr<ov::opset12::Constant> indices_const;
    std::shared_ptr<ov::opset12::Constant> axis_const;
    size_t input_idx;

    bool isEmpty() const {
        return !gather || !indices_const || !axis_const;
    }
};

/**
 * @brief Finds node first input that is a Gather operation and returns filled GatherInputsInfo
 * for it
 */
GatherInputsInfo get_first_gather_input(std::shared_ptr<ov::Node>);

/**
 * @brief Checks if @arg has any input node that is a Gather operation
 */
template <typename GatherInfoPredicate>
bool if_node_has_gather_inputs(const ov::Output<ov::Node>& output, GatherInfoPredicate gather_info_predicate) {
    GatherInputsInfo inputs_info = get_first_gather_input(output.get_node_shared_ptr());
    if (inputs_info.isEmpty())
        return false;

    return gather_info_predicate(inputs_info);
}

namespace sink_forward {
/**
 * @brief Inserts reversed Gather on @args main_node inputs. Removes input Gather specified in @arg
 * transpose_input_info
 */
void update_input_gather(std::shared_ptr<ov::Node> main_node,
                         const GatherInputsInfo&,
                         const int64_t* a_gather_negative_axis = nullptr);

/**
 * @brief Inserts Gather on each main_node output with the order specified in @arg GatherInputsInfo
 */
ov::NodeVector insert_output_gather(std::shared_ptr<ov::Node> main_node, const GatherInputsInfo&);
}  // namespace sink_forward

namespace sink_backward {
/**
 * @brief Inserts Gather layers on each input of @arg main_node with cloned indices and axes constants
 */
ov::NodeVector insert_gather_before_node(std::shared_ptr<ov::Node> main_node,
                                         const std::shared_ptr<ov::opset12::Constant>& indices_const,
                                         const std::shared_ptr<ov::opset12::Constant>& axes_const,
                                         const std::shared_ptr<ov::opset12::Gather>& gather_node,
                                         std::vector<int> input_indexes = {});
}  // namespace sink_backward

void update_forward_gather_sinking_ability(std::shared_ptr<ov::Node>);

/**
 *  @brief Checks if @arg has consumers that all are the same Gather operation. If no consumers at all
 *  returns false.
 */
bool has_same_output_gather_nodes(const ov::Output<ov::Node>&);

/**
 * Removes all direct node consumers that have one output
 */
void remove_single_output_consumers(std::shared_ptr<ov::Node>);

/**
 * @brief Checks if Output is Gather node and it could be sinked
 */
bool is_gather_sinking_enabled(const ov::Output<ov::Node>& output);

/**
 * @brief Finds Split first input Gather node and it could be sinked
 */
bool is_split_sinked(const ov::Output<ov::Node>& output);

/**
 * @brief Converts Gather indices to negative form
 */
int64_t normalize_negative_gather_axis(int64_t axis, ov::Rank::value_type gather_input_rank);

/**
 * @brief Gets Gather indices from Constant converted into negative form
 */
int64_t get_normalized_negative_gather_axis(const std::shared_ptr<ov::opset12::Constant>& axis,
                                            ov::Rank::value_type gather_input_rank);

/**
 * @brief Gets Gather axis if it's stored in a constant
 */
bool get_gather_axis(const std::shared_ptr<ov::Node>& gather, int64_t& axis);

}  // namespace gather_sinking
