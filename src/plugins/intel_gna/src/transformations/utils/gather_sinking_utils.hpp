// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>
#include <utility>

#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"

namespace gather_sinking {

struct GatherInputsInfo {
    std::shared_ptr<ov::opset9::Gather> gather;
    std::shared_ptr<ov::opset9::Constant> indices_const;
    std::shared_ptr<ov::opset9::Constant> axis_const;
    size_t input_idx;

    bool isEmpty() const {
        return !gather || !indices_const || !axis_const;
    }
};

/**
 * @brief Finds node first input that is a Gather operation and returns filled GatherInputsInfo
 * for it
 */
GatherInputsInfo GetFirstGatherInput(std::shared_ptr<ov::Node>);

/**
 * @brief Checks if @arg has any input node that is a Gather operation
 */
template <typename GatherInfoPredicate>
bool IfNodeHasGatherInputs(const ov::Output<ov::Node>& output, GatherInfoPredicate gather_info_predicate) {
    GatherInputsInfo inputs_info = GetFirstGatherInput(output.get_node_shared_ptr());
    if (inputs_info.isEmpty())
        return false;
    
    return gather_info_predicate(inputs_info);
}

/**
 * @brief Swaps @args output tensor names
 */
void SwapOutputNames(ov::Output<ov::Node>, ov::Output<ov::Node>);

/**
 * @brief Swaps @args friendly names
 */
void SwapFriendlyNames(std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>);

/**
 * @brief Swaps @args output tensor names and friendly names
 */
void SwapNames(std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>);

namespace sink_forward {
/**
 * @brief Inserts reversed Gather on @args main_node inputs. Removes input Gather specified in @arg
 * transpose_input_info
 */
void UpdateInputGather(std::shared_ptr<ov::Node> main_node, const GatherInputsInfo&);

/**
 * @brief Removes @arg input node
 */
void RemoveInputNode(std::shared_ptr<ov::Node>, size_t input_idx);

/**
 * @brief Inserts Gather on each main_node output with the order specified in @arg GatherInputsInfo
 */
ov::NodeVector InsertOutputGather(std::shared_ptr<ov::Node> main_node,
                                      const GatherInputsInfo&);
}  // namespace sink_forward


namespace sink_backward {
/**
 * @brief Inserts Gather layers on each input of @arg main_node with cloned indices and axes constants
 */
ov::NodeVector InsertGatherBeforeNode(std::shared_ptr<ov::Node> main_node,
                                      const std::shared_ptr<ov::opset9::Constant>& indices_const,
                                      const std::shared_ptr<ov::opset9::Constant>& axes_const,
                                      const std::shared_ptr<ov::opset9::Gather>& gather_node);
}  // namespace sink_backward

void UpdateForwardGatherSinkingAbility(std::shared_ptr<ov::Node>);

/**
 *  @brief Checks if @arg has consumers that all are the same Gather operation. If no consumers at all
 *  returns false.
 */
bool HasSameOutputGatherNodes(const ov::Output<ov::Node>&);

/**
 * Removes all direct node consumers that have one output
 */
void RemoveSingleOutputConsumers(std::shared_ptr<ov::Node>);

bool constant_has_rank_not_more_than(const std::shared_ptr<ov::opset9::Constant>&, const ov::Rank::value_type expected_rank);

/**
 * Checks if output has rank not more than expected
*/
std::function<bool(ov::Output<ov::Node>)> rank_not_more_than(const ov::Rank::value_type expected_rank);

}  // namespace gather_sinking
