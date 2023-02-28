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
    std::shared_ptr<ov::opset9::Constant> axes_const;
    size_t input_idx;

    bool isEmpty() const {
        return !gather || !indices_const || !axes_const;
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
bool IfNodeHasGatherInputs(const ov::Output<ov::Node>&);

namespace sink_backward {
/**
 * @brief Inserts Gather layers on each input of @arg main_node with cloned indices and axes constants
 */
ov::NodeVector InsertGatherBeforeNode(std::shared_ptr<ov::Node> main_node,
                                      const std::shared_ptr<ov::opset9::Constant>& indices_const,
                                      const std::shared_ptr<ov::opset9::Constant>& axes_const);
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

}  // namespace gather_sinking
