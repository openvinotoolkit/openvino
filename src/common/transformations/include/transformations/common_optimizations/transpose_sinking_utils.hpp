// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>
#include <utility>

#include "itt.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"

namespace transpose_sinking {

struct TransposeInputsInfo {
    std::shared_ptr<ov::opset9::Transpose> transpose;
    std::shared_ptr<ov::opset9::Constant> transpose_const;
    size_t input_idx;

    bool isEmpty() const {
        return !transpose || !transpose_const;
    }
};

/**
 * @brief Find node first input that is a transpose operation and return filled TransposeInputsInfo
 * for it
 */
TransposeInputsInfo GetFirstTransposeInput(std::shared_ptr<ov::Node>);

/**
 * @brief Check if @arg has any input node that is a transpose operation
 */
bool IfNodeHasTransposeInputs(const ov::Output<ov::Node>&);

/**
 * @brief Reverse order of transpose operation. Do it in a such way that if we had couple following one after
 * another transposes (one would be reversed version of another) we will have no transpose as a result of that
 * couple of transposes.
 */
ov::AxisVector ReverseTransposeOrder(const ov::AxisVector&);

/**
 * @brief Swap @args output tensor names
 */
void SwapOutputNames(ov::Output<ov::Node>, ov::Output<ov::Node>);

/**
 * @brief Swap @args friendly names
 */
void SwapFriendlyNames(std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>);

/**
 * @brief Swap @args output tensor names and friendly names
 */
void SwapNames(std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>);

/**
 * @brief Clone @arg node with the same inputs as original. Reconnect all output consumers to a cloned node
 * except consumers specified as @args consumers.
 */
std::shared_ptr<ov::Node> CloneNodeWithoutConsumers(std::shared_ptr<ov::Node> node, ov::NodeVector consumers);

namespace sink_forward {
// insert input reversed transposes, remove first input tranpose
/**
 * @brief Insert reversed transposed on @args main_node inputs. Remove input transpose specified in @arg transpose_input_info
 */
void UpdateInputTransposes(std::shared_ptr<ov::Node> main_node, TransposeInputsInfo& transpose_input_info);

/**
 * @brief Remove @arg input node
 */
void RemoveInputNode(std::shared_ptr<ov::Node>, size_t input_idx);

/**
 * @brief Insert transposes on each main_node output with the order specified in @arg transpose_input_info
 */
ov::NodeVector InsertOutputTransposes(std::shared_ptr<ov::Node> main_node, TransposeInputsInfo& transpose_input_info);
}  // namespace sink_forward

namespace sink_backward {
/**
 * @brief Insert transposes on each input of @arg main_node with the order specified in @arg transpose_const
 */
ov::NodeVector InsertTransposeBeforeNode(std::shared_ptr<ov::Node> main_node,
                                         std::shared_ptr<ov::opset9::Constant> transpose_const);
}  // namespace sink_backward

void UpdateForwardSinkingAbility(std::shared_ptr<ov::Node>);

}  // namespace transpose_sinking
