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

TransposeInputsInfo GetFirstTransposeInput(std::shared_ptr<ov::Node> node);
bool IfNodeHasTransposeInputs(const ov::Output<ov::Node>& output);
ov::AxisVector ReverseTransposeOrder(const ov::AxisVector& axis_order);
void SwapOutputNames(ov::Output<ov::Node> output1, ov::Output<ov::Node> output2);
void SwapFriendlyNames(std::shared_ptr<ov::Node> node1, std::shared_ptr<ov::Node> node2);
void SwapNames(std::shared_ptr<ov::Node> node1, std::shared_ptr<ov::Node> node2);

namespace sink_forward {
// insert input reversed transposes, remove first input tranpose
void UpdateInputTransposes(std::shared_ptr<ov::Node> main_node, TransposeInputsInfo& transpose_input_info);
void RemoveZeroInputNode(std::shared_ptr<ov::Node> main_node);
ov::NodeVector InsertOutputTransposes(std::shared_ptr<ov::Node> main_node, TransposeInputsInfo& transpose_input_info);
}  // namespace sink_forward

namespace sink_backward {
ov::NodeVector InsertTransposeBeforeNode(std::shared_ptr<ov::Node> main_node,
                                         std::shared_ptr<ov::opset9::Constant> transpose_const);
}  // namespace sink_backward

void UpdateForwardSinkingAbility(std::shared_ptr<ov::Node>);

}  // namespace transpose_sinking
