// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/opsets/opset9.hpp>
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

using namespace ov;
using namespace ov::opset9;

using NodePtr = std::shared_ptr<Node>;

struct TransposeInputsInfo {
    std::shared_ptr<Transpose> transpose;
    std::shared_ptr<Constant> transpose_const;
    size_t input_idx;

    bool isEmpty() const {
        return !transpose || !transpose_const;
    }
};

TransposeInputsInfo GetFirstTransposeInput(NodePtr node);
bool IfNodeHasTransposeInputs(const Output<Node>& output);
AxisVector ReverseTransposeOrder(const AxisVector& axis_order);
void SwapOutputNames(Output<Node> output1, Output<Node> output2);
void SwapFriendlyNames(NodePtr node1, NodePtr node2);
void SwapNames(NodePtr node1, NodePtr node2);

namespace sink_forward {
// insert input reversed transposes, remove first input tranpose
void UpdateInputTransposes(NodePtr main_node, TransposeInputsInfo& transpose_input_info);
void RemoveZeroInputNode(NodePtr main_node);
NodeVector InsertOutputTransposes(NodePtr main_node, TransposeInputsInfo& transpose_input_info);
}  // namespace sink_forward

namespace sink_backward {
NodeVector InsertTransposeBeforeNode(NodePtr main_node, std::shared_ptr<Constant> transpose_const);
}  // namespace sink_backward

bool IsSinkingEnable(NodePtr);
bool IsSinkingEnable(Node *);
void UpdateForwardSinkingAbility(NodePtr);

}  // namespace transpose_sinking
