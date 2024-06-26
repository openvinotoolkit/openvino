// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "snippets_mark_skipped_base.hpp"

#include "snippets/pass/tokenization.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/utils.hpp"

#include "transformations/utils/utils.hpp"
#include "transformations/utils.hpp"
#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

#include "itt.hpp"


namespace ov {
namespace intel_cpu {

bool SnippetsMarkSkippedBase::canBePerformedAsScaleShift(const std::shared_ptr<const Node> &node, const int channelAxis) {
    size_t fusingPort = 0;
    size_t numNonConstInputs = 0;
    ov::PartialShape dataShape;
    for (size_t i = 0; i < node->get_input_size(); i++) {
        const auto parent = node->get_input_node_shared_ptr(i);
        if (!ov::is_type<ov::op::v0::Constant>(parent)) {
            fusingPort = i;
            dataShape = node->get_input_partial_shape(i);
            // only one non-const parent is allowed
            if (++numNonConstInputs != 1)
                return false;
        } else {
            // every const parent must have exactly one child
            const auto out = parent->outputs();
            const bool has_only_child = (out.size() == 1) && (out[0].get_target_inputs().size() == 1);
            if (!has_only_child)
                return false;
        }
    }

    const auto isBroadcastableToDataInput = [&]() {
        for (size_t i = 0; i < node->get_input_size(); i++) {
            if (i == fusingPort)
                continue;
            const ov::PartialShape weightShape = node->get_input_partial_shape(i);
            if (!isPerTensorOrPerChannelBroadcastable(dataShape.get_max_shape(), weightShape.get_max_shape(), channelAxis, true))
                return false;
        }
        return true;
    };

    // Prelu and MulAdd are still ignored
    // isConvertablePowerStatic() is ignored
    return (ov::is_type<ov::opset1::Add>(node) ||
            ov::is_type<ov::opset1::Multiply>(node) ||
            ov::is_type<ov::opset1::Subtract>(node) ||
            ov::is_type<ov::opset1::Divide>(node)) &&
           isBroadcastableToDataInput();
}

}   // namespace intel_cpu
}   // namespace ov
