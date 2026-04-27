// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <openvino/op/squeeze.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "../convert_common.hpp"


namespace ov {
namespace mlir {

struct ConvertSqueeze {
    Operation* operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto input = context.getInputs(node)[0];

        auto src_partial_shape = node->get_input_partial_shape(0);
        auto src_rank = src_partial_shape.rank().get_length();
        SmallVector<ReassociationIndices> collapse_groups;
        ReassociationIndices group = ReassociationIndices();
        for (size_t src_i = 0; src_i < src_rank; src_i++) {
            auto src_d = src_partial_shape[src_i];
            group.push_back(src_i);
            if (src_d.is_static() && src_d.get_length() == 1) {
                // continue collecting
            } else {
                collapse_groups.emplace_back(group);
                group = ReassociationIndices();
            }
        }

        auto reshape = tensor::CollapseShapeOp::create(builder, loc, input, collapse_groups);
        return reshape;
    }
};

}  // namespace mlir
}  // namespace ov
