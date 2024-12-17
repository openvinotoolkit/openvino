// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <openvino/op/squeeze.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "squeeze.hpp"
#include "../convert_common.hpp"


namespace {

using namespace ov::mlir;

struct ConvertSqueeze {
    void operator()(ConversionContext& context, NodePtr node) {
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

        auto reshape = builder.create<tensor::CollapseShapeOp>(loc, input, collapse_groups);
        context.addOutputs(node, reshape);
    }
};

}  // namespace

namespace ov {
namespace mlir {

using namespace ov::pass::pattern;
using namespace ov::op;

SqueezePattern::SqueezePattern() : MarkPattern(wrap_type<v0::Squeeze>({any_input()}), ConvertSqueeze()) {}

}  // namespace mlir
}  // namespace ov
