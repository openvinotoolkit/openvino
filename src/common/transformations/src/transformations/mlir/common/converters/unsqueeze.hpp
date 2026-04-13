// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <openvino/op/constant.hpp>
#include <openvino/op/unsqueeze.hpp>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "../convert_common.hpp"


namespace ov {
namespace mlir {

struct ConvertUnsqueeze {
    Operation* operator()(ConversionContext& context, NodePtr node) {
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto input = context.getInputs(node)[0];
        // TODO: support dynamic inputs
        // const auto axes = context.getInputs(node)[1];

        const auto ov_output_element_type = node->get_output_element_type(0);
        const auto ov_input_shape = node->get_input_partial_shape(0);

        assert(ov_input_shape.rank().is_static() && "expecting static output shape");

        auto const_axes = dynamic_cast<ov::op::v0::Constant*>(node->get_input_node_ptr(1));
        assert(const_axes && "non-const axes not supported");
        ov::Coordinate coords = const_axes->get_coordinate_val();

        // Calculate the resulting shape.
        // E.g., for an input tensor<4x2xf32> and axes [0, 2] (tensor<2xi64>) need to build a shape 1x4x1x2
        SmallVector<ReassociationIndices> expand_groups;
        ReassociationIndices group = ReassociationIndices();
        SmallVector<int64_t> shape(coords.size() + ov_input_shape.rank().get_length());
        for (size_t input_idx = 0, coord_idx = 0, i = 0; i < shape.size(); ++i) {
            group.push_back(i);
            if (coord_idx < coords.size() && i == coords[coord_idx]) {
                shape[i] = 1;
                coord_idx++;
            } else {
                const auto& dim = ov_input_shape[input_idx];
                shape[i] = dim.is_dynamic() ? ShapedType::kDynamic : dim.get_length();
                input_idx++;
                expand_groups.push_back(group);
                group = ReassociationIndices();
            }
        }

        auto result_type = RankedTensorType::get(shape, importPrecision(context.context, ov_output_element_type));
        auto expand_shape = tensor::ExpandShapeOp::create(builder, loc, result_type, input, expand_groups);
        return expand_shape;
    }
};

}  // namespace mlir
}  // namespace ov

