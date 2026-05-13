// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/reshape.hpp>

#include "../convert_common.hpp"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace mlir {

struct ConvertReshape {
    Operation* operator()(ConversionContext& context, NodePtr node) {
        const auto in_shape = node->get_input_partial_shape(0);
        const auto out_shape = node->get_output_partial_shape(0);
        assert(in_shape.rank().is_static() && out_shape.rank().is_static());
        assert(llvm::all_of(in_shape, std::mem_fn(&ov::Dimension::is_static)));
        assert(llvm::all_of(out_shape, std::mem_fn(&ov::Dimension::is_static)));
        const auto in_rank = static_cast<size_t>(in_shape.rank().get_length());
        const auto out_rank = static_cast<size_t>(out_shape.rank().get_length());
        const bool expand = out_rank >= in_rank;

        // Build reassociation by matching accumulated products of src/dst dims.
        // Each group maps one src dim to multiple dst dims (expand) or vice versa (collapse).
        SmallVector<ReassociationIndices> reassociation;
        for (size_t src_i = 0, dst_i = 0; src_i < in_rank && dst_i < out_rank; src_i++, dst_i++) {
            ReassociationIndices group;
            int64_t src_prod = in_shape[src_i].get_length();
            int64_t dst_prod = out_shape[dst_i].get_length();
            if (expand) {
                // one src dim -> multiple dst dims
                group.push_back(dst_i);
                while (src_prod != dst_prod && dst_i < out_rank) {
                    dst_prod *= out_shape[++dst_i].get_length();
                    group.push_back(dst_i);
                }
            } else {
                // multiple src dims -> one dst dim
                group.push_back(src_i);
                while (src_prod != dst_prod && src_i < in_rank) {
                    src_prod *= in_shape[++src_i].get_length();
                    group.push_back(src_i);
                }
            }
            assert(src_prod == dst_prod && "shape mismatch: incompatible reshape");
            reassociation.push_back(group);
        }

        auto dst_shape = llvm::to_vector(llvm::map_range(out_shape, std::mem_fn(&ov::Dimension::get_length)));
        auto dst_type =
            RankedTensorType::get(dst_shape, importPrecision(context.context, node->get_output_element_type(0)));
        auto loc = createLocation(context.context, node);
        auto& builder = context.builder();
        const auto input = context.getInputs(node)[0];
        return expand ? tensor::ExpandShapeOp::create(builder, loc, dst_type, input, reassociation)
                      : tensor::CollapseShapeOp::create(builder, loc, dst_type, input, reassociation);
    }
};

}  // namespace mlir
}  // namespace ov
