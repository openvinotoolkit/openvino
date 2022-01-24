// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/error.hpp"
#include "vpu/ngraph/transformations/extract_dynamic_batch/slice_unary_eltwise.hpp"

namespace vpu {

SliceConfiguration sliceUnaryEltwise(const ngraph::Node& node) {
    VPU_THROW_UNLESS(node.get_input_size() == 1,  "Expecting unary eltwise operation {} to have {} inputs, got {}", node, 1, node.get_input_size());
    VPU_THROW_UNLESS(node.get_output_size() == 1, "Expecting unary eltwise operation {} to have {} outputs, got {}", node, 1, node.get_output_size());

    const auto& inp = node.input_value(0);
    const auto& out = node.output(0);

    const auto& inpPartialShape = inp.get_partial_shape();
    const auto& outPartialShape = out.get_partial_shape();

    const auto& inpRank = inpPartialShape.rank();
    const auto& outRank = outPartialShape.rank();

    VPU_THROW_UNLESS(inpRank == outRank,
        "Expecting unary eltwise operation {} to have the same static rank for input and output, got input rank = {}, output rank = {}",
        node, inpRank, outRank);

    const auto& inpRankLength = inpRank.get_length();
    const auto& outRankLength = outRank.get_length();

    const auto& inpBatch = inpRankLength > 0 ? inpPartialShape[0] : 0;
    const auto& outBatch = outRankLength > 0 ? outPartialShape[0] : 0;
    VPU_THROW_UNLESS(inpBatch == outBatch,
        "Expecting unary eltwise operation {} to have the same batch on input and output, got input batch = {}, output batch = {}",
        node, inpBatch, outBatch);

    if (inpBatch.is_static() && inpBatch.get_length() == 1) {
        return {};
    }

    const auto dynamicDimensionsCount = std::count_if(inpPartialShape.cbegin(), inpPartialShape.cend(),
        [](const ngraph::Dimension& dimension) { return dimension.is_dynamic(); });
    if (dynamicDimensionsCount > 1 || (dynamicDimensionsCount == 1 && inpPartialShape[0].is_static())) {
        return {};
    }

    return {{SliceMode::Slice}, {SliceMode::Slice}};
}

}  // namespace vpu
