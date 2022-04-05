// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/error.hpp"
#include "vpu/ngraph/transformations/extract_dynamic_batch/slice_binary_eltwise.hpp"

namespace vpu {

SliceConfiguration sliceBinaryEltwise(const ngraph::Node& node) {
    const auto& eltwise = dynamic_cast<const ngraph::op::util::BinaryElementwiseArithmetic&>(node);
    VPU_THROW_UNLESS(eltwise.get_input_size() == 2,  "Expecting operation {} to have {} inputs, got {}", node, 2, eltwise.get_input_size());
    VPU_THROW_UNLESS(eltwise.get_output_size() == 1, "Expecting operation {} to have {} outputs, got {}", node, 1, eltwise.get_output_size());

    const auto& lhs = eltwise.input_value(0);
    const auto& rhs = eltwise.input_value(1);
    const auto& out = eltwise.output(0);

    const auto& lhsPartialShape = lhs.get_partial_shape();
    const auto& rhsPartialShape = rhs.get_partial_shape();
    const auto& outPartialShape = out.get_partial_shape();

    const auto& broadcastSpec = eltwise.get_autob();
    auto inputPartialShape = lhsPartialShape;
    if (broadcastSpec == ngraph::op::AutoBroadcastType::NONE) {
        ngraph::PartialShape::merge_into(inputPartialShape, rhsPartialShape);
    } else {
        ngraph::PartialShape::broadcast_merge_into(inputPartialShape, rhsPartialShape, broadcastSpec);
    }

    const auto& inputRank = inputPartialShape.rank();
    const auto& lhsRank   = lhsPartialShape.rank();
    const auto& rhsRank   = rhsPartialShape.rank();
    const auto& outRank   = outPartialShape.rank();

    VPU_THROW_UNLESS(inputRank == outRank && inputRank.is_static(),
                     "Expecting operation {} to have the same static rank for inputs and output, got merged inputs rank = {}, output rank = {}",
                     node, inputRank, outRank);

    const auto& inputRankLength = inputRank.get_length();
    const auto& lhsRankLength   = lhsRank.get_length();
    const auto& rhsRankLength   = rhsRank.get_length();
    const auto& outRankLength   = outRank.get_length();

    const auto& inputsBatch = inputRankLength > 0 ? inputPartialShape[0] : 0;
    const auto& outBatch = outRankLength > 0 ? outPartialShape[0] : 0;
    VPU_THROW_UNLESS(inputsBatch == outBatch,
                     "Expecting operation {} to have the same batch on both inputs and output, got input batch = {}, output batch = {}",
                     node, inputsBatch, outBatch);


    if (inputsBatch.is_static() && inputsBatch.get_length() == 1) {
        return {};
    }

    const auto& maxRankInputPartialShape = lhsRankLength == inputRankLength ? lhsPartialShape : rhsPartialShape;
    const auto& minRankInputPartialShape = lhsRankLength == inputRankLength ? rhsPartialShape : lhsPartialShape;

    const auto checkPartialShape = [](const ngraph::PartialShape& partialShape) {
        const auto dynamicDimensionsCount = std::count_if(partialShape.cbegin(), partialShape.cend(),
                                                          [](const ngraph::Dimension& dimension) { return dimension.is_dynamic(); });
        return dynamicDimensionsCount == 0 || (dynamicDimensionsCount == 1 && partialShape[0].is_dynamic());
    };

    const auto isMaxRankInputOk = checkPartialShape(maxRankInputPartialShape);
    const auto isMinRankInputOk = minRankInputPartialShape.rank().get_length() == maxRankInputPartialShape.rank().get_length()
                                  ? checkPartialShape(minRankInputPartialShape)
                                  : minRankInputPartialShape.is_static();
    if (!isMaxRankInputOk || !isMinRankInputOk) {
        return {};
    }

    const auto lhsSplitMode = lhsRankLength < inputRankLength || lhsPartialShape[0] != inputPartialShape[0] ? SliceMode::Unchanged : SliceMode::Slice;
    const auto rhsSplitMode = rhsRankLength < inputRankLength || rhsPartialShape[0] != inputPartialShape[0] ? SliceMode::Unchanged : SliceMode::Slice;

    return {{lhsSplitMode, rhsSplitMode}, {SliceMode::Slice}};
}

}  // namespace vpu
