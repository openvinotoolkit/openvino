// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/error.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "vpu/ngraph/transformations/extract_dynamic_batch/slice_mat_mul.hpp"

namespace vpu {

SliceConfiguration sliceMatMul(const ngraph::Node& node) {
    VPU_THROW_UNLESS(node.get_input_size() == 2,  "Expecting operation {} to have {} inputs, got {}", node, 2, node.get_input_size());
    VPU_THROW_UNLESS(node.get_output_size() == 1, "Expecting operation {} to have {} outputs, got {}", node, 1, node.get_output_size());

    // target networks have MatMul only with constant second input
    // there are tests on dynamic MatMul with non-constant second input
    // if try to process MatMul with non-constant second input it will
    // affect tests and they will fail, since Loop support is not ready yet
    if (!ngraph::op::is_constant(node.input_value(1).get_node_shared_ptr())) {
        return {};
    }

    const auto& lhs = node.input_value(0);
    const auto& lhsPartialShape = lhs.get_partial_shape();
    const auto& lhsRank = lhsPartialShape.rank();
    VPU_THROW_UNLESS(lhsRank.is_static(), "Expecting operation {} to have static rank for input {}, got {}", node, lhs, lhsPartialShape);

    const auto& rhs = node.input_value(0);
    const auto& rhsPartialShape = rhs.get_partial_shape();
    const auto& rhsRank = rhsPartialShape.rank();
    VPU_THROW_UNLESS(rhsRank.is_static(), "Expecting operation {} to have static rank for input {}, got {}", node, rhs, rhsPartialShape);

    const auto& lhsRankLength = lhsRank.get_length();
    const auto& rhsRankLength = rhsRank.get_length();

    const auto maxRankLength = std::max(lhsRankLength, rhsRankLength);
    if (maxRankLength < 3) {
        return {};
    }

    const auto isBatchStatic = [](const ngraph::PartialShape& shape) {
        const auto& rank = shape.rank();
        if (rank.is_dynamic()) {
            return false;
        }
        const auto rankLength = rank.get_length();
        if (rankLength < 3) {
            return true;
        }
        return std::all_of(shape.rbegin() + 2, shape.rend(), [](const ngraph::Dimension& dimension) { return dimension.is_static(); });
    };

    if (maxRankLength > 3) {
        VPU_THROW_UNLESS(isBatchStatic(lhsPartialShape), "Encountered multi-dimensional dynamic batch for operation {}, but it's unsupported", node);
        VPU_THROW_UNLESS(isBatchStatic(rhsPartialShape), "Encountered multi-dimensional dynamic batch for operation {}, but it's unsupported", node);
        return {};
    }

    if (isBatchStatic(lhsPartialShape) && isBatchStatic(rhsPartialShape)) {
        return {};
    }

    if (std::count_if(lhsPartialShape.cbegin(), lhsPartialShape.cend(), [](const ngraph::Dimension& dimension) { return dimension.is_dynamic(); }) > 1 ||
        std::count_if(rhsPartialShape.cbegin(), rhsPartialShape.cend(), [](const ngraph::Dimension& dimension) { return dimension.is_dynamic(); }) > 1) {
        return {};
    }

    const auto& lhsSliceMode = lhsRankLength < 3 ? SliceMode::Unchanged : SliceMode::Slice;
    const auto& rhsSliceMode = rhsRankLength < 3 ? SliceMode::Unchanged : SliceMode::Slice;

    return {{lhsSliceMode, rhsSliceMode}, {SliceMode::Slice}};
}

}  // namespace vpu
