// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/error.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "vpu/ngraph/transformations/extract_dynamic_batch/slice_convolution.hpp"

namespace vpu {

SliceConfiguration sliceConvolution(const ngraph::Node& node) {
    VPU_THROW_UNLESS(node.get_input_size() == 2,  "Expecting operation {} to have {} inputs, got {}", node, 2, node.get_input_size());
    VPU_THROW_UNLESS(node.get_output_size() == 1, "Expecting operation {} to have {} outputs, got {}", node, 1, node.get_output_size());
    VPU_THROW_UNLESS(ngraph::op::is_constant(node.input_value(1).get_node_shared_ptr()), "Expecting operation {} to have constant kernel, got {}",
        node, node.input_value(1));

    const auto& data = node.input_value(0);
    const auto& dataPartialShape = data.get_partial_shape();
    const auto& dataRank = dataPartialShape.rank();
    VPU_THROW_UNLESS(dataRank.is_static(), "Expecting operation {} to have static rank for input {}, got {}", node, data, dataPartialShape);
    const auto& dataRankLength = dataRank.get_length();
    VPU_THROW_UNLESS(dataRankLength >= 3 && dataRankLength <= 5, "Expecting operation {} to have rank of input {} in [{}, {}], got {}",
        node, data, 3, 5, dataRankLength);

    const auto& batch = dataPartialShape[0];
    if (batch.is_static()) {
        return {};
    }

    if (std::count_if(dataPartialShape.cbegin(), dataPartialShape.cend(), [](const ngraph::Dimension& dimension) { return dimension.is_dynamic(); }) > 1) {
        return {};
    }

    return {{SliceMode::Slice, SliceMode::Unchanged}, {SliceMode::Slice}};
}

}  // namespace vpu
