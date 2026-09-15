// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_graph_ext_wrappers.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>

#include "common_test_utils/test_assertions.hpp"

namespace {

constexpr uint32_t kArgumentIndex = 7;

ze_graph_argument_properties_3_t makeArgument(uint32_t dimsCount) {
    ze_graph_argument_properties_3_t arg = {};
    arg.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_3;
    arg.type = ZE_GRAPH_ARGUMENT_TYPE_INPUT;
    arg.devicePrecision = ZE_GRAPH_ARGUMENT_PRECISION_FP32;
    arg.dims_count = dimsCount;
    std::strncpy(arg.name, "input", sizeof(arg.name) - 1);
    std::strncpy(arg.debug_friendly_name, "input_node", sizeof(arg.debug_friendly_name) - 1);

    const uint32_t initializedDims = std::min<uint32_t>(dimsCount, ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE);
    for (uint32_t dim = 0; dim < initializedDims; ++dim) {
        arg.dims[dim] = dim == 0 ? 4 : 32;
    }

    return arg;
}

ze_graph_argument_metadata_t makeMetadata(uint32_t shapeSize) {
    ze_graph_argument_metadata_t metadata = {};
    metadata.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_METADATA;
    metadata.type = ZE_GRAPH_ARGUMENT_TYPE_INPUT;
    metadata.shape_size = shapeSize;

    const uint32_t initializedDims = std::min<uint32_t>(shapeSize, ZE_MAX_GRAPH_TENSOR_REF_DIMS);
    for (uint32_t dim = 0; dim < initializedDims; ++dim) {
        metadata.shape[dim] = dim == 0 ? 4 : 32;
    }

    return metadata;
}

}  // namespace

TEST(ZeGraphExtWrappersTest, RejectsMetadataRankGreaterThanArgumentRank) {
    auto arg = makeArgument(1);
    auto metadata = makeMetadata(2);
    metadata.shape[1] = std::numeric_limits<uint64_t>::max();

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, metadata),
                                  ov::Exception,
                                  "metadata shape_size 2 does not match dims_count 1");
}

TEST(ZeGraphExtWrappersTest, RejectsArgumentRankAboveAbiLimit) {
    auto arg = makeArgument(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE + 1);

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, std::nullopt),
                                  ov::Exception,
                                  "dims_count 6 exceeds ABI limit 5");
}

TEST(ZeGraphExtWrappersTest, RejectsMetadataRankAboveAbiLimit) {
    auto arg = makeArgument(1);
    auto metadata = makeMetadata(ZE_MAX_GRAPH_TENSOR_REF_DIMS + 1);

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, metadata),
                                  ov::Exception,
                                  "metadata shape_size 9 exceeds ABI limit 8");
}

TEST(ZeGraphExtWrappersTest, AcceptsMatchingDynamicMetadataRank) {
    auto arg = makeArgument(2);
    auto metadata = makeMetadata(2);
    metadata.shape[1] = std::numeric_limits<uint64_t>::max();

    const auto descriptor = intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, metadata);

    EXPECT_EQ(descriptor.indexUsedByDriver, kArgumentIndex);
    ASSERT_TRUE(descriptor.shapeFromIRModel.has_value());
    EXPECT_EQ(descriptor.shapeFromCompiler, ov::PartialShape({4, 32}));
    EXPECT_EQ(*descriptor.shapeFromIRModel, ov::PartialShape({4, ov::Dimension(1, 32)}));
}