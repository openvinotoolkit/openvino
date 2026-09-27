// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_graph_ext_wrappers.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <optional>
#include <string>

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
    constexpr uint32_t argRank = 1;
    constexpr uint32_t metadataRank = 2;
    auto arg = makeArgument(argRank);
    auto metadata = makeMetadata(metadataRank);
    metadata.shape[1] = std::numeric_limits<uint64_t>::max();

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, metadata),
                                  ov::Exception,
                                  "metadata shape_size " + std::to_string(metadataRank) +
                                      " does not match dims_count " + std::to_string(argRank));
}

TEST(ZeGraphExtWrappersTest, RejectsArgumentRankAboveAbiLimit) {
    const uint32_t aboveAbiLimit = ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE + 1;
    auto arg = makeArgument(aboveAbiLimit);

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, std::nullopt),
                                  ov::Exception,
                                  "dims_count " + std::to_string(aboveAbiLimit) +
                                      " exceeds ABI limit " + std::to_string(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE));
}

TEST(ZeGraphExtWrappersTest, RejectsMetadataRankAboveAbiLimit) {
    const uint32_t aboveAbiLimit = ZE_MAX_GRAPH_TENSOR_REF_DIMS + 1;
    auto arg = makeArgument(1);
    auto metadata = makeMetadata(aboveAbiLimit);

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, metadata),
                                  ov::Exception,
                                  "metadata shape_size " + std::to_string(aboveAbiLimit) +
                                      " exceeds ABI limit " + std::to_string(ZE_MAX_GRAPH_TENSOR_REF_DIMS));
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
