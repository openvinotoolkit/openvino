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

TEST(ZeGraphExtWrappersTest, AcceptsPluginBatchingMetadata) {
    auto arg = makeArgument(2);
    arg.dims[0] = 1;
    auto metadata = makeMetadata(2);
    metadata.shape[0] = 4;

    const auto descriptor = intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, metadata);

    EXPECT_EQ(descriptor.shapeFromCompiler, ov::PartialShape({1, 32}));
    ASSERT_TRUE(descriptor.shapeFromIRModel.has_value());
    EXPECT_EQ(*descriptor.shapeFromIRModel, ov::PartialShape({4, 32}));
}

TEST(ZeGraphExtWrappersTest, RejectsUndersizedPluginBatchingMetadata) {
    auto arg = makeArgument(2);
    arg.dims[0] = 1;
    auto metadata = makeMetadata(2);
    metadata.shape[0] = 0;

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, metadata),
                                  ov::Exception,
                                  "static metadata dimension 0 value 0 does not match driver argument dimension 1");
}

TEST(ZeGraphExtWrappersTest, RejectsStaticMetadataDriverSpanMismatch) {
    // Same rank on both sides, but the static metadata shape ([1]) undersizes the real
    // driver argument span ([1024]); a naive tensor sized from metadata alone would let the
    // driver write/read out of bounds relative to the (smaller) OpenVINO-visible allocation.
    auto arg = makeArgument(1);
    arg.dims[0] = 1024;
    auto metadata = makeMetadata(1);
    metadata.shape[0] = 1;

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, metadata),
                                  ov::Exception,
                                  "static metadata dimension 0 value 1 does not match driver argument dimension 1024");
}

TEST(ZeGraphExtWrappersTest, RejectsCompilerAndIrRankMismatchZeroRank) {
    // Compiler shape rank 0 (empty shapeFromCompiler) paired with a one-axis dynamic IR shape:
    // without the rank-equality guard, indexing shapeFromCompiler[0] to resolve the dynamic
    // upper bound would read out of bounds of the empty vector.
    auto arg = makeArgument(0);
    auto metadata = makeMetadata(1);
    metadata.shape[0] = std::numeric_limits<uint64_t>::max();

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, metadata),
                                  ov::Exception,
                                  "metadata shape_size 1 does not match dims_count 0");
}

TEST(ZeGraphExtWrappersTest, RejectsUnterminatedName) {
    // A compromised driver/parser can fill arg.name with exactly ZE_MAX_GRAPH_ARGUMENT_NAME
    // non-NUL bytes; constructing a std::string from it would scan past the buffer.
    auto arg = makeArgument(0);
    std::memset(arg.name, 'A', sizeof(arg.name));

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, std::nullopt),
                                  ov::Exception,
                                  "name is not NUL-terminated");
}

TEST(ZeGraphExtWrappersTest, RejectsUnterminatedDebugFriendlyName) {
    auto arg = makeArgument(0);
    std::memset(arg.debug_friendly_name, 'A', sizeof(arg.debug_friendly_name));

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, std::nullopt),
                                  ov::Exception,
                                  "debug_friendly_name is not NUL-terminated");
}

TEST(ZeGraphExtWrappersTest, RejectsUnterminatedAssociatedTensorName) {
    auto arg = makeArgument(0);
    arg.associated_tensor_names_count = 1;
    std::memset(arg.associated_tensor_names[0], 'A', sizeof(arg.associated_tensor_names[0]));

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, std::nullopt),
                                  ov::Exception,
                                  "associated_tensor_names is not NUL-terminated");
}

TEST(ZeGraphExtWrappersTest, RejectsAssociatedTensorNamesCountAboveAbiLimit) {
    auto arg = makeArgument(0);
    const auto aboveAbiLimit = ZE_MAX_GRAPH_TENSOR_NAMES_SIZE + 1;
    arg.associated_tensor_names_count = aboveAbiLimit;

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, std::nullopt),
                                  ov::Exception,
                                  "associated_tensor_names_count " + std::to_string(aboveAbiLimit) +
                                      " exceeds ABI limit " + std::to_string(ZE_MAX_GRAPH_TENSOR_NAMES_SIZE));
}

TEST(ZeGraphExtWrappersTest, AcceptsFullCapacityNameWithTrailingNul) {
    // Negative control from the report: 255 non-NUL bytes followed by the parser-initialized
    // final NUL byte is a legitimate full-length terminated name and must still be accepted.
    auto arg = makeArgument(0);
    std::memset(arg.name, 'A', sizeof(arg.name) - 1);
    arg.name[sizeof(arg.name) - 1] = '\0';

    const auto descriptor = intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, std::nullopt);
    EXPECT_EQ(descriptor.nameFromCompiler.size(), sizeof(arg.name) - 1);
}
