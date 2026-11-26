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
#include <unordered_set>

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
                                      " does not match the rank " + std::to_string(argRank));
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

TEST(ZeGraphExtWrappersTest, DecodesUnboundedDimensionAsFullyDynamic) {
    auto arg = makeArgument(2);
    arg.dims[1] = 0;
    auto metadata = makeMetadata(2);
    metadata.shape[1] = std::numeric_limits<uint64_t>::max();

    const auto descriptor = intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, metadata);

    ASSERT_TRUE(descriptor.shapeFromIRModel.has_value());
    EXPECT_EQ(descriptor.shapeFromCompiler, ov::PartialShape({4, 0}));
    EXPECT_EQ(*descriptor.shapeFromIRModel, ov::PartialShape({4, ov::Dimension::dynamic()}));
    EXPECT_TRUE(descriptor.shapeFromIRModel->compatible(ov::PartialShape({4, 4096})));
}

TEST(ZeGraphExtWrappersTest, KeepsStaticZeroDimension) {
    auto arg = makeArgument(2);
    arg.dims[1] = 0;
    auto metadata = makeMetadata(2);
    metadata.shape[1] = 0;

    const auto descriptor = intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, metadata);

    ASSERT_TRUE(descriptor.shapeFromIRModel.has_value());
    EXPECT_EQ(*descriptor.shapeFromIRModel, ov::PartialShape({4, 0}));
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
                                  "metadata shape_size 1 does not match the rank 0");
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

namespace {

constexpr uint32_t kMaxRank = ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE_8;

// The dimensions past the rank hold `padding`, which is zero in the header but one in drivers
ze_graph_argument_properties_4_t makeArgument4(uint32_t rank, uint32_t padding = 1) {
    ze_graph_argument_properties_4_t arg = {};
    arg.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES_4;
    arg.type = ZE_GRAPH_ARGUMENT_TYPE_INPUT;
    arg.precision = ZE_GRAPH_ARGUMENT_PRECISION_FP32;
    std::fill(std::begin(arg.dims), std::end(arg.dims), padding);
    for (uint32_t dim = 0; dim < rank; ++dim) {
        arg.dims[dim] = dim + 1;
    }
    return arg;
}

// Static IR shape equal to the leading dimensions of `arg`
ze_graph_argument_metadata_t makeMetadata4(const ze_graph_argument_properties_4_t& arg, uint32_t rank) {
    auto metadata = makeMetadata(rank);
    for (uint32_t dim = 0; dim < std::min(rank, kMaxRank); ++dim) {
        metadata.shape[dim] = arg.dims[dim];
    }
    return metadata;
}

intel_npu::IODescriptor createDescriptor4(const ze_graph_argument_properties_4_t& arg,
                                          const ze_graph_argument_metadata_t& metadata) {
    return intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg, "input", "input_node", {}, metadata);
}

// The driver answers "properties_3" with the first dimensions of a higher rank argument, without any error
ze_graph_argument_properties_3_t makeCutArgument3(const ze_graph_argument_properties_4_t& arg4) {
    auto arg = makeArgument(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE);
    std::copy_n(std::begin(arg4.dims), ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE, std::begin(arg.dims));
    return arg;
}

}  // namespace

TEST(ZeGraphExtWrappersTest, Properties4ReportsFullAbiRank) {
    const auto arg = makeArgument4(kMaxRank);

    const auto descriptor = createDescriptor4(arg, makeMetadata4(arg, kMaxRank));

    EXPECT_EQ(descriptor.shapeFromCompiler, ov::PartialShape({1, 2, 3, 4, 5, 6, 7, 8}));
    EXPECT_EQ(descriptor.indexUsedByDriver, kArgumentIndex);
}

TEST(ZeGraphExtWrappersTest, Properties4ReadsPrecisionAndNames) {
    auto arg = makeArgument4(2);
    arg.precision = ZE_GRAPH_ARGUMENT_PRECISION_INT8;

    const auto descriptor = intel_npu::createIODescriptorFromLevelZero(kArgumentIndex,
                                                                       arg,
                                                                       "input",
                                                                       "input_node",
                                                                       {"a", "b"},
                                                                       makeMetadata4(arg, 2));

    EXPECT_EQ(descriptor.precision, ov::element::i8);
    EXPECT_EQ(descriptor.nameFromCompiler, "input");
    EXPECT_EQ(descriptor.nodeFriendlyName, "input_node");
    EXPECT_EQ(descriptor.outputTensorNames, (std::unordered_set<std::string>{"a", "b"}));
}

TEST(ZeGraphExtWrappersTest, Properties4TakesRankFromMetadataWhateverTheDriverPadsWith) {
    for (const uint32_t padding : {0u, 1u}) {
        const auto arg = makeArgument4(2, padding);

        const auto descriptor = createDescriptor4(arg, makeMetadata4(arg, 2));

        EXPECT_EQ(descriptor.shapeFromCompiler, ov::PartialShape({1, 2})) << "padding " << padding;
        ASSERT_TRUE(descriptor.shapeFromIRModel.has_value());
        EXPECT_EQ(*descriptor.shapeFromIRModel, ov::PartialShape({1, 2})) << "padding " << padding;
    }
}

TEST(ZeGraphExtWrappersTest, Properties4KeepsTrailingOnesInsideMetadataRank) {
    // dims are all ones: the first one is a dimension, the rest could be padding
    const auto arg = makeArgument4(1);

    EXPECT_EQ(createDescriptor4(arg, makeMetadata4(arg, 3)).shapeFromCompiler, ov::PartialShape({1, 1, 1}));
    EXPECT_EQ(createDescriptor4(arg, makeMetadata4(arg, 0)).shapeFromCompiler, ov::PartialShape{});
}

TEST(ZeGraphExtWrappersTest, Properties4RejectsMetadataRankBelowDimensionsReported) {
    const auto arg = makeArgument4(3);
    auto metadata = makeMetadata4(arg, 2);
    metadata.shape[1] = std::numeric_limits<uint64_t>::max();

    OV_EXPECT_THROW_HAS_SUBSTRING(createDescriptor4(arg, metadata),
                                  ov::Exception,
                                  "metadata shape_size 2 is below the rank 3 implied by the dimensions");
}

TEST(ZeGraphExtWrappersTest, Properties4RejectsMetadataRankAboveAbiLimit) {
    const auto arg = makeArgument4(kMaxRank);
    const auto metadata = makeMetadata(kMaxRank + 1);

    OV_EXPECT_THROW_HAS_SUBSTRING(createDescriptor4(arg, metadata),
                                  ov::Exception,
                                  "metadata shape_size " + std::to_string(kMaxRank + 1) + " exceeds ABI limit " +
                                      std::to_string(ZE_MAX_GRAPH_TENSOR_REF_DIMS));
}

TEST(ZeGraphExtWrappersTest, Properties4AcceptsMetadataWithFullAbiRank) {
    const auto arg = makeArgument4(kMaxRank);
    auto metadata = makeMetadata4(arg, kMaxRank);
    metadata.shape[kMaxRank - 1] = std::numeric_limits<uint64_t>::max();

    const auto descriptor = createDescriptor4(arg, metadata);

    ASSERT_TRUE(descriptor.shapeFromIRModel.has_value());
    EXPECT_EQ(*descriptor.shapeFromIRModel, ov::PartialShape({1, 2, 3, 4, 5, 6, 7, ov::Dimension(1, 8)}));
}

TEST(ZeGraphExtWrappersTest, Properties4ReportsDynamicStridesSupportOnlyUpToTheStridesAbiRank) {
    ze_graph_argument_property_strides_t supportsStrides = {};
    supportsStrides.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTY_STRIDES;

    // "userStrides" has room for ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE entries, whatever the driver reports
    for (const uint32_t rank : std::initializer_list<uint32_t>{ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE, kMaxRank}) {
        auto arg = makeArgument4(rank);
        arg.pNext = &supportsStrides;
        const auto metadata = makeMetadata4(arg, rank);
        const bool fitsUserStrides = rank <= ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE;

        supportsStrides.supportsDynamicStrides = true;
        EXPECT_EQ(createDescriptor4(arg, metadata).supportsStridedLayout, fitsUserStrides) << "rank " << rank;

        supportsStrides.supportsDynamicStrides = false;
        EXPECT_FALSE(createDescriptor4(arg, metadata).supportsStridedLayout) << "rank " << rank;
    }
}

TEST(ZeGraphExtWrappersTest, Properties3WithProperties4TakesTheRankAboveFiveFromProperties4) {
    for (const uint32_t padding : {0u, 1u}) {
        for (uint32_t rank = ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE + 1; rank <= kMaxRank; ++rank) {
            const auto arg4 = makeArgument4(rank, padding);

            const auto descriptor =
                intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, makeCutArgument3(arg4), arg4);

            EXPECT_EQ(descriptor.shapeFromCompiler.rank().get_length(), static_cast<int64_t>(rank)) << "padding " << padding;
            EXPECT_EQ(descriptor.shapeFromCompiler[rank - 1], rank) << "padding " << padding;
            EXPECT_FALSE(descriptor.shapeFromIRModel.has_value());
        }
    }
}

TEST(ZeGraphExtWrappersTest, Properties3WithProperties4KeepsFiveDimensionsWhenProperties4HoldsNoMore) {
    for (const uint32_t padding : {0u, 1u}) {
        const auto arg4 = makeArgument4(ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE, padding);

        const auto descriptor =
            intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, makeCutArgument3(arg4), arg4);

        EXPECT_EQ(descriptor.shapeFromCompiler, ov::PartialShape({1, 2, 3, 4, 5})) << "padding " << padding;
    }
}

TEST(ZeGraphExtWrappersTest, Properties3WithProperties4ReportsDynamicStridesSupportOnlyUpToTheStridesAbiRank) {
    ze_graph_argument_property_strides_t supportsStrides = {};
    supportsStrides.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTY_STRIDES;
    supportsStrides.supportsDynamicStrides = true;

    for (const uint32_t rank : std::initializer_list<uint32_t>{ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE, kMaxRank}) {
        const auto arg4 = makeArgument4(rank);
        auto arg3 = makeCutArgument3(arg4);
        arg3.pNext = &supportsStrides;

        const auto descriptor = intel_npu::createIODescriptorFromLevelZero(kArgumentIndex, arg3, arg4);

        EXPECT_EQ(descriptor.supportsStridedLayout, rank <= ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE) << "rank " << rank;
    }
}

TEST(ZeGraphExtWrappersTest, ParseTensorNamesSplitsOnCommas) {
    EXPECT_EQ(intel_npu::parseTensorNames("a,b,c"), (std::unordered_set<std::string>{"a", "b", "c"}));
    EXPECT_EQ(intel_npu::parseTensorNames("only"), (std::unordered_set<std::string>{"only"}));
}

TEST(ZeGraphExtWrappersTest, ParseTensorNamesDropsTheTerminatorOfTheDocumentedList) {
    EXPECT_EQ(intel_npu::parseTensorNames(std::string("a,b\0", 4)), (std::unordered_set<std::string>{"a", "b"}));
    EXPECT_EQ(intel_npu::parseTensorNames(std::string("only\0", 5)), (std::unordered_set<std::string>{"only"}));
}

TEST(ZeGraphExtWrappersTest, ParseTensorNamesSplitsNulPaddedSlots) {
    // Layout seen from drivers: 32 slots of 256 bytes, the unused ones zeroed
    std::string slots(32 * 256, '\0');
    slots.replace(0, 4, "data");
    slots.replace(256, 3, "in1");

    EXPECT_EQ(intel_npu::parseTensorNames(slots), (std::unordered_set<std::string>{"data", "in1"}));
    EXPECT_EQ(intel_npu::parseTensorNames(std::string("a\0b\0", 4)), (std::unordered_set<std::string>{"a", "b"}));
}

TEST(ZeGraphExtWrappersTest, ParseTensorNamesKeepsCommasInsideNulPaddedSlots) {
    std::string slots(32 * 256, '\0');
    slots.replace(0, 3, "a,b");
    slots.replace(256, 3, "in1");
    EXPECT_EQ(intel_npu::parseTensorNames(slots), (std::unordered_set<std::string>{"a,b", "in1"}));

    std::string singleSlot(256, '\0');
    singleSlot.replace(0, 3, "a,b");
    EXPECT_EQ(intel_npu::parseTensorNames(singleSlot), (std::unordered_set<std::string>{"a,b"}));
}

TEST(ZeGraphExtWrappersTest, ParseTensorNamesDropsEmptyEntries) {
    EXPECT_TRUE(intel_npu::parseTensorNames("").empty());
    EXPECT_TRUE(intel_npu::parseTensorNames(",,").empty());
    EXPECT_EQ(intel_npu::parseTensorNames(",a,,b,"), (std::unordered_set<std::string>{"a", "b"}));
}
