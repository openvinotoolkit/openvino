// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_graph.hpp"

#include <gtest/gtest.h>

#include <iterator>
#include <limits>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/except.hpp"

namespace {

ze_graph_argument_properties_3_t makeArgument() {
    ze_graph_argument_properties_3_t arg = {};
    arg.type = ZE_GRAPH_ARGUMENT_TYPE_INPUT;
    arg.devicePrecision = ZE_GRAPH_ARGUMENT_PRECISION_FP32;
    arg.dims_count = 1;
    arg.dims[0] = 4;
    return arg;
}

ze_graph_argument_metadata_t makeMetadata() {
    ze_graph_argument_metadata_t metadata = {};
    metadata.shape_size = 1;
    metadata.shape[0] = 4;
    return metadata;
}

TEST(DynamicGraphTests, RejectsMetadataRankGreaterThanCompilerRank) {
    auto arg = makeArgument();
    auto metadata = makeMetadata();
    metadata.shape_size = 2;
    metadata.shape[1] = std::numeric_limits<uint64_t>::max();

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::getIODescriptor(arg, metadata),
                                  ov::Exception,
                                  "Inconsistent tensor ranks in dynamic graph metadata");
}

TEST(DynamicGraphTests, RejectsAssociatedTensorNameCountGreaterThanCapacity) {
    auto arg = makeArgument();
    arg.associated_tensor_names_count = static_cast<uint32_t>(std::size(arg.associated_tensor_names) + 1);

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::getIODescriptor(arg, makeMetadata()),
                                  ov::Exception,
                                  "Associated tensor name count exceeds dynamic graph metadata capacity");
}

TEST(DynamicGraphTests, RejectsCompilerRankGreaterThanCapacity) {
    auto arg = makeArgument();
    arg.dims_count = static_cast<uint32_t>(std::size(arg.dims) + 1);

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::getIODescriptor(arg, makeMetadata()),
                                  ov::Exception,
                                  "Compiler shape rank exceeds dynamic graph metadata capacity");
}

TEST(DynamicGraphTests, RejectsMetadataRankGreaterThanCapacity) {
    auto metadata = makeMetadata();
    metadata.shape_size = static_cast<uint32_t>(std::size(metadata.shape) + 1);

    OV_EXPECT_THROW_HAS_SUBSTRING(intel_npu::getIODescriptor(makeArgument(), metadata),
                                  ov::Exception,
                                  "IR shape rank exceeds dynamic graph metadata capacity");
}

TEST(DynamicGraphTests, AcceptsMatchingCompilerAndMetadataRanks) {
    const auto descriptor = intel_npu::getIODescriptor(makeArgument(), makeMetadata());

    ASSERT_TRUE(descriptor.shapeFromIRModel.has_value());
    EXPECT_EQ(descriptor.shapeFromCompiler, ov::PartialShape({4}));
    EXPECT_EQ(descriptor.shapeFromIRModel.value(), ov::PartialShape({4}));
}

}  // namespace
