// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Verifies that Graph::set_batch_size() rejects metadata holding a descriptor without a batch
// axis.
//
// Batching handled by the plugin overrides the batch axis of a descriptor while allocating the
// Level Zero tensor backing it. A rank zero shape owns no such axis, and the subscript operator of
// ov::Shape performs no bound checking, so enabling batching for one would write through the null
// data pointer of an empty allocation instead of being reported. The compilation path rules this
// out before enabling the feature, but a batch size taken from the metadata of an imported blob is
// applied directly, hence the check inside the setter every path passes through.
//
// The graph is built without Level Zero handles, so these tests run on the host with no NPU.

#include <gtest/gtest.h>

#include "graph.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/network_metadata.hpp"
#include "openvino/core/except.hpp"

using intel_npu::FilteredConfig;
using intel_npu::Graph;
using intel_npu::GraphDescriptor;
using intel_npu::IODescriptor;
using intel_npu::NetworkMetadata;
using intel_npu::OptionsDesc;

namespace {

IODescriptor makeDescriptor(const std::string& name, const ov::PartialShape& shape) {
    IODescriptor descriptor;
    descriptor.nameFromCompiler = name;
    descriptor.precision = ov::element::f32;
    descriptor.shapeFromCompiler = shape;
    return descriptor;
}

}  // namespace

struct NpuGraphBatchSizeTest : public ::testing::Test {
    NpuGraphBatchSizeTest() : config(std::make_shared<OptionsDesc>()) {}

    FilteredConfig config;
};

// An input without a batch axis must be reported rather than written past.
TEST_F(NpuGraphBatchSizeTest, SetBatchSizeThrowsOnInputWithoutBatchAxis) {
    NetworkMetadata metadata;
    metadata.inputs.push_back(makeDescriptor("input", ov::PartialShape{}));
    metadata.outputs.push_back(makeDescriptor("output", ov::PartialShape{1, 3}));

    Graph graph(nullptr, nullptr, GraphDescriptor{}, std::move(metadata), std::nullopt, config);

    EXPECT_THROW(graph.set_batch_size(4), ov::Exception);
    EXPECT_FALSE(graph.get_batch_size().has_value());
}

// The outputs are allocated the same way, so they have to be checked as well.
TEST_F(NpuGraphBatchSizeTest, SetBatchSizeThrowsOnOutputWithoutBatchAxis) {
    NetworkMetadata metadata;
    metadata.inputs.push_back(makeDescriptor("input", ov::PartialShape{1, 3}));
    metadata.outputs.push_back(makeDescriptor("output", ov::PartialShape{}));

    Graph graph(nullptr, nullptr, GraphDescriptor{}, std::move(metadata), std::nullopt, config);

    EXPECT_THROW(graph.set_batch_size(4), ov::Exception);
    EXPECT_FALSE(graph.get_batch_size().has_value());
}

// A dynamic rank offers no guarantee that a batch axis will be there either.
TEST_F(NpuGraphBatchSizeTest, SetBatchSizeThrowsOnDynamicRank) {
    NetworkMetadata metadata;
    metadata.inputs.push_back(makeDescriptor("input", ov::PartialShape::dynamic()));
    metadata.outputs.push_back(makeDescriptor("output", ov::PartialShape{1, 3}));

    Graph graph(nullptr, nullptr, GraphDescriptor{}, std::move(metadata), std::nullopt, config);

    EXPECT_THROW(graph.set_batch_size(4), ov::Exception);
    EXPECT_FALSE(graph.get_batch_size().has_value());
}

// Descriptors owning a batch axis are accepted, the batch size being stored as before.
TEST_F(NpuGraphBatchSizeTest, SetBatchSizeAcceptsDescriptorsWithBatchAxis) {
    NetworkMetadata metadata;
    metadata.inputs.push_back(makeDescriptor("input", ov::PartialShape{1, 3, 224, 224}));
    metadata.outputs.push_back(makeDescriptor("output", ov::PartialShape{1, 1000}));

    Graph graph(nullptr, nullptr, GraphDescriptor{}, std::move(metadata), std::nullopt, config);

    ASSERT_NO_THROW(graph.set_batch_size(4));
    ASSERT_TRUE(graph.get_batch_size().has_value());
    EXPECT_EQ(*graph.get_batch_size(), 4u);
}
