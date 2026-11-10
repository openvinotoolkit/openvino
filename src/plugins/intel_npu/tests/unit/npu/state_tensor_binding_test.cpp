// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Verifies that NetworkMetadata::bindRelatedDescriptors() rejects a state input
// and state output that share a name but differ in shape or precision.
//
// A state input and its state output are two views of the same variable, so
// their shape and precision must match; the plugin later shares a single Level
// Zero buffer between them, sized from the input. If the output were larger, the
// device would write past that buffer. These tests run on the host with no NPU.

#include <gtest/gtest.h>

#include "intel_npu/common/network_metadata.hpp"
#include "openvino/core/except.hpp"

using intel_npu::IODescriptor;
using intel_npu::NetworkMetadata;

namespace {

IODescriptor makeStateInput(const std::string& name,
                            const ov::PartialShape& shape,
                            const ov::element::Type& precision) {
    IODescriptor desc;
    desc.nameFromCompiler = name;
    desc.precision = precision;
    desc.shapeFromCompiler = shape;
    desc.isStateInput = true;
    return desc;
}

IODescriptor makeStateOutput(const std::string& name,
                             const ov::PartialShape& shape,
                             const ov::element::Type& precision) {
    IODescriptor desc;
    desc.nameFromCompiler = name;
    desc.precision = precision;
    desc.shapeFromCompiler = shape;
    desc.isStateOutput = true;
    return desc;
}

}  // namespace

// A shape mismatch between the paired state descriptors must be rejected.
TEST(NpuStateTensorBindingTest, BindThrowsOnStateShapeMismatch) {
    NetworkMetadata metadata;
    metadata.inputs.push_back(makeStateInput("state", ov::PartialShape{1}, ov::element::f32));
    metadata.outputs.push_back(makeStateOutput("state", ov::PartialShape{1048576}, ov::element::f32));

    EXPECT_THROW(metadata.bindRelatedDescriptors(), ov::Exception);
}

// A precision mismatch between the paired state descriptors must be rejected.
TEST(NpuStateTensorBindingTest, BindThrowsOnStatePrecisionMismatch) {
    NetworkMetadata metadata;
    metadata.inputs.push_back(makeStateInput("state", ov::PartialShape{4}, ov::element::u8));
    metadata.outputs.push_back(makeStateOutput("state", ov::PartialShape{4}, ov::element::f32));

    EXPECT_THROW(metadata.bindRelatedDescriptors(), ov::Exception);
}

// Matching state descriptors are bound to each other without error.
TEST(NpuStateTensorBindingTest, BindAcceptsMatchingStateDescriptors) {
    NetworkMetadata metadata;
    metadata.inputs.push_back(makeStateInput("state", ov::PartialShape{2, 3}, ov::element::f32));
    metadata.outputs.push_back(makeStateOutput("state", ov::PartialShape{2, 3}, ov::element::f32));

    ASSERT_NO_THROW(metadata.bindRelatedDescriptors());

    ASSERT_TRUE(metadata.inputs.at(0).relatedDescriptorIndex.has_value());
    ASSERT_TRUE(metadata.outputs.at(0).relatedDescriptorIndex.has_value());
    EXPECT_EQ(*metadata.inputs.at(0).relatedDescriptorIndex, 0u);
    EXPECT_EQ(*metadata.outputs.at(0).relatedDescriptorIndex, 0u);
}
