// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/model/data_desc.hpp"
#include "gtest/gtest.h"

using namespace vpu;

TEST(DataDesc, CanCreateScalarFromTypeOrderIterators) {
    auto dims = std::vector<int>{};
    auto descriptor = DataDesc{DataType::FP16, DimsOrder::C, dims.begin(), dims.end()};
    ASSERT_EQ(descriptor, (DataDesc{DataType::FP16, DimsOrder::C, {1}}));
}

TEST(DataDesc, CanCreateScalarFromTypeOrderVector) {
    auto descriptor = DataDesc{DataType::FP16, DimsOrder::C,std::vector<int>{}};
    ASSERT_EQ(descriptor, (DataDesc{DataType::FP16, DimsOrder::C, {1}}));
}

TEST(DataDesc, CanCreateScalarFromTypeOrderInitializerList) {
    auto descriptor = DataDesc{DataType::FP16, DimsOrder::C, std::initializer_list<int>{}};
    ASSERT_EQ(descriptor, (DataDesc{DataType::FP16, DimsOrder::C, {1}}));
}

TEST(DataDesc, CanCreateScalarFromOrderInitializerList) {
    auto descriptor = DataDesc{DimsOrder::C, std::initializer_list<int>{}};
    ASSERT_EQ(descriptor, (DataDesc{DataType::FP16, DimsOrder::C, {1}}));
}

TEST(DataDesc, CanCreateScalarFromInitializerList) {
    auto descriptor = DataDesc{std::initializer_list<int>{}};
    ASSERT_EQ(descriptor, (DataDesc{DataType::FP16, DimsOrder::C, {1}}));
}

TEST(DataDesc, CanCreateScalarFromVector) {
    auto descriptor = DataDesc{ std::vector<int>{}};
    ASSERT_EQ(descriptor, (DataDesc{DataType::FP16, DimsOrder::C, {1}}));
}

TEST(DataDesc, CanCreateScalarFromTensorDesc) {
    auto tensorDesc = InferenceEngine::TensorDesc{InferenceEngine::Precision::FP16, {}, InferenceEngine::Layout::SCALAR};
    auto descriptor = DataDesc{ tensorDesc};
    ASSERT_EQ(descriptor, (DataDesc{DataType::FP16, DimsOrder::C, {1}}));
}

TEST(DataDesc, CanCreateScalarFromTypeOrderDims) {
    auto descriptor = DataDesc{DataType::FP16, DimsOrder::C, DimValues{}};
    ASSERT_EQ(descriptor, (DataDesc{DataType::FP16, DimsOrder::C, {1}}));
}

TEST(DimsOrder, CanCreateScalarFromNumDims) {
    ASSERT_EQ(DimsOrder::fromNumDims(0), DimsOrder::fromNumDims(1));
}

TEST(DimsOrder, CanCreateScalarFromLayout) {
    ASSERT_EQ(DimsOrder::fromLayout(InferenceEngine::Layout::SCALAR), DimsOrder::fromLayout(InferenceEngine::Layout::C));
}
