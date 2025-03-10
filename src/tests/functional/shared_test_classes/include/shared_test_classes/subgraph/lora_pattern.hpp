// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

class LoraPatternBase : public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<const char*>& obj);

protected:
    void run_test_empty_tensors();
    void run_test_random_tensors();

protected:
    static constexpr auto t4_name = "lora/MatMul.B";
    static constexpr auto t5_name = "lora/MatMul.alpha";
    static constexpr auto t6_name = "lora/MatMul.A";
    static constexpr auto netType = ov::element::f32;
};

class LoraPatternMatmul : public LoraPatternBase, public testing::WithParamInterface<const char*> {
public:
    void SetUp() override;

protected:
    static constexpr size_t K = 563ul; // Weights matrix K dimension
    static constexpr size_t N = 2048ul; // Weights matrix N dimension
};

class LoraPatternConvolution : public LoraPatternBase, public testing::WithParamInterface<const char*> {
public:
    void SetUp() override;

protected:
    static constexpr size_t num_channels = 64ul;
};

} // namespace test
} // namespace ov