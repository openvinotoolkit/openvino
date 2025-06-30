// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

class LoraPatternBase : public SubgraphBaseTest {
protected:
    void run_test_empty_tensors();
    void run_test_random_tensors(ov::element::Type net_type, size_t lora_rank);

    static constexpr auto t4_name = "lora/MatMul.B";
    static constexpr auto t5_name = "lora/MatMul.alpha";
    static constexpr auto t6_name = "lora/MatMul.A";
};

using LoraMatMulParams = std::tuple<std::string,         // Device name
                                    ov::element::Type,   // Network type
                                    size_t,              // Input matrix M dimension
                                    size_t,              // Weights matrix N dimension
                                    size_t,              // Weights matrix K dimension
                                    size_t>;             // LoRA rank

class LoraPatternMatmul : public LoraPatternBase, public testing::WithParamInterface<LoraMatMulParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LoraMatMulParams> obj);
    void SetUp() override;
};

using LoraConvolutionParams = std::tuple<std::string,         // Device name
                                         ov::element::Type,   // Network type
                                         size_t,              // Number of channels
                                         size_t>;             // LoRA rank

class LoraPatternConvolution : public LoraPatternBase, public testing::WithParamInterface<LoraConvolutionParams> {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LoraConvolutionParams> obj);
    void SetUp() override;
};

} // namespace test
} // namespace ov
