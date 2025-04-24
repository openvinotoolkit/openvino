// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<ov::test::utils::MemoryTransformation,  // Apply Memory transformation
                   std::string,                            // Target device name
                   ov::element::Type,                      // Input element type
                   size_t,                                 // Input size
                   size_t,                                 // Hidden size
                   ov::AnyMap                              // Configuration
                   >
    memoryLSTMCellParams;

class MemoryLSTMCellTest : virtual public ov::test::SubgraphBaseStaticTest,
                           public testing::WithParamInterface<memoryLSTMCellParams> {
private:
    // you have to Unroll TI manually and remove memory until supports it
    // since we switching models we need to generate and save weights biases and inputs in SetUp
    void switch_to_friendly_model();
    void create_pure_tensor_iterator_model();
    void init_memory();
    void apply_low_latency();

    ov::test::utils::MemoryTransformation transformation;
    std::vector<float> input_bias;
    std::vector<float> input_weights;
    std::vector<float> hidden_memory_init;
    std::vector<float> cell_memory_init;
    std::vector<float> weights_vals;
    std::vector<float> reccurrenceWeights_vals;
    std::vector<float> bias_vals;
    std::vector<ov::Shape> input_shapes;

protected:
    void SetUp() override;
    void run() override;
    void compile_model() override;
    void infer() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<memoryLSTMCellParams>& obj);
};

}  // namespace test
}  // namespace ov
