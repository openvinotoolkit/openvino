// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

typedef std::tuple<ov::test::utils::MemoryTransformation,  // Apply Memory transformation
                   std::string,                            // Target device name
                   ov::element::Type,                      // Input precision
                   size_t,                                 // Input size
                   size_t,                                 // Hidden size
                   ov::AnyMap                              // Configuration
                   >
    multipleLSTMCellParams;

class MultipleLSTMCellTest : virtual public ov::test::SubgraphBaseStaticTest,
                             public testing::WithParamInterface<multipleLSTMCellParams> {
private:
    // you have to Unroll TI manually and remove memory until supports it
    // since we switching models we need to generate and save weights biases and inputs in SetUp
    void switch_to_friendly_model();
    void create_pure_tensor_iterator_model();
    void init_memory();
    void apply_low_latency();

    size_t hiddenSize;
    std::vector<float> input_bias;
    std::vector<float> input_weights;
    std::vector<float> hidden_memory_init;
    std::vector<float> cell_memory_init;
    std::vector<float> weights_vals;
    std::vector<float> weights_2_vals;
    std::vector<float> reccurrenceWeights_vals;
    std::vector<float> bias_vals;
    ov::test::utils::MemoryTransformation transformation;
    std::vector<ov::Shape> input_shapes;

protected:
    void SetUp() override;
    void compile_model() override;
    void run() override;
    void infer() override;

public:
    static std::string getTestCaseName(const testing::TestParamInfo<multipleLSTMCellParams>& obj);
};

}  // namespace test
}  // namespace ov
