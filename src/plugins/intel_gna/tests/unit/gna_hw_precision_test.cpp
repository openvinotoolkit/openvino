// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "any_copy.hpp"
#include "gna_data_types.hpp"
#include "gna_plugin.hpp"
#include "memory/gna_memory.hpp"
#include "ov_models/builders.hpp"

using namespace InferenceEngine;
namespace testing {

class GNAPluginForPrecisionTest : public GNAPlugin {
public:
    GNAPluginForPrecisionTest(const std::map<std::string, std::string>& configMap) : GNAPlugin(configMap) {
        gnamem.reset(new gna_memory_float(memory::GNAFloatAllocator{}));
        m_graph_compiler->setGNAMemoryPtr(gnamem);
        gnadevice.reset();
    }
    std::vector<intel_dnn_component_t> get_components() {
        return this->dnn->component;
    }
    void set_low_precision_input() {
        this->gnaFlags->input_low_precision = true;
        this->config.gnaFlags.input_low_precision = true;
    }
};

class GNAHwPrecisionTest : public ::testing::Test {
public:
    void Run(const ov::AnyMap gna_config, bool low_precision = false) {
        auto plugin = GNAPluginForPrecisionTest(any_copy(gna_config));
        auto function = getFunction();
        CNNNetwork cnnNetwork(function);
        if (low_precision) {
            plugin.set_low_precision_input();
        }
        plugin.LoadNetwork(cnnNetwork);
        auto components = plugin.get_components();
        for (int k = 0; k < components.size(); ++k) {
            if (components[k].operation == kDnnAffineOp || components[k].operation == kDnnDiagonalOp) {
                i_precision_size = components[k].num_bytes_per_input;
                o_precision_size = components[k].num_bytes_per_output;
                weights_sizes.push_back(components[k].op.affine.num_bytes_per_weight);
                bias_sizes.push_back(components[k].op.affine.num_bytes_per_bias);
            }
        }
    }

protected:
    std::shared_ptr<ov::Model> getFunction() {
        auto firstInput = std::make_shared<ngraph::opset8::Parameter>(net_precision, shape);
        auto secondInput = std::make_shared<ngraph::opset8::Constant>(net_precision, shape);
        auto matmul = std::make_shared<ngraph::opset8::MatMul>(firstInput, secondInput, false, true);
        auto result = std::make_shared<ngraph::opset8::Result>(matmul);
        auto function =
            std::make_shared<ov::Model>(ov::ResultVector({result}), ov::ParameterVector({firstInput}), "MatMul");
        return function;
    }
    void compare(ngraph::element::Type i_precision, ngraph::element::Type o_precision, size_t w_size, size_t b_size) {
        EXPECT_EQ(i_precision_size, i_precision.size());
        EXPECT_EQ(o_precision_size, o_precision.size());
        for (size_t i = 0; i < weights_sizes.size(); ++i) {
            EXPECT_EQ(w_size, weights_sizes[i]);
            EXPECT_EQ(b_size, bias_sizes[i]);
        }
    }
    const ngraph::element::Type net_precision = ngraph::element::f32;
    const ngraph::Shape shape = {1, 10};
    uint32_t i_precision_size;
    uint32_t o_precision_size;
    std::vector<int> weights_sizes;
    std::vector<int> bias_sizes;
};

TEST_F(GNAHwPrecisionTest, GNAHwPrecisionTestDefault) {
    Run({
        ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
    });
    compare(ngraph::element::i16, ngraph::element::i32, sizeof(int16_t), sizeof(uint32_t));
}

TEST_F(GNAHwPrecisionTest, GNAHwPrecisionTestI16) {
    Run({ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
         ov::hint::inference_precision(ngraph::element::i16)});
    compare(ngraph::element::i16, ngraph::element::i32, sizeof(int16_t), sizeof(uint32_t));
}

TEST_F(GNAHwPrecisionTest, GNAHwPrecisionTestI8) {
    Run({ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
         ov::hint::inference_precision(ngraph::element::i8)});
    compare(ngraph::element::i16,
            ngraph::element::i32,
            sizeof(int8_t),
            Precision::fromType<gna_compound_bias_t>().size());
}

TEST_F(GNAHwPrecisionTest, GNAHwPrecisionTestI8LP) {
    Run({ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_EXACT),
         ov::hint::inference_precision(ngraph::element::i8)},
        true);
    compare(ngraph::element::i8, ngraph::element::i32, sizeof(int8_t), sizeof(int8_t));
}

TEST_F(GNAHwPrecisionTest, GNAHwPrecisionTestFP32) {
    Run({
        ov::intel_gna::execution_mode(ov::intel_gna::ExecutionMode::SW_FP32),
    });
    compare(ngraph::element::f32, ngraph::element::f32, sizeof(float), sizeof(float));
}
}  // namespace testing
