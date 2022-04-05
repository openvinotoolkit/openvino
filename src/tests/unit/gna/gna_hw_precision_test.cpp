// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "ngraph_functions/builders.hpp"
#include "gna_plugin.hpp"
#include "memory/gna_memory.hpp"
#include "gna_data_types.hpp"

using namespace InferenceEngine;
namespace testing {

class GNAPluginForPrecisionTest : public GNAPluginNS::GNAPlugin {
public:
    std::shared_ptr<GNAPluginNS::gna_memory_type> gnamem_t;
    GNAPluginForPrecisionTest(const std::map<std::string, std::string>& configMap) :
                            GNAPluginNS::GNAPlugin(configMap) {
        gnamem_t = std::make_shared<GNAPluginNS::gna_memory_type>(GNAPluginNS::memory::make_polymorph<std::allocator<uint8_t>>());
        gnamem = gnamem_t;
        graphCompiler.setGNAMemoryPtr(gnamem);
        gnadevice.reset();
    }
    std::vector<intel_dnn_component_t> getComponents() {
        return this->dnn->component;
    }
};

class GNAHwPrecisionTest: public ::testing::Test {
public:
    void Run() {
        auto plugin = GNAPluginForPrecisionTest(gna_config);
        auto function = getFunction();
        CNNNetwork cnnNetwork(function);
        plugin.LoadNetwork(cnnNetwork);

        auto components = plugin.getComponents();
        for (int k = 0; k < components.size(); ++k) {
            if (components[k].operation == kDnnAffineOp || components[k].operation == kDnnDiagonalOp) {
                weights_sizes.push_back(components[k].op.affine.num_bytes_per_weight);
                bias_sizes.push_back(components[k].op.affine.num_bytes_per_bias);
            }
        }
    }

protected:
    std::shared_ptr<ov::Model> getFunction() {
        auto firstInput = std::make_shared<ngraph::opset8::Parameter>(netPrecision, shape);
        auto secondInput = std::make_shared<ngraph::opset8::Constant>(netPrecision, shape);
        auto matmul = std::make_shared<ngraph::opset8::MatMul>(firstInput, secondInput, false, true);
        auto result = std::make_shared<ngraph::opset8::Result>(matmul);
        auto function = std::make_shared<ov::Model>(ov::ResultVector({result}), ov::ParameterVector({firstInput}), "MatMul");
        return function;
    }
    ngraph::element::Type netPrecision = ngraph::element::f32;
    ngraph::Shape shape = {1, 10};
    std::map<std::string, std::string> gna_config;
    std::vector<int> weights_sizes;
    std::vector<int> bias_sizes;
};

TEST_F(GNAHwPrecisionTest, GNAHwPrecisionTestI16) {
    gna_config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_PRECISION", "I16"}
    };
    Run();
    for (int i = 0; i < weights_sizes.size(); ++i) {
        EXPECT_EQ(sizeof(int16_t), weights_sizes[i]);
        EXPECT_EQ(sizeof(uint32_t), bias_sizes[i]);
    }
}

TEST_F(GNAHwPrecisionTest, GNAHwPrecisionTestI8) {
    gna_config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_PRECISION", "I8"}
    };
    Run();
    for (int i = 0; i < weights_sizes.size(); ++i) {
        EXPECT_EQ(sizeof(int8_t), weights_sizes[i]);
        EXPECT_EQ(Precision::fromType<gna_compound_bias_t>().size(), bias_sizes[i]);
    }
}

TEST_F(GNAHwPrecisionTest, GNAHwPrecisionTestFP32) {
    gna_config = {
        {"GNA_DEVICE_MODE", "GNA_SW_FP32"}
    };
    Run();
    for (int i = 0; i < weights_sizes.size(); ++i) {
        EXPECT_EQ(sizeof(float), weights_sizes[i]);
        EXPECT_EQ(sizeof(float), bias_sizes[i]);
    }
}
} // namespace testing
