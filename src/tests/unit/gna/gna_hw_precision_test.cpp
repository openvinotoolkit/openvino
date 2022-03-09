// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>
#include "ngraph_functions/builders.hpp"
#include "gna_plugin.hpp"

// #include "shared_test_classes/base/layer_test_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace InferenceEngine;

namespace testing {

class GNAPluginForPrecisionTest : public GNAPluginNS::GNAPlugin {
public:
    std::vector<intel_dnn_component_t> getComponents() {
        return this->dnn->component;
    }
};

class GNAHwPrecisionTest: public ::testing::Test {
public:
void SetConfig(std::map<std::string, std::string> & config) {
    gna_config = config;
    // configuration.insert(gna_config.begin(), gna_config.end());
}

void Run() {
    auto function = getFunction();
    CNNNetwork cnnNetwork = CNNNetwork{function};
    auto plugin = std::make_shared<GNAPluginForPrecisionTest>();
    plugin->SetConfig(gna_config);
    plugin->LoadNetwork(cnnNetwork);

    auto components = plugin->getComponents();
    for (int k = 0; k < components.size(); ++k) {
        if (components[k].operation == kDnnAffineOp || components[k].operation == kDnnDiagonalOp) {
            weights_size = components[k].op.affine.num_bytes_per_weight;
            bias_size = components[k].op.affine.num_bytes_per_bias;
            // auto gnaPrc =  gna_config.find("GNA_PRECISION");
            // if (gnaPrc != gna_config.end() && gnaPrc->second != "I16") {

            // } else if (gnaPrc != gna_config.end() && gnaPrc->second != "I16") {

            // }
            // components[k].op.affine.num_bytes_per_weight
            // components[k].op.affine.num_bytes_per_bias
        }
    }
}

protected:
    std::shared_ptr<ov::Model> getFunction() {
        // auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto params = ngraph::builder::makeParams(netPrecision, {shape});
        auto secondInput = ngraph::builder::makeInputLayer(netPrecision, ngraph::helpers::InputLayerType::CONSTANT, shape);
        // auto secondaryInput = ngraph::builder::makeInputLayer(ngraph::element::f32, ngraph::helpers::InputLayerType::PARAMETER, shape);
        auto matmul = ngraph::builder::makeMatMul(params[0], secondInput, false, true);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(matmul)};
        auto function = std::make_shared<ngraph::Function>(results, params, "MatMul");
        return function;
    }
    // Precision netPrecision = Precision::FP32;
    ngraph::element::Type netPrecision = ngraph::element::f32;
    // ngraph::Shape shape = ngraph::Shape({1, 10});
    ngraph::Shape shape = {1, 10};
    std::map<std::string, std::string> gna_config;
    int weights_size;
    int bias_size;
};

TEST_F(GNAHwPrecisionTest, GNAHwPrecisionTestI16) {
    std::map<std::string, std::string> config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1"},
        {"GNA_PRECISION", "I16"}
    };
    SetConfig(config);
    Run();
    EXPECT_EQ(sizeof(int16_t), weights_size);
    EXPECT_EQ(sizeof(uint32_t), bias_size);
}


TEST_F(GNAHwPrecisionTest, GNAHwPrecisionTestI8) {
    std::map<std::string, std::string> config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1"},
        {"GNA_PRECISION", "I8"}
    };
    SetConfig(config);
    Run();
    EXPECT_EQ(sizeof(int8_t), weights_size);
    EXPECT_EQ(Precision::fromType<gna_compound_bias_t>().size(), bias_size);
}

} // namespace testing