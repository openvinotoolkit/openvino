// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common/gna_target.hpp"
#include "gna_mock_api_initializer.hpp"
#include "gna_plugin.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ngraph::helpers;
using namespace ov::intel_gna::target;

namespace {

typedef struct ConvParams {
    ConvParams() = default;
    ConvParams(ngraph::Shape inputs_shape,
               ngraph::Shape kernels_shape,
               size_t channels,
               ngraph::Shape maxpool_kernels_shape,
               ActivationTypes activation = ActivationTypes::None)
        : inputs_shape(inputs_shape),
          kernels_shape(kernels_shape),
          channels(channels),
          maxpool_kernels_shape(maxpool_kernels_shape),
          activation(activation) {}

    ngraph::Shape inputs_shape;
    ngraph::Shape kernels_shape;
    size_t channels;
    ngraph::Shape maxpool_kernels_shape;
    ActivationTypes activation;
} ConvParams;

typedef struct {
    DeviceVersion mock_target;
    ConvParams conv;
    const size_t scratch_region_size;
} ConvModelTestParams;

class GNAPluginTest : public GNAPlugin {
public:
    GNAPluginTest() : GNAPlugin(std::map<std::string, std::string>{{GNA_CONFIG_KEY(COMPACT_MODE), CONFIG_VALUE(NO)}}) {}

    size_t get_scratch_region_size() const {
        return this->gnamem->getQueue(rRegion::REGION_SCRATCH)->getSize();
    }
};

class ConvolutionScratchAreaSizeTest : public ::testing::Test,
                                       public ::testing::WithParamInterface<ConvModelTestParams> {
protected:
    void Run() {
        GNAPluginTest gna_plugin{};
        InferenceEngine::CNNNetwork cnn_network{m_function};

        gna_plugin.LoadNetwork(cnn_network);
        gna_plugin.get_scratch_region_size();

        EXPECT_EQ(m_test_param.scratch_region_size, gna_plugin.get_scratch_region_size());
    }

    void SetUp() override {
        m_mock.set_gna_device_version(DeviceToGna(m_test_param.mock_target));
        m_mock.init();
        create_model();
    }

private:
    void create_model() {
        auto inputs = std::make_shared<ngraph::opset9::Parameter>(m_prec, m_test_param.conv.inputs_shape);
        std::shared_ptr<ngraph::Node> conv;
        std::shared_ptr<ngraph::Node> maxpool;
        std::shared_ptr<ngraph::Node> activation;
        std::shared_ptr<ngraph::Node> result_operation;

        conv = ngraph::builder::makeConvolution(
            inputs,
            m_prec,
            m_test_param.conv.kernels_shape,
            // Set the stride and kernel shape to the same value in order to force padding equal to 0
            m_test_param.conv.kernels_shape,
            {0, 0},
            {0, 0},
            {1, 1},
            ngraph::op::PadType::VALID,
            m_test_param.conv.channels);
        result_operation = conv;

        if (m_test_param.conv.maxpool_kernels_shape.size()) {
            maxpool = ngraph::builder::makePooling(
                result_operation,
                // Set the stride and kernel shape to the same value in order to force padding equal to 0
                m_test_param.conv.maxpool_kernels_shape,
                {0, 0},
                {0, 0},
                m_test_param.conv.maxpool_kernels_shape,
                ngraph::op::RoundingType::FLOOR,
                ngraph::op::PadType::VALID,
                false,
                PoolingTypes::MAX);
            result_operation = maxpool;
        }

        if (m_test_param.conv.activation != ActivationTypes::None) {
            activation = ngraph::builder::makeActivation(result_operation, m_prec, m_test_param.conv.activation);
            result_operation = activation;
        }

        auto result = std::make_shared<ngraph::opset9::Result>(result_operation);
        m_function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                        ov::ParameterVector{inputs},
                                                        "Convolution");
    }

    std::shared_ptr<ngraph::Function> m_function;
    const ov::element::Type m_prec = ngraph::element::f32;
    ConvModelTestParams m_test_param = GetParam();
    GnaMockApiInitializer m_mock;
};

// Memory regions assignment for tested model:
//           |   inp/out   | precision | GNA3.0  |  GNA3.5+   |
//  input    |------------------------------------------------|
//    |      | conv inp    |    I16    | Input   | Input      |
//  conv     |             |           |         |            |
//    |      | conv out    |    I32    | Scratch | Int buffer |
//  maxPool  |             |           |         |            |
//    |      | maxPool out |    I16    | Scratch | Int buffer |
//  relu     |             |           |         |            |
//    |      | relu out    |    I16    | Output  | Output     |
//  output   |             |           |         |            |
//

std::vector<ConvModelTestParams> tests_vectors{
    // target | inp shape | conv kernel shape | conv out count | maxPool kernel shape | activation func | scratch size
    // conv
    {DeviceVersion::GNA3_0, ConvParams({1, 8, 16, 32}, {2, 2}, 8, {}), 0},
    {DeviceVersion::GNA3_5, ConvParams({1, 8, 16, 32}, {2, 2}, 8, {}), 0},
    // conv + maxpool
    {DeviceVersion::GNA3_0, ConvParams({1, 8, 16, 32}, {2, 2}, 8, {2, 2}), 4096},
    {DeviceVersion::GNA3_5, ConvParams({1, 8, 16, 32}, {2, 2}, 8, {2, 2}), 0},
    // conv + maxpool + relu
    {DeviceVersion::GNA3_0, ConvParams({1, 8, 16, 32}, {2, 2}, 8, {2, 2}, ActivationTypes::Relu), 4096 + 512},
    {DeviceVersion::GNA3_5, ConvParams({1, 8, 16, 32}, {2, 2}, 8, {2, 2}, ActivationTypes::Relu), 0}};

TEST_P(ConvolutionScratchAreaSizeTest, CheckScratchAreaSizeForConvolution) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(ScratchMemoryTest, ConvolutionScratchAreaSizeTest, ::testing::ValuesIn(tests_vectors));
}  // namespace