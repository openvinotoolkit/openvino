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

class GNAPluginTest : public GNAPlugin {
public:
    GNAPluginTest() : GNAPlugin(std::map<std::string, std::string>{{GNA_CONFIG_KEY(COMPACT_MODE), CONFIG_VALUE(NO)}}) {}

    size_t get_scratch_region_requests_cnt() const {
        return this->gnamem->getQueue(rRegion::REGION_SCRATCH)->futureHeap().size();
    }

    size_t get_scratch_region_size() const {
        return this->gnamem->getQueue(rRegion::REGION_SCRATCH)->getSize();
    }
};

template <typename T>
class ScratchAreaSizeTestBase : public ::testing::Test, public ::testing::WithParamInterface<T> {
protected:
    ScratchAreaSizeTestBase(DeviceVersion target, size_t scratch_region_requests_cnt, size_t scratch_region_size)
        : m_target(target),
          m_scratch_region_requests_cnt(scratch_region_requests_cnt),
          m_scratch_region_size(scratch_region_size) {}

    virtual std::shared_ptr<ngraph::Function> create_model() = 0;

    void Run() {
        GNAPluginTest gna_plugin{};
        InferenceEngine::CNNNetwork cnn_network{m_function};

        gna_plugin.LoadNetwork(cnn_network);

        EXPECT_EQ(m_scratch_region_requests_cnt, gna_plugin.get_scratch_region_requests_cnt());
        EXPECT_EQ(m_scratch_region_size, gna_plugin.get_scratch_region_size());
    }

    void SetUp() override {
        m_mock.set_gna_device_version(DeviceToGna(m_target));
        m_mock.init();
        m_function = create_model();
    }

private:
    std::shared_ptr<ngraph::Function> m_function;
    GnaMockApiInitializer m_mock;
    DeviceVersion m_target;
    size_t m_scratch_region_requests_cnt;
    size_t m_scratch_region_size;
};

struct ConvolutionTestParams {
    DeviceVersion target;
    size_t scratch_region_requests_cnt;
    size_t scratch_region_size;
    ngraph::Shape input_shape;
    ngraph::Shape kernels_shape;
    size_t channels;
    ngraph::Shape maxpool_kernels_shape;
    ActivationTypes activation;
};

class ConvolutionScratchAreaSizeTest : public ScratchAreaSizeTestBase<ConvolutionTestParams> {
public:
    ConvolutionScratchAreaSizeTest()
        : ScratchAreaSizeTestBase(GetParam().target,
                                  GetParam().scratch_region_requests_cnt,
                                  GetParam().scratch_region_size) {}

private:
    std::shared_ptr<ngraph::Function> create_model() override {
        ConvolutionTestParams param = GetParam();
        const ov::element::Type ng_prec = ngraph::element::f32;
        const auto input_shape = std::make_shared<ngraph::opset9::Parameter>(ng_prec, param.input_shape);
        std::shared_ptr<ngraph::Node> maxpool;
        std::shared_ptr<ngraph::Node> activation;
        std::shared_ptr<ngraph::Node> result_operation;

        result_operation = ngraph::builder::makeConvolution(
            input_shape,
            ng_prec,
            param.kernels_shape,
            // Set the stride and kernel shape to the same value in order to force padding equal to 0
            param.kernels_shape,
            {0, 0},
            {0, 0},
            {1, 1},
            ngraph::op::PadType::VALID,
            param.channels);

        if (param.maxpool_kernels_shape.size()) {
            maxpool = ngraph::builder::makePooling(
                result_operation,
                // Set the stride and kernel shape to the same value in order to force padding equal to 0
                param.maxpool_kernels_shape,
                {0, 0},
                {0, 0},
                param.maxpool_kernels_shape,
                ngraph::op::RoundingType::FLOOR,
                ngraph::op::PadType::VALID,
                false,
                PoolingTypes::MAX);
            result_operation = maxpool;
        }

        if (param.activation != ActivationTypes::None) {
            activation = ngraph::builder::makeActivation(result_operation, ng_prec, param.activation);
            result_operation = activation;
        }

        auto result = std::make_shared<ngraph::opset9::Result>(result_operation);

        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ov::ParameterVector{input_shape},
                                                  "Convolution");
    }
};

struct FullyConnectedTestParams {
    DeviceVersion target;
    size_t scratch_region_requests_cnt;
    size_t scratch_region_size;
    ngraph::Shape input_shape;
    size_t out_size;
    ActivationTypes activation;
};

class FullyConnectedScratchAreaSizeTest : public ScratchAreaSizeTestBase<FullyConnectedTestParams> {
public:
    FullyConnectedScratchAreaSizeTest()
        : ScratchAreaSizeTestBase(GetParam().target,
                                  GetParam().scratch_region_requests_cnt,
                                  GetParam().scratch_region_size) {}

private:
    std::shared_ptr<ngraph::Function> create_model() override {
        FullyConnectedTestParams param = GetParam();
        const ov::element::Type ng_prec = ngraph::element::f32;
        const auto input_shape = std::make_shared<ngraph::opset9::Parameter>(ng_prec, param.input_shape);
        std::shared_ptr<ngraph::Node> activation;
        std::shared_ptr<ngraph::Node> result_operation;

        result_operation = ngraph::builder::makeFullyConnected(input_shape, ng_prec, param.out_size, false);

        if (param.activation != ActivationTypes::None) {
            activation = ngraph::builder::makeActivation(result_operation, ng_prec, param.activation);
            result_operation = activation;
        }

        auto result = std::make_shared<ngraph::opset9::Result>(result_operation);

        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ov::ParameterVector{input_shape},
                                                  "FullyConnected");
    }
};

struct EltwiseTestParams {
    DeviceVersion target;
    size_t scratch_region_requests_cnt;
    size_t scratch_region_size;
    ngraph::Shape input_shape;
    ActivationTypes activation;
};

class EltwiseScratchAreaSizeTest : public ScratchAreaSizeTestBase<EltwiseTestParams> {
public:
    EltwiseScratchAreaSizeTest()
        : ScratchAreaSizeTestBase(GetParam().target,
                                  GetParam().scratch_region_requests_cnt,
                                  GetParam().scratch_region_size) {}

private:
    std::shared_ptr<ngraph::Function> create_model() override {
        EltwiseTestParams param = GetParam();
        const ov::element::Type ng_prec = ngraph::element::f32;
        const auto input_shape = std::make_shared<ngraph::opset9::Parameter>(ng_prec, param.input_shape);
        std::shared_ptr<ngraph::Node> activation;
        std::shared_ptr<ngraph::Node> result_operation;

        result_operation =
            ngraph::builder::makeEltwise(input_shape, input_shape, ngraph::helpers::EltwiseTypes::MULTIPLY);

        if (param.activation != ActivationTypes::None) {
            activation = ngraph::builder::makeActivation(result_operation, ng_prec, param.activation);
            result_operation = activation;
        }

        auto result = std::make_shared<ngraph::opset9::Result>(result_operation);

        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ov::ParameterVector{input_shape},
                                                  "Eltwise");
    }
};

// Memory regions assignment for model with convolution:
//           |   inp/out   | precision | GNA3.0  |      GNA3.5+     |
//  input    |------------------------------------------------------|
//    |      | conv inp    |    I16    | Input   | Input            |
//  conv     |             |           |         |                  |
//    |      | conv out    |    I32    | Scratch | Internal buffer  |
//  maxPool  |             |           |         |                  |
//    |      | maxPool out |    I16    | Scratch | Iternal buffer   |
//  relu     |             |           |         |                  |
//    |      | relu out    |    I16    | Output  | Output           |
//  output   |             |           |         |                  |
//
// conv scrach size = H/2 * W/2 * outC * prec (4) maxpool scratch size = H/2 * W/2 * C * prec (2)
// params: target | scratch reuquest cnt | scratch size | inp shape | conv kernel shape | conv out count |
// maxPool kernel shape | activation
std::vector<ConvolutionTestParams> vectors_for_conv_test{
    // only convolution
    {DeviceVersion::GNA3_0, 0, 0, {1, 8, 16, 32}, {2, 2}, 8, {}},
    {DeviceVersion::GNA3_5, 0, 0, {1, 8, 16, 32}, {2, 2}, 8, {}},
    // convolution + maxpool
    {DeviceVersion::GNA3_0, 1, 4096, {1, 8, 16, 32}, {2, 2}, 8, {2, 2}},
    {DeviceVersion::GNA3_5, 0, 0, {1, 8, 16, 32}, {2, 2}, 8, {2, 2}},
    // convolution + maxpool + relu
    {DeviceVersion::GNA3_0, 2, 4096 + 512, {1, 8, 16, 32}, {2, 2}, 8, {2, 2}, ActivationTypes::Relu},
    {DeviceVersion::GNA3_5, 0, 0, {1, 8, 16, 32}, {2, 2}, 8, {2, 2}, ActivationTypes::Relu}};

TEST_P(ConvolutionScratchAreaSizeTest, CheckScratchAreaSizeForConvolution) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(ScratchMemoryTest, ConvolutionScratchAreaSizeTest, ::testing::ValuesIn(vectors_for_conv_test));

// Memory regions assignment for model with fullyConnected layer:
//                  |   inp/out   | precision | GNA3.0  |     GNA3.5+       |
//     input        |-------------------------------------------------------|
//       |          | conv inp    |    I16    | Input   | Input             |
// fullyConnected   |             |           |         |                   |
//       |          | conv out    |    I32    | Scratch | Scratch reserved  |
//       |          |             |           |         | by GNA library    |
//     relu         |             |           |         |                   |
//       |          | relu out    |    I16    | Output  | Output            |
//     output       |             |           |         |                   |
//
// fullyConnected scratch size = H * out * prec (4)
// params: target | scratch reuquest cnt |scratch size | inp shape | out size | activation
std::vector<FullyConnectedTestParams> vectors_for_fullyconnected_test{
    // only fullyConnected
    {DeviceVersion::GNA3_0, 0, 0, {6, 7}, 8, {}},
    {DeviceVersion::GNA3_5, 0, 0, {6, 7}, 8, {}},
    // fullyConnected + relu
    {DeviceVersion::GNA3_0, 1, 6 * 8 * 4, {6, 7}, 8, ActivationTypes::Relu},
    {DeviceVersion::GNA3_5, 0, 0, {6, 7}, 8, ActivationTypes::Relu}};

TEST_P(FullyConnectedScratchAreaSizeTest, CheckScratchAreaSizeForFullyConnected) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(ScratchMemoryTest,
                         FullyConnectedScratchAreaSizeTest,
                         ::testing::ValuesIn(vectors_for_fullyconnected_test));

// Memory regions assignment for model with eltwise layer:
//                  |   inp/out   | precision | GNA3.0  |      GNA3.5+      |
//     input        |-------------------------------------------------------|
//       |          | conv inp    |    I16    | Input   | Input             |
//    eltwise       |             |           |         |                   |
//       |          | conv out    |    I32    | Scratch | Scratch reserved  |
//       |          |             |           |         | by GNA library    |
//     relu         |             |           |         |                   |
//       |          | relu out    |    I16    | Output  | Output            |
//     output       |             |           |         |                   |
//
// eltwise scratch size = H * (W + padding) * prec (4)
// params: target | scratch reuquest cnt | scratch size | inp shape | activation
std::vector<EltwiseTestParams> vectors_for_eltwise_test{
    // only eltwise
    {DeviceVersion::GNA3_0, 0, 0, {8, 8}, {}},
    {DeviceVersion::GNA3_5, 0, 0, {8, 8}, {}},
    // eltwise + relu
    {DeviceVersion::GNA3_0, 1, 8 * 8 * 4, {8, 8}, ActivationTypes::Relu},
    {DeviceVersion::GNA3_5, 0, 0, {8, 8}, ActivationTypes::Relu}};

TEST_P(EltwiseScratchAreaSizeTest, CheckScratchAreaSizeForEltwise) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(ScratchMemoryTest, EltwiseScratchAreaSizeTest, ::testing::ValuesIn(vectors_for_eltwise_test));
}  // namespace