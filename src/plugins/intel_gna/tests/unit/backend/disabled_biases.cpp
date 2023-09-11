// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <any_copy.hpp>
#include <common_test_utils/data_utils.hpp>
#include <memory>
#include <ngraph_functions/builders.hpp>
#include <openvino/core/model.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/opsets/opset12.hpp>
#include <openvino/pass/manager.hpp>
#include <vector>

#include "../gna_mock_api.hpp"
#include "gna2_model_helper.hpp"
#include "gna_plugin.hpp"
#include "request/worker_impl.hpp"
#include "request/worker_pool.hpp"

using namespace ::testing;

static constexpr uint32_t kConvolutionBiasOperandIndex = 3;
static constexpr uint32_t kConvolutionChannelsNumber = 8;
constexpr uint32_t kChannelIndexNCHW = 1;

std::shared_ptr<ov::Model> create_conv_model(const ov::Shape& input_shape,
                                             const ov::Shape& kernel_shape,
                                             const std::string& model_name,
                                             bool with_add = false) {
    auto ng_precision = ov::element::f32;
    const std::vector<size_t> strides{1, 1};
    const std::vector<ptrdiff_t> pads_begin{0, 0};
    const std::vector<ptrdiff_t> pads_end{0, 0};
    const std::vector<size_t> dilations{1, 1};
    const ov::op::PadType pad_type = ov::op::PadType::VALID;

    ov::ParameterVector params{std::make_shared<ov::opset12::Parameter>(ng_precision, input_shape)};

    const auto weights_size =
        ov::shape_size(kernel_shape) * kConvolutionChannelsNumber * input_shape[kChannelIndexNCHW];
    auto weights_values = ov::test::utils::generate_float_numbers(weights_size, -0.2f, 0.2f);

    auto convolution = ngraph::builder::makeConvolution(params[0],
                                                        ng_precision,
                                                        kernel_shape,
                                                        strides,
                                                        pads_begin,
                                                        pads_end,
                                                        dilations,
                                                        pad_type,
                                                        kConvolutionChannelsNumber,
                                                        false,
                                                        weights_values);
    std::shared_ptr<ov::opset12::Result> result;

    if (with_add) {
        ov::Shape bias_shape = {1, input_shape[kChannelIndexNCHW], 1, 1};
        std::vector<float> bias_data(std::accumulate(bias_shape.begin(), bias_shape.end(), 1, std::multiplies<float>()),
                                     +1.0f);
        auto bias = std::make_shared<ngraph::opset1::Constant>(ng_precision, bias_shape, bias_data);
        auto add = std::make_shared<ngraph::opset1::Add>(convolution, bias);

        result = std::make_shared<ov::opset12::Result>(add);
    } else {
        result = std::make_shared<ov::opset12::Result>(convolution);
    }

    auto model = std::make_shared<ov::Model>(result, params, model_name);

    return model;
}

class GNAPluginForDisabledBiasesTest : public GNAPlugin {
public:
    GNAPluginForDisabledBiasesTest(const std::map<std::string, std::string>& configMap) : GNAPlugin(configMap) {}
    ~GNAPluginForDisabledBiasesTest() override = default;

    const size_t get_memory_REGION_RO_request_num() const;

    Gna2Model* get_gna_model() const;
};

const size_t GNAPluginForDisabledBiasesTest::get_memory_REGION_RO_request_num() const {
    return this->gnamem->getQueue(ov::intel_gna::memory::REGION_RO)->_mem_requests.size();
}

Gna2Model* GNAPluginForDisabledBiasesTest::get_gna_model() const {
    return requestWorkerPool_->firstWorker().model();
}

struct DisabledBiasesTestParams {
    std::shared_ptr<ov::Model> model;           // execution model
    std::map<std::string, std::string> config;  // plugin config
    Gna2OperationType operation_type;           // operation to check
    uint32_t index_of_operand_to_check;         // index of operand to check
    bool should_be_nullptr;                     // true if operand should be nullptr
    uint32_t expected_num_ro_region_request;    // number of RO region request
};

class DisabledBiasesTest : public ::testing::TestWithParam<DisabledBiasesTestParams> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<DisabledBiasesTestParams>& obj) {
        std::ostringstream test_name;
        test_name << "Model=" << obj.param.model->get_friendly_name() << "_";
        test_name << "OperationType=" << GetLayerType(obj.param.operation_type) << "_";
        test_name << "BiasDisabled=" << obj.param.should_be_nullptr;

        return test_name.str();
    }

protected:
    void SetUp() override;

    void set_expect_loadnetwork_calls(
        GNACppApi* mockApi,
        std::vector<std::vector<uint8_t>>& data,
        Gna2DeviceVersion device_version = Gna2DeviceVersion::Gna2DeviceVersionSoftwareEmulation);
    DisabledBiasesTestParams m_params;
    InferenceEngine::CNNNetwork m_cnn_network;
    std::vector<std::vector<uint8_t>> m_data;
    GNACppApi m_mock_api;
};

void DisabledBiasesTest::SetUp() {
    m_params = GetParam();
    m_cnn_network = InferenceEngine::CNNNetwork{m_params.model};
    set_expect_loadnetwork_calls(&m_mock_api, m_data);
}

void DisabledBiasesTest::set_expect_loadnetwork_calls(GNACppApi* mock_api,
                                                      std::vector<std::vector<uint8_t>>& data,
                                                      Gna2DeviceVersion device_version_to_set) {
    EXPECT_CALL(*mock_api, Gna2MemoryAlloc(_, _, _))
        .Times(AtLeast(1))
        .WillRepeatedly(Invoke([&data](uint32_t requested_size, uint32_t* granted_size, void** memory_address) {
            data.push_back(std::vector<uint8_t>(requested_size));
            *granted_size = requested_size;
            *memory_address = data.back().data();
            return Gna2StatusSuccess;
        }));

    EXPECT_CALL(*mock_api, Gna2DeviceGetVersion(_, _))
        .WillOnce(Invoke([device_version_to_set](uint32_t deviceIndex, enum Gna2DeviceVersion* device_version) {
            *device_version = device_version_to_set;
            return Gna2StatusSuccess;
        }));

    EXPECT_CALL(*mock_api, Gna2DeviceOpen(_)).WillOnce(Return(Gna2StatusSuccess));

    EXPECT_CALL(*mock_api, Gna2GetLibraryVersion(_, _)).Times(AtLeast(0)).WillRepeatedly(Return(Gna2StatusSuccess));

    EXPECT_CALL(*mock_api, Gna2InstrumentationConfigCreate(_, _, _, _)).WillOnce(Return(Gna2StatusSuccess));

    EXPECT_CALL(*mock_api, Gna2ModelCreate(_, _, _))
        .WillOnce(Invoke([](uint32_t device_index, struct Gna2Model const* model, uint32_t* model_id) {
            *model_id = 0;
            return Gna2StatusSuccess;
        }));

    EXPECT_CALL(*mock_api, Gna2RequestConfigCreate(_, _))
        .WillOnce(Invoke([](uint32_t model_id, uint32_t* request_config_id) {
            *request_config_id = 0;
            return Gna2StatusSuccess;
        }));

    EXPECT_CALL(*mock_api, Gna2InstrumentationConfigAssignToRequestConfig(_, _))
        .Times(AtLeast(1))
        .WillRepeatedly(Return(Gna2StatusSuccess));

    InSequence seq;
    EXPECT_CALL(*mock_api, Gna2DeviceClose(_)).WillOnce(Return(Gna2StatusSuccess));
    EXPECT_CALL(*mock_api, Gna2MemoryFree(_)).Times(AtLeast(1)).WillRepeatedly(Return(Gna2StatusSuccess));
}

TEST_P(DisabledBiasesTest, CompareInpShapeVsReservedMemRegion) {
    GNAPluginForDisabledBiasesTest plugin(m_params.config);
    plugin.LoadNetwork(m_cnn_network);

    auto gna_model = plugin.get_gna_model();
    ASSERT_NE(gna_model, nullptr);

    bool at_least_one_hit = false;

    uint32_t ro_request_number = 0;
    uint32_t ro_request_size = 0;
    for (uint32_t i = 0; i < gna_model->NumberOfOperations; ++i) {
        auto& operation = gna_model->Operations[i];
        // Check if there is expected operation
        if (operation.Type == m_params.operation_type) {
            at_least_one_hit = true;
            // Check if bias operand is nullptr or valid ptr depending on network.
            ASSERT_GE(operation.NumberOfOperands, m_params.index_of_operand_to_check);
            EXPECT_EQ(operation.Operands[m_params.index_of_operand_to_check] == nullptr, m_params.should_be_nullptr);
        }
    }

    EXPECT_EQ(m_params.expected_num_ro_region_request, plugin.get_memory_REGION_RO_request_num());

    // TO ensure that at least one assert was done. To ensure that there was no issue with model.
    EXPECT_TRUE(at_least_one_hit);
}

const ov::Shape input_shape_1d = {1, kConvolutionChannelsNumber, 1, 32};
const ov::Shape kernel_1d = {1, 1};
const ov::Shape input_shape_2d = {1, kConvolutionChannelsNumber, 32, 32};
const ov::Shape kernel_2d = {2, 2};

const std::map<std::string, std::string> configs = {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}};

const uint32_t kExpectedConvolutionROAllocationsWithBiases = 2;
const uint32_t kExpectedConvolutionROAllocationsWithoutBiases = 1;

const DisabledBiasesTestParams conv2d_no_biases_param = {create_conv_model(input_shape_2d, kernel_2d, "Convolution2D"),
                                                         configs,
                                                         Gna2OperationTypeConvolution,
                                                         kConvolutionBiasOperandIndex,
                                                         true,
                                                         kExpectedConvolutionROAllocationsWithoutBiases};

const DisabledBiasesTestParams conv2d_with_biases_param = {
    create_conv_model(input_shape_2d, kernel_2d, "Convolution2DWithAdd", true),
    configs,
    Gna2OperationTypeConvolution,
    kConvolutionBiasOperandIndex,
    false,
    kExpectedConvolutionROAllocationsWithBiases};

INSTANTIATE_TEST_SUITE_P(biases_check_convolution2d,
                         DisabledBiasesTest,
                         ::testing::Values(conv2d_no_biases_param, conv2d_with_biases_param),
                         DisabledBiasesTest::get_test_case_name);
