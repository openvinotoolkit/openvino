// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common/gna_target.hpp"
#include "gna_mock_api_initializer.hpp"
#include "gna_plugin.hpp"
#include "ov_models/builders.hpp"

namespace {
typedef struct {
    const std::vector<size_t> input_size;
    const std::vector<size_t> filter_size;
    const std::vector<ptrdiff_t> pads_begin;
    const std::vector<ptrdiff_t> pads_end;
} ConvModel;

const std::vector<ConvModel> models{{{1, 8, 32, 1}, {2, 1}, {1, 0}, {1, 0}}, {{1, 8, 1, 32}, {1, 2}, {0, 1}, {0, 1}}};

typedef struct {
    ConvModel model;
    Gna2DeviceVersion mock_target;
    bool load_succesfull;
} ConvModelTestParams;

std::vector<ConvModelTestParams> all_tests{{models[0], Gna2DeviceVersion::Gna2DeviceVersion3_0, false},
                                           {models[1], Gna2DeviceVersion::Gna2DeviceVersion3_0, false},
                                           {models[0], Gna2DeviceVersion::Gna2DeviceVersion3_5, true},
                                           {models[1], Gna2DeviceVersion::Gna2DeviceVersion3_5, true}};

class GNAPluginLoadNetworkTest : public ::testing::Test, public ::testing::WithParamInterface<ConvModelTestParams> {
    std::shared_ptr<ngraph::Function> function;

protected:
    void Run() {
        SetUp();
        const auto test_parameter = GetParam();
        GnaMockApiInitializer mock;
        mock.set_gna_device_version(test_parameter.mock_target);
        mock.set_create_model(test_parameter.load_succesfull);
        mock.init();

        GNAPlugin gna_plugin{};
        InferenceEngine::CNNNetwork cnn_network{function};
        bool load_succesfull = true;
        try {
            gna_plugin.LoadNetwork(cnn_network);
        } catch (std::exception&) {
            load_succesfull = false;
        }
        EXPECT_EQ(test_parameter.load_succesfull, load_succesfull);
    }

    void SetUp() override {
        const std::vector<size_t> c_strides{1, 1};
        const std::vector<size_t> c_dilations{1, 1};
        constexpr size_t c_num_out_channels = 8;
        const auto& model = GetParam().model;

        using ngraph::element::f32;
        auto parameter = std::make_shared<ngraph::opset9::Parameter>(f32, ngraph::Shape{model.input_size});

        auto conv = std::dynamic_pointer_cast<ngraph::opset9::Convolution>(
            ngraph::builder::makeConvolution(parameter,
                                             f32,
                                             model.filter_size,
                                             c_strides,
                                             model.pads_begin,
                                             model.pads_end,
                                             c_dilations,
                                             ngraph::op::PadType::EXPLICIT,
                                             c_num_out_channels));
        auto result = std::make_shared<ngraph::opset9::Result>(conv);
        function = std::make_shared<ngraph::Function>(result, ov::ParameterVector{parameter}, "convolution");
    }
};

// This test covers GNAGraphCompiler::ShouldUseOnlyConv2DGnaIface()
// behavior when specific Gna2DeviceVersion is detected by Gna2DeviceGetVersion()
TEST_P(GNAPluginLoadNetworkTest, ReturnsSpecificGna2DeviceVersion) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_LoadConvolution1D, GNAPluginLoadNetworkTest, ::testing::ValuesIn(all_tests));
}  // namespace
