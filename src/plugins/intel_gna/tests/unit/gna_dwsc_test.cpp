// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common/gna_target.hpp"
#include "gna_mock_api_initializer.hpp"
#include "gna_plugin.hpp"
#include "ngraph_functions/builders.hpp"

namespace {
typedef struct {
    const std::vector<size_t> input_size;
    const std::vector<size_t> filter_size;
    const std::vector<ptrdiff_t> pads_begin;
    const std::vector<ptrdiff_t> pads_end;
    const size_t num_groups;
} GroupConvModel;

const std::vector<GroupConvModel> models{{{1, 8, 32, 1}, {2, 1}, {1, 0}, {1, 0}, 8},
                                         {{1, 8, 1, 32}, {1, 2}, {0, 1}, {0, 1}, 8}};

typedef struct {
    GroupConvModel model;
    Gna2DeviceVersion mock_target;
    bool load_succesfull;
} GroupConvModelTestParams;

std::vector<GroupConvModelTestParams> all_tests{
    {models[0], Gna2DeviceVersion3_5, false},
    {models[1], Gna2DeviceVersion3_5, false},
    {models[0], static_cast<Gna2DeviceVersion>(target::DeviceVersion::GNA3_6), true},
    {models[1], static_cast<Gna2DeviceVersion>(target::DeviceVersion::GNA3_6), true}};

class GNAPluginDwscLoadTest : public ::testing::Test, public ::testing::WithParamInterface<GroupConvModelTestParams> {
    std::shared_ptr<ngraph::Function> function;

protected:
    void Run() {
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

        auto conv = std::dynamic_pointer_cast<ngraph::opset9::GroupConvolution>(
            ngraph::builder::makeGroupConvolution(parameter,
                                                  f32,
                                                  model.filter_size,
                                                  c_strides,
                                                  model.pads_begin,
                                                  model.pads_end,
                                                  c_dilations,
                                                  ngraph::op::PadType::EXPLICIT,
                                                  c_num_out_channels,
                                                  model.num_groups));
        auto result = std::make_shared<ngraph::opset9::Result>(conv);
        function = std::make_shared<ngraph::Function>(result, ov::ParameterVector{parameter}, "group_convolution");
    }
};

TEST_P(GNAPluginDwscLoadTest, ReturnsSpecificGna2DeviceVersion) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_LoadDwsc, GNAPluginDwscLoadTest, ::testing::ValuesIn(all_tests));

}  // namespace
