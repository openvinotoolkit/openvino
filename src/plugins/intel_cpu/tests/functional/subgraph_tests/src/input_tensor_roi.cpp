// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"

using namespace ngraph;
using namespace ngraph::op;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

struct InputTensorROIParamType {
    ov::PartialShape shape;
    ov::element::Type type;
    ov::Layout layout;
};

typedef std::tuple<
        InputTensorROIParamType,
        std::map<std::string, std::string>   // Device config
> InputTensorROITestParamsSet;


class InputTensorROI : public ::testing::TestWithParam<InputTensorROITestParamsSet> {
public:
    static std::string getTestCaseName(::testing::TestParamInfo<InputTensorROITestParamsSet> obj) {
        std::ostringstream result;
        InputTensorROIParamType param;
        std::map<std::string, std::string> additionalConfig;
        std::tie(param, additionalConfig) = obj.param;
        result << "type=" << param.type << "_";
        result << "shape=" << param.shape << "_";
        result << "layout=" << param.layout.to_string();
        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                result << "_" << item.first << "=" << item.second;
            }
        }


        return result.str();
    }

protected:
    std::shared_ptr<ov::Model>
    create_test_function(element::Type type,
                        const ov::PartialShape & shape,
                        const ov::Layout & layout) {
        ResultVector res;
        ParameterVector params;

        auto param = std::make_shared<opset8::Parameter>(type, shape);
        param->set_friendly_name("input_0");
        param->get_output_tensor(0).set_names({"tensor_input_0"});
        param->set_layout(layout);

        auto constant = opset8::Constant::create(type, {1}, {1});

        auto add = std::make_shared<opset8::Add>(param, constant);
        add->set_friendly_name("Add");

        auto result = std::make_shared<opset8::Result>(add);
        result->set_friendly_name("result_0");
        result->get_output_tensor(0).set_names({"tensor_output_0"});

        params.push_back(param);
        res.push_back(result);

        return std::make_shared<ov::Model>(res, params);
    }

    template<typename T>
    void Run() {
        std::shared_ptr<ov::Core> ie = ov::test::utils::PluginCache::get().core();

        // Compile model

        InputTensorROIParamType param;
        std::map<std::string, std::string> additionalConfig;
        std::tie(param, additionalConfig) = GetParam();
        auto fn_shape = param.shape;
        auto model = create_test_function(param.type, fn_shape, param.layout);
        ov::AnyMap configuration;
        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        auto compiled_model = ie->compile_model(model, "CPU", configuration);

        // Create InferRequest
        ov::InferRequest req = compiled_model.create_infer_request();

        // Create input tensor
        auto input_shape = Shape{ 1, 4, 4, 4 };
        auto input_shape_size = ov::shape_size(input_shape);
        std::vector<T> data(input_shape_size);
        std::iota(data.begin(), data.end(), 0);
        auto input_tensor = ov::Tensor(param.type, input_shape, &data[0]);

        // Set ROI
        auto roi = ov::Tensor(input_tensor, { 0, 1, 1, 1 }, { 1, 3, 3, 3 });
        req.set_tensor("tensor_input_0", roi);

        // Infer
        req.infer();

        // Check result
        auto actual_tensor = req.get_tensor("tensor_output_0");
        auto* actual = actual_tensor.data<T>();
        EXPECT_EQ(actual[0], 21 + 1);
        EXPECT_EQ(actual[1], 22 + 1);
        EXPECT_EQ(actual[2], 25 + 1);
        EXPECT_EQ(actual[3], 26 + 1);
        EXPECT_EQ(actual[4], 37 + 1);
        EXPECT_EQ(actual[5], 38 + 1);
        EXPECT_EQ(actual[6], 41 + 1);
        EXPECT_EQ(actual[7], 42 + 1);
    }
};

TEST_P(InputTensorROI, SetInputTensorROI) {
    InputTensorROIParamType param;
    std::map<std::string, std::string> additionalConfig;
    std::tie(param, additionalConfig) = GetParam();

    if (additionalConfig.count(ov::hint::inference_precision.name())
        && additionalConfig[ov::hint::inference_precision.name()] == ov::element::f16.to_string() &&
        (!(ov::with_cpu_x86_avx512_core_fp16() || ov::with_cpu_x86_avx512_core_amx_fp16()))) {
        GTEST_SKIP() << "Skipping test, platform don't support precision f16";
    }

    switch (param.type) {
        case ov::element::Type_t::f32: {
            Run<float>();
            break;
        }
        case ov::element::Type_t::u8: {
            Run<uint8_t>();
            break;
        }
        default:
            break;
    }
}

static std::vector<InputTensorROIParamType> InputTensorROIParams = {
    { ov::PartialShape{ 1, 2, 2, 2 }, element::f32, "NCHW" },
    { ov::PartialShape{ 1, 2, ov::Dimension::dynamic(), ov::Dimension::dynamic() }, element::f32, "NCHW" },
    { ov::PartialShape{ 1, 2, 2, 2 }, element::u8, "NCHW" },
    { ov::PartialShape{ 1, 2, ov::Dimension::dynamic(), ov::Dimension::dynamic() }, element::u8, "NCHW" },
};

INSTANTIATE_TEST_SUITE_P(smoke_InputTensorROI,
                         InputTensorROI,
                         ::testing::Combine(
                             ::testing::ValuesIn(InputTensorROIParams),
                             ::testing::ValuesIn({cpuEmptyPluginConfig, cpuFP16PluginConfig})),
                         InputTensorROI::getTestCaseName);

} // namespace SubgraphTestsDefinitions
