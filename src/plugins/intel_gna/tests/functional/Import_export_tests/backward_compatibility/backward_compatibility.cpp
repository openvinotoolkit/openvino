// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <map>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type.hpp"
#include "openvino/opsets/opset9.hpp"
#include "helpers/test_model_repo.hpp"
#include "ngraph_functions/builders.hpp"


using namespace ov::opset9;

typedef std::tuple<
        ov::element::Type,                  // Network Precision
        std::string,                        // Target Device
        std::string,                        // Name Export Model
        std::map<std::string, std::string>, // Export Configuration
        std::map<std::string, std::string>  // Import Configuration
> exportImportNetworkParams;

class BackwardCompatibility : public testing::WithParamInterface<exportImportNetworkParams>,
                                   virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<exportImportNetworkParams> obj) {
        ov::element::Type input_prc;
        std::string targetDevice;
        std::map<std::string, std::string> exportConfiguration;
        std::map<std::string, std::string> importConfiguration;
        std::string nameExportModel;
        std::tie(input_prc, targetDevice, nameExportModel, exportConfiguration, importConfiguration) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << input_prc << "_";
        result << "targetDevice=" << targetDevice << "_";
        result << "nameExportModel=" << nameExportModel << "_";
        for (auto const& configItem : exportConfiguration) {
            result << "_exportConfigItem=" << configItem.first << "_" << configItem.second;
        }
        for (auto const& configItem : importConfiguration) {
            result << "_importConfigItem=" << configItem.first << "_" << configItem.second;
        }
        return result.str();
    }

    void Run() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        functionRefs = ngraph::clone_function(*function);
        // load export configuration and save outputs
        configuration.insert(exportConfiguration.begin(), exportConfiguration.end());
        LoadNetwork();
        GenerateInputs();
        Infer();
        auto actualOutputs = GetOutputs();

        auto referenceOutputs = CalculateRefs();
        Compare(referenceOutputs, actualOutputs);

        for (auto const& configItem : importConfiguration) {
            configuration[configItem.first] = configItem.second;
        }

        const auto compiledExecNetwork = executableNetwork;
        auto model = TestDataHelpers::get_data_path() + "/gna/" + m_export_model_name;

        const auto importedExecNetwork = core->ImportNetwork(model, targetDevice, configuration);

        GenerateInputs();
        Infer();

        ASSERT_EQ(importedExecNetwork.GetInputsInfo().size(), compiledExecNetwork.GetInputsInfo().size());
        ASSERT_EQ(importedExecNetwork.GetOutputsInfo().size(), compiledExecNetwork.GetOutputsInfo().size());

        for (const auto& next_output : importedExecNetwork.GetOutputsInfo()) {
            ASSERT_NO_THROW(compiledExecNetwork.GetOutputsInfo()[next_output.first]);
        }
        auto importedOutputs = GetOutputs();

        ASSERT_EQ(actualOutputs.size(), importedOutputs.size());

        for (size_t i = 0; i < actualOutputs.size(); i++) {
            Compare(actualOutputs[i], importedOutputs[i]);
        }
    }

protected:
    void SetUp() override {
        ov::element::Type prc = ov::element::undefined;
        std::tie(prc, targetDevice, m_export_model_name, exportConfiguration, importConfiguration) = this->GetParam();
        ov::Shape input_shape{1, 80};
        ov::Shape conv_shape{1, 2, 1, 40};
        ov::Shape split_shape = {input_shape[0], 2 * input_shape[1]};
        ov::ParameterVector inputs = {std::make_shared<Parameter>(prc, split_shape),
                                      std::make_shared<Parameter>(prc, input_shape)};

        // split layer to split inputs and transpose the part connected to convolution only
        auto axis_const = std::make_shared<Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{1});
        auto split = std::make_shared<Split>(inputs[0], axis_const, 2);

        std::vector<int32_t> reshape_pattern{1, 2, 1, -1};
        auto reshape_const = std::make_shared<Constant>(ov::element::i32, ov::Shape{reshape_pattern.size()}, reshape_pattern);

        auto split_0_reshape = std::make_shared<Reshape>(split->output(0), reshape_const, true);
        auto split_1_reshape = std::make_shared<Reshape>(split->output(1), reshape_const, true);
        auto input_1_reshape = std::make_shared<Reshape>(inputs[1], reshape_const, true);

        auto add = std::make_shared<Add>(split_0_reshape, input_1_reshape);
        auto relu_1 = std::make_shared<Relu>(add);

        // Convolution to test nchw->nhwc
        size_t num_out_channels = 8;
        size_t kernel_size = 8;
        std::vector<float> filter_weights =
            CommonTestUtils::generate_float_numbers(num_out_channels * reshape_pattern[1] * kernel_size, -0.1f, 0.1f);
        auto conv = ngraph::builder::makeConvolution(relu_1,
                                                     prc,
                                                     {1, kernel_size},
                                                     {1, 1},
                                                     {0, 0},
                                                     {0, 0},
                                                     {1, 1},
                                                     ngraph::op::PadType::VALID,
                                                     num_out_channels,
                                                     true,
                                                     filter_weights);

        auto relu_2 = std::make_shared<Relu>(conv);

        // Memory layers
        ov::op::util::VariableInfo vi{};
        vi.data_shape = ov::PartialShape(conv_shape);
        vi.variable_id = "test_variable";
        vi.data_type = prc;
        const auto var = std::make_shared<ov::op::util::Variable>(vi);
        std::vector<float> initial_state = CommonTestUtils::generate_float_numbers(ov::shape_size(conv_shape), -3.f, 3.f);
        auto initial_state_node = std::make_shared<Constant>(prc, conv_shape, initial_state);
        auto read = std::make_shared<ReadValue>(initial_state_node, var);
        auto mul = std::make_shared<Multiply>(split_1_reshape, read);
        auto assign = std::make_shared<Assign>(mul, var);
        auto relu_3 = std::make_shared<Relu>(mul);

        ov::SinkVector sinks = {assign};
        ov::ResultVector results;
        results.emplace_back(std::make_shared<Result>(relu_2));
        results.emplace_back(std::make_shared<Result>(relu_3));

        function = std::make_shared<ov::Model>(results, sinks, inputs, "universal_export_model");
    }

    std::map<std::string, std::string> exportConfiguration;
    std::map<std::string, std::string> importConfiguration;
    std::string m_export_model_name;
};

class BackwardCompatibilityLegacy : public BackwardCompatibility {
protected:
    void SetUp() override {
        ov::element::Type prc = ov::element::undefined;
        std::tie(prc, targetDevice, m_export_model_name, exportConfiguration, importConfiguration) = this->GetParam();
        ov::Shape input_shape{1, 336};

        auto param = std::make_shared<Parameter>(prc, input_shape);
        auto const_eltwise = std::make_shared<Constant>(prc, input_shape, std::vector<float>{-1});
        auto mul = std::make_shared<Multiply>(param, const_eltwise);

        ov::ResultVector results{std::make_shared<Result>(mul)};

        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "ExportBackwordCompatibility");
    }
};


TEST_P(BackwardCompatibility, smoke_BackwardCompatibility){
    Run();
}

TEST_P(BackwardCompatibilityLegacy, smoke_BackwardCompatibility){
    Run();
}

const std::vector<ov::element::Type> input_precisions = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<std::map<std::string, std::string>> export_configs_legacy = {
        {
                {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                {"GNA_SCALE_FACTOR_0", "327.67"}
        }
};

const std::vector<std::map<std::string, std::string>> import_configs_legacy = {
        {
                {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                {"GNA_SCALE_FACTOR_0", "327.67"}
        },
};

const std::vector<std::map<std::string, std::string>> export_configs = {
        {
                {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                {"GNA_SCALE_FACTOR_0", "327.67"},
                {"GNA_SCALE_FACTOR_1", "327.67"}
        }
};

const std::vector<std::map<std::string, std::string>> import_configs = {
        {
                {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                {"GNA_SCALE_FACTOR_0", "327.67"},
                {"GNA_SCALE_FACTOR_1", "327.67"}
        }
};

const std::vector<std::string> export_models_legacy = {"export2dot1.blob", "export2dot2.blob", "export2dot3.blob", "export2dot4.blob", "export2dot5.blob"};
const std::vector<std::string> export_models = {"export2dot6.blob", "export2dot7.blob", "export2dot8.blob", "export2dot9.blob"};

INSTANTIATE_TEST_SUITE_P(smoke_OldVersion, BackwardCompatibilityLegacy,
                        ::testing::Combine(
                                ::testing::ValuesIn(input_precisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(export_models_legacy),
                                ::testing::ValuesIn(export_configs_legacy),
                                ::testing::ValuesIn(import_configs_legacy)),
                        BackwardCompatibilityLegacy::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OldVersion, BackwardCompatibility,
                        ::testing::Combine(
                                ::testing::ValuesIn(input_precisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(export_models),
                                ::testing::ValuesIn(export_configs),
                                ::testing::ValuesIn(import_configs)),
                        BackwardCompatibility::getTestCaseName);
