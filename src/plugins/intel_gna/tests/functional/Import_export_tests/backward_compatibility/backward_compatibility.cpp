// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>

#include <map>
#include <memory>
#include <vector>

#include "helpers/test_model_repo.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type.hpp"
#include "openvino/opsets/opset10.hpp"
#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

using namespace ov::opset10;

typedef std::tuple<ov::element::Type,                   // Network Precision
                   std::string,                         // Target Device
                   std::string,                         // Name Export Model
                   std::map<std::string, std::string>,  // Export Configuration
                   std::map<std::string, std::string>   // Import Configuration
                   >
    exportImportNetworkParams;

class BackwardCompatibility : public testing::WithParamInterface<exportImportNetworkParams>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string get_test_case_name(testing::TestParamInfo<exportImportNetworkParams> obj) {
        ov::element::Type input_prc;
        std::string target_device;
        std::map<std::string, std::string> conf_export;
        std::map<std::string, std::string> conf_import;
        std::string name_export_model;
        std::tie(input_prc, target_device, name_export_model, conf_export, conf_import) = obj.param;

        std::ostringstream result;
        result << "input_prc=" << input_prc << "_";
        result << "target_device=" << target_device << "_";
        result << "name_export_model=" << name_export_model << "_";
        for (auto const& conf_item : conf_export) {
            result << "_exportConfigItem=" << conf_item.first << "_" << conf_item.second;
        }
        for (auto const& conf_item : conf_import) {
            result << "_importConfigItem=" << conf_item.first << "_" << conf_item.second;
        }
        return result.str();
    }

    void Run() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        functionRefs = ngraph::clone_function(*function);
        // load export configuration and save outputs
        configuration.insert(m_conf_export.begin(), m_conf_export.end());
        LoadNetwork();
        GenerateInputs();
        Infer();
        auto outputs = GetOutputs();

        auto output_refs = CalculateRefs();
        Compare(output_refs, outputs);

        for (auto const& config_item : m_conf_import) {
            configuration[config_item.first] = config_item.second;
        }

        const auto compiled_network = executableNetwork;
        auto model = TestDataHelpers::get_data_path() + "/exported_models/" + m_export_model_name;

        const auto imported_network = core->ImportNetwork(model, targetDevice, configuration);

        GenerateInputs();
        Infer();

        ASSERT_EQ(imported_network.GetInputsInfo().size(), compiled_network.GetInputsInfo().size());
        ASSERT_EQ(imported_network.GetOutputsInfo().size(), compiled_network.GetOutputsInfo().size());

        for (const auto& next_output : imported_network.GetOutputsInfo()) {
            ASSERT_NO_THROW(compiled_network.GetOutputsInfo()[next_output.first]);
        }
        auto outputs_imported = GetOutputs();

        ASSERT_EQ(outputs.size(), outputs_imported.size());

        for (size_t i = 0; i < outputs.size(); i++) {
            Compare(outputs[i], outputs_imported[i]);
        }
    }

protected:
    void SetUp() override {
        ov::element::Type prc = ov::element::undefined;
        std::tie(prc, targetDevice, m_export_model_name, m_conf_export, m_conf_import) = this->GetParam();
        ov::Shape input_shape{1, 80};
        ov::Shape conv_shape{1, 2, 1, 40};
        ov::Shape split_shape = {input_shape[0], 2 * input_shape[1]};
        ov::ParameterVector inputs = {std::make_shared<Parameter>(prc, split_shape),
                                      std::make_shared<Parameter>(prc, input_shape)};

        // split layer to split inputs and transpose the part connected to convolution only
        auto axis_const = std::make_shared<Constant>(ov::element::i32, ov::Shape{}, std::vector<int32_t>{1});
        auto split = std::make_shared<Split>(inputs[0], axis_const, 2);

        std::vector<int32_t> reshape_pattern{1, 2, 1, -1};
        auto reshape_const =
            std::make_shared<Constant>(ov::element::i32, ov::Shape{reshape_pattern.size()}, reshape_pattern);

        auto split_0_reshape = std::make_shared<Reshape>(split->output(0), reshape_const, true);
        auto split_1_reshape = std::make_shared<Reshape>(split->output(1), reshape_const, true);
        auto input_1_reshape = std::make_shared<Reshape>(inputs[1], reshape_const, true);

        auto add = std::make_shared<Add>(split_0_reshape, input_1_reshape);
        auto relu_1 = std::make_shared<Relu>(add);

        // Convolution to test nchw->nhwc
        size_t num_out_channels = 8;
        size_t kernel_size = 8;
        std::vector<float> filter_weights =
            ov::test::utils::generate_float_numbers(num_out_channels * reshape_pattern[1] * kernel_size, -0.1f, 0.1f);
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
        std::vector<float> initial_state =
            ov::test::utils::generate_float_numbers(ov::shape_size(conv_shape), -3.f, 3.f);
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

    std::map<std::string, std::string> m_conf_export;
    std::map<std::string, std::string> m_conf_import;
    std::string m_export_model_name;
};

class BackwardCompatibilityLegacy : public BackwardCompatibility {
protected:
    void SetUp() override {
        ov::element::Type prc = ov::element::undefined;
        std::tie(prc, targetDevice, m_export_model_name, m_conf_export, m_conf_import) = this->GetParam();
        ov::Shape input_shape{1, 336};

        auto param = std::make_shared<Parameter>(prc, input_shape);
        auto const_eltwise = std::make_shared<Constant>(prc, input_shape, std::vector<float>{-1});
        auto mul = std::make_shared<Multiply>(param, const_eltwise);

        ov::ResultVector results{std::make_shared<Result>(mul)};

        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param}, "ExportBackwordCompatibility");
    }
};

TEST_P(BackwardCompatibility, BackwardCompatibility) {
    Run();
}

TEST_P(BackwardCompatibilityLegacy, BackwardCompatibility) {
    Run();
}

const std::vector<ov::element::Type> input_precisions = {ov::element::f32, ov::element::f16};

const std::vector<std::map<std::string, std::string>> export_configs_legacy = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "327.67"}}};

const std::vector<std::map<std::string, std::string>> import_configs_legacy = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "327.67"}},
};

const std::vector<std::map<std::string, std::string>> export_configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "327.67"}, {"GNA_SCALE_FACTOR_1", "327.67"}}};

const std::vector<std::map<std::string, std::string>> import_configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "327.67"}, {"GNA_SCALE_FACTOR_1", "327.67"}}};

const std::vector<std::string> export_models_legacy = {"export2dot1.blob",
                                                       "export2dot2.blob",
                                                       "export2dot3.blob",
                                                       "export2dot4.blob",
                                                       "export2dot5.blob"};

const std::vector<std::string> export_models = {"export2dot6.blob", "export2dot7.blob", "export2dot8.blob"};

// Those tests should not be run in CI due to dependency on model blobs
INSTANTIATE_TEST_SUITE_P(OldVersion,
                         BackwardCompatibilityLegacy,
                         ::testing::Combine(::testing::ValuesIn(input_precisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(export_models_legacy),
                                            ::testing::ValuesIn(export_configs_legacy),
                                            ::testing::ValuesIn(import_configs_legacy)),
                         BackwardCompatibilityLegacy::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(OldVersion,
                         BackwardCompatibility,
                         ::testing::Combine(::testing::ValuesIn(input_precisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(export_models),
                                            ::testing::ValuesIn(export_configs),
                                            ::testing::ValuesIn(import_configs)),
                         BackwardCompatibility::get_test_case_name);
