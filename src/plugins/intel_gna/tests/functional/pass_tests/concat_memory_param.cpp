// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/pass/convert_prc.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

typedef std::tuple<InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   std::vector<size_t>                  // shape to split
                   >
    concat_memory_test_params;

namespace LayerTestsDefinitions {

class ConcatMemoryTest : public testing::WithParamInterface<concat_memory_test_params>,
                         public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<concat_memory_test_params> obj) {
        InferenceEngine::Precision net_prc;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::vector<size_t> input_shape;
        std::tie(net_prc, targetDevice, configuration, input_shape) = obj.param;

        std::ostringstream result;
        result << "net_prc=" << net_prc.name() << "_";
        result << "device=" << targetDevice << "_";
        for (auto const& config_item : configuration) {
            result << "_config_item=" << config_item.first << "_" << config_item.second;
        }
        result << "_input_shape=" << ov::test::utils::vec2str(input_shape);
        return result.str();
    }

protected:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* raw_blob_data_ptr = blob->buffer().as<float*>();
        std::vector<float> values = ov::test::utils::generate_float_numbers(blob->size(), -2.f, 2.f);
        for (size_t i = 0; i < blob->size(); i++) {
            raw_blob_data_ptr[i] = values[i];
        }
        return blob;
    }

    void SetUp() override {
        InferenceEngine::Precision net_prc;
        std::vector<size_t> input_shape;
        std::tie(net_prc, targetDevice, configuration, input_shape) = this->GetParam();

        auto ng_prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(net_prc);

        size_t in_total_dims_size = ov::shape_size(input_shape);
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ng_prc, ov::Shape{1, in_total_dims_size})};
        auto reshape_pattern =
            std::make_shared<ngraph::opset9::Constant>(ov::element::Type_t::i64, ov::Shape{2}, input_shape);
        auto reshape = std::make_shared<ngraph::opset9::Reshape>(params[0], reshape_pattern, false);

        ov::op::util::VariableInfo vi{};
        vi.data_shape = ov::PartialShape(input_shape);
        vi.variable_id = "test_variable";
        vi.data_type = ov::element::Type_t::f32;
        const auto var = std::make_shared<ov::op::util::Variable>(vi);
        std::vector<float> initial_state = ov::test::utils::generate_float_numbers(in_total_dims_size, -3.f, 3.f);
        auto initial_state_node = ngraph::builder::makeConstant(ov::element::Type_t::f32, input_shape, initial_state);
        auto readValue = std::make_shared<ngraph::opset9::ReadValue>(initial_state_node, var);

        const int axis = 1;
        ov::OutputVector to_concat{readValue, reshape};
        auto concat = ngraph::builder::makeConcat(to_concat, axis);

        const auto concat_shape = concat->get_output_shape(0);
        const auto concat_shape_size = ov::shape_size(concat_shape);

        auto etlwise_data = ov::test::utils::generate_float_numbers(concat_shape_size, -1.f, 1.f);
        auto etlwise_node = ngraph::builder::makeConstant(ov::element::Type_t::f32, concat_shape, etlwise_data);
        auto etlwise_result_node = std::make_shared<ngraph::opset9::Multiply>(concat, etlwise_node);

        ov::ResultVector results{std::make_shared<ngraph::opset9::Result>(etlwise_result_node)};
        auto split_node = ngraph::builder::makeSplit(concat, ng_prc, 2, axis);

        auto assign_node = std::make_shared<ngraph::opset9::Assign>(split_node->output(1), var);
        ngraph::SinkVector sinks{assign_node};
        function = std::make_shared<ov::Model>(results, sinks, params);
    }
};

TEST_P(ConcatMemoryTest, CompareWithRefImpl) {
    Run();
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}},
                                                                 {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

const std::vector<std::vector<size_t>> shapes{
    {1, 64},
};

INSTANTIATE_TEST_SUITE_P(smoke_concat_memory,
                         ConcatMemoryTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(shapes)),
                         ConcatMemoryTest::getTestCaseName);

}  // namespace LayerTestsDefinitions