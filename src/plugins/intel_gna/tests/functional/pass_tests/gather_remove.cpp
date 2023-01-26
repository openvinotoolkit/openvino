// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// EMUTEX FIXME: cleanup header list
#include <algorithm>
#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

typedef std::tuple<
    InferenceEngine::Precision,         // Network Precision
    std::string,                        // Target Device
    std::map<std::string, std::string>  // Configuration
> GatherRemoveConvsParams;

namespace {

std::vector<size_t> GenerateVector(size_t size, size_t initial_value) {
    std::vector<size_t> vec(size);
    std::iota(vec.begin(), vec.end(), initial_value);
    return vec;
}

std::vector<size_t> MakeGatherIndexes(size_t size) {
    std::vector<size_t> indexes = GenerateVector(size, 0);
    std::reverse(indexes.begin(), indexes.end());
    return indexes;
}

} // namespace

namespace LayerTestsDefinitions {

class RemoveInputGather : public testing::WithParamInterface<GatherRemoveConvsParams>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherRemoveConvsParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(netPrecision, targetDevice, configuration) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = CommonTestUtils::generate_float_numbers(blob->size(), -0.2f, 0.2f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        const std::vector<size_t> input_shape = {1, 128};
        const size_t input_shape_product = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

        auto input_params = ngraph::builder::makeParams(ngPrc, { input_shape });

        const std::vector<size_t> indexes = MakeGatherIndexes(input_shape_product);
        auto gather_indexes_node = ngraph::opset9::Constant::create(ngraph::element::i64, ov::Shape{indexes.size()}, indexes);
        const size_t gather_axis = 1;
        auto gather_axis_node = ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{}, {gather_axis});
        auto gather_node = std::make_shared<ngraph::opset9::Gather>(input_params[0],
                                                                gather_indexes_node,
                                                                gather_axis_node);

        auto multiply_input_const_node = ngraph::opset9::Constant::create(ngPrc, input_shape, GenerateVector(input_shape_product, 1));

        auto matmul_node = std::make_shared<ngraph::opset9::Multiply>(gather_node,
                                                                  multiply_input_const_node);

        auto add_input_const_node = ngraph::opset9::Constant::create(ngPrc, input_shape, GenerateVector(input_shape_product, 1));

        auto add_node = std::make_shared<ngraph::opset9::Add>(matmul_node,
                                                                  add_input_const_node);

        auto result = std::make_shared<ngraph::opset9::Result>(add_node);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});
    }
};

class RemoveOutputGather : public testing::WithParamInterface<GatherRemoveConvsParams>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GatherRemoveConvsParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::tie(netPrecision, targetDevice, configuration) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        InferenceEngine::Blob::Ptr blob = make_blob_with_precision(info.getTensorDesc());
        blob->allocate();

        auto* rawBlobDataPtr = blob->buffer().as<float*>();
        std::vector<float> values = CommonTestUtils::generate_float_numbers(blob->size(), -0.2f, 0.2f);
        for (size_t i = 0; i < blob->size(); i++) {
            rawBlobDataPtr[i] = values[i];
        }
        return blob;
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::tie(netPrecision, targetDevice, configuration) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        const std::vector<size_t> input_shape = {1, 128};
        const size_t input_shape_product = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());

        auto input_params = ngraph::builder::makeParams(ngPrc, { input_shape });

        auto abs = std::make_shared<ngraph::opset9::Abs>(input_params[0]);

        const std::vector<size_t> indexes = MakeGatherIndexes(input_shape_product);
        auto gather_indexes_node = ngraph::opset9::Constant::create(ngraph::element::i64, ov::Shape{indexes.size()}, indexes);
        const size_t gather_axis = 1;
        auto gather_axis_node = ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{}, {gather_axis});
        auto gather_node = std::make_shared<ngraph::opset9::Gather>(abs,
                                                                gather_indexes_node,
                                                                gather_axis_node);

        auto result = std::make_shared<ngraph::opset9::Result>(gather_node);
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});
    }
};


TEST_P(RemoveInputGather, CompareWithRefs) {
    Run();
}

TEST_P(RemoveOutputGather, CompareWithRefs) {
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<std::map<std::string, std::string>> configs = {
    {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}
    },
    {
        {"GNA_DEVICE_MODE", "GNA_SW_FP32"}
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_gather_on_cpu, RemoveInputGather,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs)),
    RemoveInputGather::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_gather_on_cpu, RemoveOutputGather,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs)),
    RemoveOutputGather::getTestCaseName);


} // namespace LayerTestsDefinitions
