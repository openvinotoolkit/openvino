// Copyright (C) 2018-2023 Intel Corporation
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
#include "openvino/opsets/opset10.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/pass/convert_prc.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

using namespace ov::opset10;

enum NonFunctionalLayer { RESHAPE, SQUEEZE, UNSQUEEZE, TRANSPOSE, GATHER, NONE };

namespace std {
inline std::ostream& operator<<(std::ostream& os, NonFunctionalLayer layer_type) {
    switch (layer_type) {
    case NonFunctionalLayer::RESHAPE:
        os << "RESHAPE";
        break;
    case NonFunctionalLayer::SQUEEZE:
        os << "SQUEEZE";
        break;
    case NonFunctionalLayer::UNSQUEEZE:
        os << "UNSQUEEZE";
        break;
    case NonFunctionalLayer::TRANSPOSE:
        os << "TRANSPOSE";
        break;
    case NonFunctionalLayer::GATHER:
        os << "GATHER";
        break;
    default:
        os << "NONE";
        break;
    }
    return os;
}
}  // namespace std

typedef std::tuple<InferenceEngine::Precision,          // Network Precision
                   std::string,                         // Target Device
                   std::map<std::string, std::string>,  // Configuration
                   std::pair<float, float>,             // Input values
                   NonFunctionalLayer                   // Layer between Input and FQ
                   >
    fqScaleFactorParams;

namespace LayerTestsDefinitions {

class TestFQScaleFactorsTest : public testing::WithParamInterface<fqScaleFactorParams>,
                               public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<fqScaleFactorParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        std::pair<float, float> inputValues;
        NonFunctionalLayer non_func_layer;
        std::tie(netPrecision, targetDevice, configuration, inputValues, non_func_layer) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_range=(" << inputValues.first << ", " << inputValues.second << ")";
        result << "layer=" << non_func_layer;

        return result.str();
    }

protected:
    void SetUp() override {
        InferenceEngine::Precision netPrecision;
        std::pair<float, float> inputValues;

        std::tie(netPrecision, targetDevice, configuration, inputValues, m_non_func_layer) = this->GetParam();
        std::tie(inputDataMin, inputDataMax) = inputValues;
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        const ngraph::Shape shape = {1, 1, 128};
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shape))};
        std::shared_ptr<ov::Node> test_node = params[0];
        switch (m_non_func_layer) {
        case NonFunctionalLayer::RESHAPE:
            test_node = AddReshapeNode(test_node);
            break;
        case NonFunctionalLayer::SQUEEZE:
            test_node = AddSqueezeNode(test_node);
            break;
        case NonFunctionalLayer::UNSQUEEZE:
            test_node = AddUnSqueezeNode(test_node);
            break;
        case NonFunctionalLayer::TRANSPOSE:
            test_node = AddTransposeNode(test_node);
            break;
        case NonFunctionalLayer::GATHER:
            test_node = AddGatherNode(test_node, shape);
            break;
        default:
            break;
        }

        auto lowNodeIn = ngraph::builder::makeConstant<float>(ngPrc, {1}, {inputDataMin});
        auto highNodeIn = ngraph::builder::makeConstant<float>(ngPrc, {1}, {inputDataMax});
        auto fqIn = std::make_shared<ngraph::opset8::FakeQuantize>(test_node,
                                                                   lowNodeIn,
                                                                   highNodeIn,
                                                                   lowNodeIn,
                                                                   highNodeIn,
                                                                   levels);

        auto mul = std::make_shared<ngraph::opset8::Multiply>(fqIn, test_node);

        auto lowNodeOut = ngraph::builder::makeConstant<float>(ngPrc, {1}, {-inputDataMin * inputDataMin});
        auto highNodeOut = ngraph::builder::makeConstant<float>(ngPrc, {1}, {inputDataMax * inputDataMax});
        auto fqOut = std::make_shared<ngraph::opset8::FakeQuantize>(mul,
                                                                    lowNodeOut,
                                                                    highNodeOut,
                                                                    lowNodeOut,
                                                                    highNodeOut,
                                                                    levels);

        ngraph::ResultVector results{std::make_shared<ngraph::opset8::Result>(fqOut)};
        function = std::make_shared<ngraph::Function>(results, params, "FQWithSmallScaleFactor");
        functionRefs = ngraph::clone_function(*function);
    }

    std::shared_ptr<ov::Node> AddReshapeNode(std::shared_ptr<ov::Node> node) {
        const ov::Shape reshape_pattern = {1, 128};
        auto reshape_const =
            std::make_shared<Constant>(ov::element::i32, ov::Shape{reshape_pattern.size()}, reshape_pattern);
        auto reshape = std::make_shared<Reshape>(node, reshape_const, false);
        return reshape;
    }

    std::shared_ptr<ov::Node> AddSqueezeNode(std::shared_ptr<ov::Node> node) {
        const ov::Shape sq_dims = {0, 1};
        auto sq_axes = std::make_shared<Constant>(ov::element::i32, ov::Shape{sq_dims.size()}, sq_dims);
        auto sq_node = std::make_shared<Squeeze>(node, sq_axes);
        return sq_node;
    }

    std::shared_ptr<ov::Node> AddUnSqueezeNode(std::shared_ptr<ov::Node> node) {
        const ov::Shape unsq_dims = {0};
        auto unsq_axes = std::make_shared<Constant>(ov::element::i32, ov::Shape{unsq_dims.size()}, unsq_dims);
        auto unsq_node = std::make_shared<Unsqueeze>(node, unsq_axes);
        return unsq_node;
    }

    std::shared_ptr<ov::Node> AddTransposeNode(std::shared_ptr<ov::Node> node) {
        const ov::Shape transpose_axes = {0, 2, 1};
        auto transpose_const =
            std::make_shared<Constant>(ov::element::i32, ov::Shape{transpose_axes.size()}, transpose_axes);
        auto transpose = std::make_shared<Transpose>(node, transpose_const);
        return transpose;
    }

    std::shared_ptr<ov::Node> AddGatherNode(std::shared_ptr<ov::Node> node, ov::Shape shape) {
        ov::Shape gather_indices(shape.size());
        std::iota(gather_indices.begin(), gather_indices.end(), 0);

        auto gather_const =
            std::make_shared<Constant>(ov::element::i32, ov::Shape{gather_indices.size()}, gather_indices);
        auto gather_axis_const = Constant::create(ov::element::i64, ov::Shape{}, {shape.size() - 1});
        auto gather = std::make_shared<Gather>(node, gather_const, gather_axis_const);
        return gather;
    }

    NonFunctionalLayer m_non_func_layer = NonFunctionalLayer::NONE;
    float inputDataMax = 1.0;
    float inputDataMin = -1.0;
    size_t levels = std::numeric_limits<uint16_t>::max();
};

TEST_P(TestFQScaleFactorsTest, CompareWithRefImpl) {
    LoadNetwork();
    GenerateInputs();
    Infer();
    auto refs = CalculateRefs();
    auto results = GetOutputs();
    const auto expected = reinterpret_cast<const float*>(refs.front().second.data());
    size_t size = results.front()->size();
    auto memory = InferenceEngine::as<InferenceEngine::MemoryBlob>(results.front());
    IE_ASSERT(memory);
    const auto lockedMemory = memory->wmap();
    const auto actualBuffer = lockedMemory.as<const float*>();

    /* the absolute threshold is calculated as 1.25 * (1 / last_fq_out_scale_factor) = 1.25 * (2 * maxValue) / (levels -
    1), the most of accuracy degradation in this model is introduced by the output scale factor of FakeQuantize, 1 / sf
    is a part of the value which can be represented by one level, so we can't get more accurate resolution than this
    part, maxValue = inputDataMax * inputDataMax since this model multiplies input values with itself,
    1.25 is a reserve factor to cover other errors in this model */
    abs_threshold = 2.5 * inputDataMax * inputDataMax / (levels - 1);

    for (size_t i = 0; i < size; ++i) {
        const auto& ref = expected[i];
        const auto& res = actualBuffer[i];
        if (ov::test::utils::ie_abs(res - ref) > abs_threshold) {
            IE_THROW() << "Absolute comparison of values expected: " << ref << " and actual: " << res << " at index "
                       << i << " with absolute threshold " << abs_threshold << " failed";
        }
    }
};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
}};

// Need to enable Gather when it is supported by the plugin
const std::vector<NonFunctionalLayer> non_func_layers = {NONE, RESHAPE, SQUEEZE, UNSQUEEZE, TRANSPOSE};

const std::vector<std::pair<float, float>> inputValues = {{-188.0, 188.0}, {-90.0, 90.0}, {-20.0, 20.0}, {-10.0, 10.0}};

INSTANTIATE_TEST_SUITE_P(smoke_base,
                         TestFQScaleFactorsTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputValues),
                                            ::testing::ValuesIn(non_func_layers)),
                         TestFQScaleFactorsTest::getTestCaseName);

}  // namespace LayerTestsDefinitions
