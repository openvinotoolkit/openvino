// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>
#include <tuple>
#include <vector>
#include <string>

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
    std::map<std::string, std::string>, // Configuration
    size_t,                             // Input Shape
    bool                                // first input is const
> insertTransposeBeforeMatmulParams;

namespace LayerTestsDefinitions {

class InsertTransposeBeforeMatmul : public testing::WithParamInterface<insertTransposeBeforeMatmulParams>,
    public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<insertTransposeBeforeMatmulParams> obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetDevice;
        std::map<std::string, std::string> configuration;
        size_t inputShape;
        bool firstInConst;
        std::tie(netPrecision, targetDevice, configuration, inputShape, firstInConst) = obj.param;

        std::ostringstream result;
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice << "_";
        for (auto const& configItem : configuration) {
            result << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        result << "_inputShape=" << inputShape;
        result << "_firstInConst=" << firstInConst;
        return result.str();
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const {
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
        size_t inputShape;
        bool firstInConst;
        std::tie(netPrecision, targetDevice, configuration, inputShape, firstInConst) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        auto params = ngraph::builder::makeParams(ngPrc, {{1, inputShape}});

        auto matmul_in_shape = firstInConst ? ngraph::Shape{inputShape / 8, 8} : ngraph::Shape{8, inputShape / 8};
        auto pattern = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 2 }, matmul_in_shape);
        auto reshape = std::make_shared<ngraph::opset1::Reshape>(params[0], pattern, false);

        std::shared_ptr<ngraph::Node> weights_node;
        if (firstInConst) {
            std::vector<float> weights = CommonTestUtils::generate_float_numbers(matmul_in_shape[0], -0.2f, 0.2f);
            weights_node = std::make_shared<ngraph::opset1::Constant>(ngPrc, ngraph::Shape{ 1, matmul_in_shape[0] }, weights);
        } else {
            std::vector<float> weights = CommonTestUtils::generate_float_numbers(matmul_in_shape[1], -0.2f, 0.2f);
            weights_node = std::make_shared<ngraph::opset1::Constant>(ngPrc, ngraph::Shape{ matmul_in_shape[1], 1 }, weights);
        }

        auto matmul = firstInConst ? ngraph::builder::makeMatMul(weights_node, reshape, false, false) :
                                     ngraph::builder::makeMatMul(reshape, weights_node, false, false);

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(matmul)};
        function = std::make_shared<ngraph::Function>(results, params, "InsertTransposeBeforeMatmul");
    }
};

TEST_P(InsertTransposeBeforeMatmul, CompareWithRefImpl) {
    Run();
};

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

const std::vector<size_t> inputShape = {
    64,
    128,
    192
};

const std::vector<bool> firstInputConst = {false, true};

INSTANTIATE_TEST_SUITE_P(smoke_InsertTransposeBeforeMatmulTest, InsertTransposeBeforeMatmul,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs),
        ::testing::ValuesIn(inputShape),
        ::testing::ValuesIn(firstInputConst)),
    InsertTransposeBeforeMatmul::getTestCaseName);

} // namespace LayerTestsDefinitions
