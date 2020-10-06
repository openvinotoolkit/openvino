// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <functional>

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"
#include "subgraph_tests/multiple_concat.hpp"

namespace SubgraphTestsDefinitions {

std::string MultipleConcatTest::getTestCaseName(const testing::TestParamInfo<multipleConcatParams> &obj) {
    std::string targetDevice;
    InferenceEngine::Precision netPrecision;
    size_t inputSize;
    size_t constantSize;
    std::map<std::string, std::string> config;
    std::tie(targetDevice, netPrecision, inputSize, constantSize, config) = obj.param;
    std::ostringstream result;

    result << "netPrecision=" << netPrecision.name() << "_";
    result << "IS=" << inputSize << "_";
    result << "CS=" << constantSize << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MultipleConcatTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> config;
    size_t inputSize;
    size_t constantSize;
    std::tie(targetDevice, netPrecision, inputSize, constantSize, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::vector<size_t> input_dims { 1, inputSize };
    std::vector<size_t> constant_dims {1, constantSize};

    const int seed = 0;
    std::mt19937 gen(static_cast<float>(seed));

    auto generateFloatNumbers = [gen](std::size_t vec_len, float min, float max) mutable {
        std::vector<float> res;

        std::uniform_real_distribution<float> dist(min, max);
        for (int i = 0; i < vec_len; i++)
            res.emplace_back(static_cast<float>(dist(gen)));

        return res;
    };

    auto concat_1_vals = generateFloatNumbers(constantSize, -2.0f, 2.0f);
    auto concat_2_vals = generateFloatNumbers(constantSize, -5.0f, 5.0f);

    auto input_parameter = ngraph::builder::makeParams(ngPrc, {input_dims});

    auto const_1 = ngraph::builder::makeConstant(ngPrc, constant_dims, concat_1_vals);
    auto concat_1 = ngraph::builder::makeConcat({const_1, input_parameter[0]}, 1);

    auto const_2 = ngraph::builder::makeConstant(ngPrc, constant_dims, concat_1_vals);
    auto concat_2 = ngraph::builder::makeConcat({concat_1, const_2}, 1);

    auto act = ngraph::builder::makeActivation(concat_2, ngPrc, ngraph::helpers::ActivationTypes::Relu);

    function = std::make_shared<ngraph::Function>(act, input_parameter, "multiple_concat");
}

TEST_P(MultipleConcatTest, CompareWithRefs) {
    Run();
};
}  // namespace SubgraphTestsDefinitions
