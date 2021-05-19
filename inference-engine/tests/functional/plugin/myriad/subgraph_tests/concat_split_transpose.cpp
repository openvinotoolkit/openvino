// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/layer_test_utils.hpp>
#include "vpu/private_plugin_config.hpp"

#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/myriad_plugin_config.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = std::vector<std::vector<std::size_t>>;

using Parameters = std::tuple<
        DataType,
        DataDims,
        std::int64_t,
        std::vector<std::size_t>,
        LayerTestsUtils::TargetDevice>;

class Concat_Split_Transpose : public testing::WithParamInterface<Parameters>, virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        configuration[InferenceEngine::MYRIAD_DISABLE_CONVERT_STAGES] = CONFIG_VALUE(YES);
        configuration[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);

        const auto& dataType = std::get<0>(GetParam());
        const auto& dataDims = std::get<1>(GetParam());
        const auto& axis = std::get<2>(GetParam());
        const auto& length = std::get<3>(GetParam());
        targetDevice = std::get<4>(GetParam());

        auto params = ngraph::builder::makeParams(dataType, dataDims);
        auto paramOuts = ngraph::helpers::convert2OutputVector(
                ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

        auto concat = std::make_shared<ngraph::opset1::Concat>(paramOuts, axis);

        const auto lengthData = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64,
                                                                            ngraph::Shape{length.size()},
                                                                            length);
        const auto axisData = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64,
                                                                          ngraph::Shape{1},
                                                                          axis);
        auto split = std::make_shared<ngraph::opset3::VariadicSplit>(concat, axisData, lengthData);

        auto permutation = std::vector<std::int64_t>(split->get_output_shape(0).size());
        std::iota(permutation.rbegin(), permutation.rend(), 0);
        const auto transposition = std::make_shared<ngraph::opset3::Constant>(ngraph::element::i64,
                                                                              ngraph::Shape{split->get_output_shape(0).size()},
                                                                              permutation);

        ngraph::ResultVector results;
        for (int i = 0; i < 2; i++) {
            const auto transpose = std::make_shared<ngraph::opset3::Transpose>(split->output(i), transposition);
            results.push_back(std::make_shared<ngraph::opset1::Result>(transpose));
        }
        function = std::make_shared<ngraph::Function>(results, params, "concat-split-transpose");
    }
};

TEST_P(Concat_Split_Transpose, CompareWithRefs) {
    Run();
}

std::vector<DataDims> dims = {
        {{400, 1}, {600, 1}}
};

std::vector<std::vector<std::size_t>> length = {
        {500, 500}
};

INSTANTIATE_TEST_CASE_P(SpecialStages, Concat_Split_Transpose,
                        ::testing::Combine(
                                ::testing::Values(ngraph::element::i32),
                                ::testing::ValuesIn(dims),
                                ::testing::Values(0),
                                ::testing::ValuesIn(length),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
