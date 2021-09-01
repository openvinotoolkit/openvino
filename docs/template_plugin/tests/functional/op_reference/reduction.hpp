// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <vector>

#include "base_reference_test.hpp"
#include "ngraph_functions/builders.hpp"

namespace reference_tests {
namespace ReductionOpsRefTestDefinitions {

struct ReductionParams {
    ReductionParams(const ngraph::helpers::ReductionType& reductType, const bool keepDims, const std::vector<int64_t>& axes,
        const Tensor& dataTensor, const Tensor& outputTensor) : reductionType(reductType), keepDimensions(keepDims), reductionAxes(axes),
        data(dataTensor), output(outputTensor) {}

    ngraph::helpers::ReductionType reductionType;
    bool keepDimensions;
    std::vector<int64_t> reductionAxes;
    Tensor data;
    Tensor output;
};

class ReferenceReductionLayerTest : public  testing::TestWithParam<ReductionParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.data.data};
        refOutData = {params.output.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ReductionParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "reductionType=" << param.reductionType << "_";
        result << "dataType=" << param.data.type << "_";
        result << "dataShape=" << param.data.shape << "_";
        result << "axes=" << CommonTestUtils::vec2str(param.reductionAxes);
        if (param.keepDimensions) {
            result << "_keepDims";
        }
        return result.str();
    }

private:
    static std::shared_ptr<ngraph::Function> CreateFunction(const ReductionParams& params) {
        const auto data = std::make_shared<ngraph::op::Parameter>(params.data.type, params.data.shape);
        const auto axes = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                 ngraph::Shape{params.reductionAxes.size()},
                                                                 params.reductionAxes);
        const auto reduction = ngraph::builder::makeReduce(data, axes, params.keepDimensions, params.reductionType);
        return std::make_shared<ngraph::Function>(reduction, ngraph::ParameterVector{data});
    }
};
} // namespace ReductionOpsRefTestDefinitions
} // namespace reference_tests
