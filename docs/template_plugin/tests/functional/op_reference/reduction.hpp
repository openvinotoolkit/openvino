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
    template <class ET>
    ReductionParams(const ngraph::PartialShape& inShape, const ngraph::element::Type& inType, const std::vector<ET> inValues,
                    const std::vector<int64_t> axesValues, const std::vector<ET> outValues, const bool keepDims,
                    const ngraph::helpers::ReductionType& reductType)
            : dataShape(inShape), elemType(inType), inputData(CreateBlob(inType, inValues)), axesData(axesValues), refData(CreateBlob(inType, outValues)),
              keepDimensions(keepDims), reductionType(reductType) {}
    ngraph::PartialShape dataShape;
    ngraph::element::Type elemType;
    InferenceEngine::Blob::Ptr inputData;
    std::vector<int64_t> axesData;
    InferenceEngine::Blob::Ptr refData;
    bool keepDimensions;
    ngraph::helpers::ReductionType reductionType;
};

class ReferenceReductionLayerTest : public  testing::TestWithParam<ReductionParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.dataShape, params.elemType, params.axesData, params.keepDimensions, params.reductionType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ReductionParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "reductionType=" << param.reductionType << "_";
        result << "dataShape=" << param.dataShape << "_";
        result << "dataType=" << param.elemType << "_";
        result << "axes=" << CommonTestUtils::vec2str(param.axesData);
        if (param.keepDimensions) {
            result << "_keepDims";
        }
        return result.str();
    }

private:
    static std::shared_ptr<ngraph::Function> CreateFunction(const ngraph::PartialShape& data_shape,
                                                            const ngraph::element::Type& elem_type,
                                                            const std::vector<int64_t>& axes_values, const bool keep_dims,
                                                            const ngraph::helpers::ReductionType& reduction_type) {
        const auto data = std::make_shared<ngraph::op::Parameter>(elem_type, data_shape);
        const auto axes = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{axes_values.size()}, axes_values);
        const auto reduction = ngraph::builder::makeReduce(data, axes, keep_dims, reduction_type);
        return std::make_shared<ngraph::Function>(reduction, ngraph::ParameterVector{data});
    }
};
} // namespace ReductionOpsRefTestDefinitions
} // namespace reference_tests
