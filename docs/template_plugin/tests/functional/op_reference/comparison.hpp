// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>

#include "base_reference_test.hpp"
#include "ngraph_functions/builders.hpp"

namespace ComparisonOpsRefTestDefinitions {
struct RefComparisonParams {
    template <class IT, class OT>
    RefComparisonParams(const ngraph::helpers::ComparisonTypes comp, const ngraph::PartialShape& input_shape1, const ngraph::PartialShape& input_shape2,
                        const ngraph::element::Type& iType, const ngraph::element::Type& oType, const std::vector<IT>& iValues1,
                        const std::vector<IT>& iValues2, const std::vector<OT>& oValues)
        : comparisonType(comp),
          pshape1(input_shape1),
          pshape2(input_shape2),
          inType(iType),
          outType(oType),
          inputData1(CreateBlob(iType, iValues1)),
          inputData2(CreateBlob(iType, iValues2)),
          refData(CreateBlob(oType, oValues)) {}
    ngraph::helpers::ComparisonTypes comparisonType;
    ngraph::PartialShape pshape1;
    ngraph::PartialShape pshape2;
    ngraph::element::Type inType;
    ngraph::element::Type outType;
    InferenceEngine::Blob::Ptr inputData1;
    InferenceEngine::Blob::Ptr inputData2;
    InferenceEngine::Blob::Ptr refData;
};

class ReferenceComparisonLayerTest : public testing::TestWithParam<RefComparisonParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.comparisonType, params.pshape1, params.pshape2, params.inType, params.outType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RefComparisonParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "comparisonType=" << param.comparisonType << "_";
        result << "inpt_shape1=" << param.pshape1 << "_";
        result << "inpt_shape2=" << param.pshape2 << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<ngraph::Function> CreateFunction(ngraph::helpers::ComparisonTypes comp_op_type, ngraph::PartialShape& input_shape1,
                                                            const ngraph::PartialShape& input_shape2, const ngraph::element::Type& input_type,
                                                            const ngraph::element::Type& expected_output_type) {
        const auto in = std::make_shared<ngraph::op::Parameter>(input_type, input_shape1);
        const auto in2 = std::make_shared<ngraph::op::Parameter>(input_type, input_shape2);
        const auto comp = ngraph::builder::makeComparison(in, in2, comp_op_type);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector {comp}, ngraph::ParameterVector {in, in2});
    }
};
}  // namespace ComparisonOpsRefTestDefinitions
