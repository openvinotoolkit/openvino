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
namespace ConvertionOpsRefTestDefinitions {

static std::map<ngraph::helpers::ConvertionTypes, std::string> convertionNames = {
    {ngraph::helpers::ConvertionTypes::CONVERT,      "Convert"},
    {ngraph::helpers::ConvertionTypes::CONVERT_LIKE, "ConvertLike"}
};

struct ConvertParams {
    template <class IT, class OT>
    ConvertParams(ngraph::helpers::ConvertionTypes convType, const ngraph::PartialShape& shape, const ngraph::element::Type& iType,
                  const ngraph::element::Type& oType, const std::vector<IT>& iValues, const std::vector<OT>& oValues, size_t iSize = 0, size_t oSize = 0)
        : convertionType(convType), pshape(shape), inType(iType), outType(oType), inputData(CreateBlob(iType, iValues, iSize)),
          refData(CreateBlob(oType, oValues, oSize)) {}
    ngraph::helpers::ConvertionTypes convertionType;
    ngraph::PartialShape pshape;
    ngraph::element::Type inType;
    ngraph::element::Type outType;
    InferenceEngine::Blob::Ptr inputData;
    InferenceEngine::Blob::Ptr refData;
};

class ReferenceConvertionLayerTest : public testing::TestWithParam<ConvertParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType, params.convertionType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConvertParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "convertionType=" << convertionNames[param.convertionType] << "_";
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<ngraph::Function> CreateFunction(const ngraph::PartialShape& input_shape, const ngraph::element::Type& input_type,
                                                            const ngraph::element::Type& expected_output_type,
                                                            const ngraph::helpers::ConvertionTypes& convertion_type) {
        const auto in = std::make_shared<ngraph::op::Parameter>(input_type, input_shape);
        const auto convert = ngraph::builder::makeConvertion(in, expected_output_type, convertion_type);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector {convert}, ngraph::ParameterVector {in});
    }
};

namespace {
TEST_P(ReferenceConvertionLayerTest, CompareWithHardcodedRefs) {
    Exec();
}
} // namespace
} // namespace ConvertionOpsRefTestDefinitions
} // namespace reference_tests
