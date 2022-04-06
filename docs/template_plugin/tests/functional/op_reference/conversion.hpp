// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ov;

namespace reference_tests {
namespace ConversionOpsRefTestDefinitions {

static std::map<ngraph::helpers::ConversionTypes, std::string> conversionNames = {
    {ngraph::helpers::ConversionTypes::CONVERT,      "Convert"},
    {ngraph::helpers::ConversionTypes::CONVERT_LIKE, "ConvertLike"}
};

struct ConvertParams {
    template <class IT, class OT>
    ConvertParams(ngraph::helpers::ConversionTypes convType, const ov::PartialShape& shape, const ov::element::Type& iType,
                  const ov::element::Type& oType, const std::vector<IT>& iValues, const std::vector<OT>& oValues, size_t iSize = 0, size_t oSize = 0)
        : conversionType(convType), pshape(shape), inType(iType), outType(oType), inputData(CreateTensor(iType, iValues, iSize)),
          refData(CreateTensor(oType, oValues, oSize)) {}
    ngraph::helpers::ConversionTypes conversionType;
    ov::PartialShape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceConversionLayerTest : public testing::TestWithParam<ConvertParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType, params.conversionType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConvertParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "conversionType=" << conversionNames[param.conversionType] << "_";
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const ov::PartialShape& input_shape, const ov::element::Type& input_type,
                                                        const ov::element::Type& expected_output_type,
                                                        const ngraph::helpers::ConversionTypes& conversion_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto convert = ngraph::builder::makeConversion(in, expected_output_type, conversion_type);
        return std::make_shared<ov::Model>(ov::NodeVector {convert}, ov::ParameterVector {in});
    }
};
} // namespace ConversionOpsRefTestDefinitions
} // namespace reference_tests
