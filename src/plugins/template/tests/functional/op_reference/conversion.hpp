// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "common_test_utils/test_enums.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"

using namespace ov;

namespace reference_tests {
namespace ConversionOpsRefTestDefinitions {

using ov::test::utils::ConversionTypes;

static std::map<ConversionTypes, std::string> conversionNames = {{ConversionTypes::CONVERT, "Convert"},
                                                                 {ConversionTypes::CONVERT_LIKE, "ConvertLike"}};

struct ConvertParams {
    template <class IT, class OT>
    ConvertParams(ConversionTypes convType,
                  const ov::PartialShape& shape,
                  const ov::element::Type& iType,
                  const ov::element::Type& oType,
                  const std::vector<IT>& iValues,
                  const std::vector<OT>& oValues)
        : conversionType(convType),
          pshape(shape),
          inType(iType),
          outType(oType),
          inputData(CreateTensor(shape.get_shape(), iType, iValues)),
          refData(CreateTensor(shape.get_shape(), oType, oValues)) {}
    ConversionTypes conversionType;
    ov::PartialShape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceConversionLayerTest : public testing::TestWithParam<ConvertParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        legacy_compare = true;
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
    static std::shared_ptr<ov::Model> CreateFunction(const ov::PartialShape& input_shape,
                                                     const ov::element::Type& input_type,
                                                     const ov::element::Type& expected_output_type,
                                                     const ConversionTypes& conversion_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        std::shared_ptr<ov::Node> convert;
        if (conversion_type == ConversionTypes::CONVERT) {
            convert = std::make_shared<ov::op::v0::Convert>(in, expected_output_type);
        } else if (conversion_type == ConversionTypes::CONVERT_LIKE) {
            const auto like = std::make_shared<ov::op::v0::Constant>(expected_output_type, ov::Shape{1});
            convert = std::make_shared<ov::op::v1::ConvertLike>(in, like);
        } else {
            throw std::runtime_error("Incorrect type of Conversion operation");
        }
        return std::make_shared<ov::Model>(ov::NodeVector{convert}, ov::ParameterVector{in});
    }
};
}  // namespace ConversionOpsRefTestDefinitions
}  // namespace reference_tests
