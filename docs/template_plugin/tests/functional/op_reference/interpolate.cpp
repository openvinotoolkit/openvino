// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "base_reference_test.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct InterpolateParams {
    template <class IT>
    InterpolateParams(const Shape& iShape,
                        const Shape& oShape,
                        const element::Type& iType,
                        const element::Type& oType,
                        const std::vector<IT>& iValues,
                        const std::vector<IT>& oValues,
                        const std::vector<size_t> outShapeInput,
                        const element::Type& outShapeInputType,
                        const AxisSet& axis,
                        const std::string& mode,
                        const bool align_corners)
        : inShape(iShape),
          outShape(oShape),
          inType(iType),
          outType(oType),
          inData(CreateTensor(iType, iValues)),
          outData(CreateTensor(oType, oValues)),
          outShapeInput(outShapeInput),
          outShapeInputType(outShapeInputType){
        attrs.axes = axis;
        attrs.mode = mode;
        attrs.align_corners = align_corners;
    }

    Shape inShape;
    Shape outShape;
    element::Type inType;
    element::Type outType;
    runtime::Tensor inData;
    runtime::Tensor outData;
    std::vector<size_t> outShapeInput;
    element::Type outShapeInputType;
    op::v0::Interpolate::Attributes attrs;
};

class ReferenceInterpolateLayerTest : public testing::TestWithParam<InterpolateParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.inShape,
                                  params.outShape,
                                  params.inType,
                                  params.outType,
                                  params.outShapeInput,
                                  params.outShapeInputType,
                                  params.attrs);
        inputData = {params.inData};
        refOutData = {params.outData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<InterpolateParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iShape=" << param.inShape << "_";
        result << "oShape=" << param.outShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const Shape& input_shape,
                                                    const Shape& output_shape,
                                                    const element::Type& input_type,
                                                    const element::Type& output_type,
                                                    const std::vector<size_t> outShapeInput,
                                                    const element::Type& outShapeInputType,
                                                    const op::v0::Interpolate::Attributes& attrs) {
        const auto input = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto output_shape_input = op::v0::Constant::create(outShapeInputType, outShapeInput, output_shape);
       auto interpolate = std::make_shared<op::v0::Interpolate>(input, output_shape_input, attrs);
        return std::make_shared<Function>(NodeVector{interpolate}, ParameterVector{input});
    }
};

TEST_P(ReferenceInterpolateLayerTest, InterpolateWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<InterpolateParams> generateParamsForInterpolate() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<InterpolateParams> params{
        InterpolateParams(ov::Shape{1, 1, 2, 4},
                          ov::Shape{1, 1, 1, 2},
                          IN_ET,
                          IN_ET,
                          std::vector<T>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                          std::vector<T>{1.0f, 2.66666651f},
                          {4},
                          element::i64,
                          AxisSet{0, 1, 2, 3},
                          "linear",
                          false)
    };
    return params;
}

std::vector<InterpolateParams> generateCombinedParamsForInterpolate() {
    const std::vector<std::vector<InterpolateParams>> allTypeParams{
        generateParamsForInterpolate<element::Type_t::f32>()
    };

    std::vector<InterpolateParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Interpolate_With_Hardcoded_Refs,
                         ReferenceInterpolateLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForInterpolate()),
                         ReferenceInterpolateLayerTest::getTestCaseName);

}  // namespace
