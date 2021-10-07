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
/*
struct InterpolateParams {
    template <class IT>
    InterpolateParams(const PartialShape& iShape,
                      const PartialShape& oShape,
                      const element::Type& iType,
                      const element::Type& oType,
                      const std::vector<IT>& iValues,
                      const std::vector<IT>& oValues,
                      const std::vector<int> dims,
                      const std::vector<size_t> axisSet,
                      const std::string mode)
        : inShape(iShape),
          outShape(oShape),
          inType(iType),
          outType(oType),
          inData(CreateTensor(iType, iValues)),
          outData(CreateTensor(oType, oValues)),
          dims(dims),
          axisSet(axisSet),
          mode(mode) {}

    PartialShape inShape;
    PartialShape outShape;
    element::Type inType;
    element::Type outType;
    runtime::Tensor inData;
    runtime::Tensor outData;
    std::vector<int> dims;
    std::vector<size_t> axisSet;
    std::string mode;
};

class ReferenceInterpolateLayerTest : public testing::TestWithParam<InterpolateParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.inShape,
                                  params.outShape,
                                  params.inType,
                                  params.outType,
                                  params.dims,
                                  params.axisSet,
                                  params.mode);
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
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape,
                                                    const PartialShape& output_shape,
                                                    const element::Type& input_type,
                                                    const element::Type& output_type,
                                                    const std::vector<int> dims,
                                                    const std::vector<size_t> axisSet,
                                                    const std::string mode) {
        const auto input = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto output_shape_input = std::make_shared<op::v0::Constant>(element::Type_t::i64, dims, output_shape);
        op::v0::Interpolate::Attributes attrs;
        attrs.axes = AxisSet(axisSet);
        attrs.mode = mode;
        attrs.align_corners = false;
        auto interpolate = std::make_shared<op::v0::Interpolate>(input, output_shape_input, attrs);
        return std::make_shared<Function>(NodeVector{interpolate}, ParameterVector{input});
    }
};
*/

struct InterpolateParams {
    template <class IT>
    InterpolateParams(const PartialShape& iShape,
                      const PartialShape& oShape,
                      const element::Type& iType,
                      const element::Type& oType,
                      const std::vector<IT>& iValues,
                      const std::vector<IT>& oValues)
        : inShape(iShape),
          outShape(oShape),
          inType(iType),
          outType(oType),
          inData(CreateTensor(iType, iValues)),
          outData(CreateTensor(oType, oValues)){}

    PartialShape inShape;
    PartialShape outShape;
    element::Type inType;
    element::Type outType;
    runtime::Tensor inData;
    runtime::Tensor outData;
    std::vector<int> dims;
    std::vector<size_t> axisSet;
    std::string mode;
};

class ReferenceInterpolateLayerTest : public testing::TestWithParam<InterpolateParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.inShape,
                                  params.outShape,
                                  params.inType,
                                  params.outType);
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
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape,
                                                    const PartialShape& output_shape,
                                                    const element::Type& input_type,
                                                    const element::Type& output_type) {
        const auto input = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto output_shape_input = op::v0::Constant::create(element::Type_t::i64, {4}, {1, 1, 1, 2});
        op::v0::Interpolate::Attributes attrs;
        attrs.axes = {0, 1, 2, 3};
        attrs.mode = "linear";
        attrs.align_corners = false;
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
        InterpolateParams(ov::PartialShape{1, 1, 2, 4},
                          ov::PartialShape{1, 1, 1, 2},
                          IN_ET,
                          IN_ET,
                          std::vector<T>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                          std::vector<T>{1.0f, 2.66666651f})
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