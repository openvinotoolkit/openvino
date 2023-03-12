// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/grn.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct GrnParams {
    template <class IT>
    GrnParams(const float bias, const PartialShape& shape, const element::Type& iType, const std::vector<IT>& iValues,
              const std::vector<IT>& oValues)
        : bias(bias), pshape(shape), inType(iType), outType(iType), inputData(CreateTensor(iType, iValues)), refData(CreateTensor(iType, oValues)) {}
    float bias;
    PartialShape pshape;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceGrnLayerTest : public testing::TestWithParam<GrnParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.bias, params.pshape, params.inType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<GrnParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "bias=" << param.bias << "_";
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(float bias, const PartialShape& input_shape, const element::Type& input_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto grn = std::make_shared<op::v0::GRN>(in, bias);
        return std::make_shared<ov::Model>(NodeVector {grn}, ParameterVector {in});
    }
};

TEST_P(ReferenceGrnLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<GrnParams> generateGrnParams(const element::Type& type) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<GrnParams> grnParams {
        // bias 1e-6 // 2D // 3D // 4D
        GrnParams(1e-6, PartialShape {3, 4}, type, std::vector<T> {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                  std::vector<T> {0.182574f, 0.365148f, 0.547723f, 0.730297f, 0.379049f, 0.454859f, 0.530669f, 0.606478f, 0.426162f, 0.473514f, 0.520865f, 0.568217f}),
        GrnParams(1e-6, PartialShape {2, 3, 4}, type,
                  std::vector<T> {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                  std::vector<T> {0.0966737f, 0.169031f, 0.224231f, 0.267261f, 0.483368f, 0.507093f, 0.523205f, 0.534522f, 0.870063f, 0.845154f, 0.822179f, 0.801784f,
                                  0.433574f,  0.441836f, 0.449215f, 0.455842f, 0.566982f, 0.568075f, 0.569005f, 0.569803f, 0.700389f, 0.694314f, 0.688796f, 0.683763f}),
        GrnParams(1e-6, PartialShape {1, 2, 3, 4}, type,
                  std::vector<T> {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                  std::vector<T> {0.0766965f, 0.141421f, 0.196116f, 0.242536f, 0.282166f, 0.316228f, 0.345705f, 0.371391f, 0.393919f, 0.413803f, 0.431455f, 0.447214f,
                                  0.997055f,  0.989949f, 0.980581f, 0.970143f, 0.959365f, 0.948683f, 0.938343f, 0.928477f, 0.919145f, 0.910366f, 0.902134f, 0.894427f}),
        GrnParams(1e-6, PartialShape {2, 2, 3, 4}, type,
                  std::vector<T> {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                  25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48},
                  std::vector<T> {0.0766965f, 0.141421f, 0.196116f, 0.242536f, 0.282166f, 0.316228f, 0.345705f, 0.371391f, 0.393919f, 0.413803f, 0.431455f, 0.447214f,
                                  0.997055f,  0.989949f, 0.980581f, 0.970143f, 0.959365f, 0.948683f, 0.938343f, 0.928477f, 0.919145f, 0.910366f, 0.902134f, 0.894427f,
                                  0.559857f,  0.564684f, 0.56921f,  0.573462f, 0.577465f, 0.581238f, 0.584802f, 0.588172f, 0.591364f, 0.594391f, 0.597266f, 0.6f,
                                  0.828589f,  0.825307f, 0.822192f, 0.819232f, 0.816416f, 0.813733f, 0.811176f, 0.808736f, 0.806405f, 0.804176f, 0.802043f, 0.8f}),
        // bias 100.25 // 2D // 3D // 4D
        GrnParams(100.25, PartialShape {3, 4}, type, std::vector<T> {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                  std::vector<T> {0.0876216f, 0.175243f, 0.262865f, 0.350486f, 0.301923f, 0.362308f, 0.422693f, 0.483077f, 0.385076f, 0.427863f, 0.470649f, 0.513435f}),
        GrnParams(100.25, PartialShape {2, 3, 4}, type,
                  std::vector<T> {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                  std::vector<T> {0.0694629f, 0.129032f, 0.179525f, 0.222137f, 0.347314f, 0.387097f, 0.418891f, 0.444273f, 0.625166f, 0.645161f, 0.658258f, 0.66641f,
                                  0.41125f,   0.421303f, 0.430287f, 0.438356f, 0.537789f, 0.541675f, 0.54503f,  0.547945f, 0.664327f, 0.662047f, 0.659774f, 0.657534f}),
        GrnParams(100.25, PartialShape {1, 2, 3, 4}, type,
                  std::vector<T> {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                  std::vector<T> {0.0608299f, 0.115422f, 0.164091f, 0.207321f, 0.245662f, 0.279675f, 0.309889f, 0.336786f, 0.360795f, 0.38229f,  0.401596f, 0.418994f,
                                  0.790789f,  0.807954f, 0.820457f, 0.829283f, 0.835252f, 0.839026f, 0.841128f, 0.841965f, 0.841854f, 0.841037f, 0.839701f, 0.837989f}),
        GrnParams(100.25, PartialShape {2, 2, 3, 4}, type,
                  std::vector<T> {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                  25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48},
                  std::vector<T> {0.0608299f, 0.115422f, 0.164091f, 0.207321f, 0.245662f, 0.279675f, 0.309889f, 0.336786f, 0.360795f, 0.38229f,  0.401596f, 0.418994f,
                                  0.790789f,  0.807954f, 0.820457f, 0.829283f, 0.835252f, 0.839026f, 0.841128f, 0.841965f, 0.841854f, 0.841037f, 0.839701f, 0.837989f,
                                  0.546293f,  0.551788f, 0.556938f, 0.561772f, 0.566319f, 0.570601f, 0.574641f, 0.578458f, 0.582069f, 0.585489f, 0.588734f, 0.591816f,
                                  0.808514f,  0.80646f,  0.804466f, 0.802532f, 0.800658f, 0.798842f, 0.797083f, 0.795379f, 0.79373f,  0.792133f, 0.790586f, 0.789088f})};
    return grnParams;
}

std::vector<GrnParams> generateGrnCombinedParams() {
    const std::vector<std::vector<GrnParams>> grnTypeParams {generateGrnParams<element::Type_t::bf16>(element::bf16),
                                                             generateGrnParams<element::Type_t::f16>(element::f16),
                                                             generateGrnParams<element::Type_t::f32>(element::f32)};
    std::vector<GrnParams> combinedParams;
    std::for_each(grnTypeParams.begin(), grnTypeParams.end(), [&](std::vector<GrnParams> params) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    });
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_GRN_With_Hardcoded_Refs, ReferenceGrnLayerTest, ::testing::ValuesIn(generateGrnCombinedParams()),
                         ReferenceGrnLayerTest::getTestCaseName);
}  // namespace
