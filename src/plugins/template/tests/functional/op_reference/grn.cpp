// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grn.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct GrnParams {
    template <class IT>
    GrnParams(const float bias,
              const PartialShape& shape,
              const element::Type& iType,
              const std::vector<IT>& iValues,
              const std::vector<IT>& oValues)
        : bias(bias),
          pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(pshape.get_shape(), iType, iValues)),
          refData(CreateTensor(pshape.get_shape(), iType, oValues)) {}
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
        const auto& params = GetParam();
        function = CreateFunction(params.bias, params.pshape, params.inType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<GrnParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "bias=" << param.bias << "_";
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(float bias,
                                                 const PartialShape& input_shape,
                                                 const element::Type& input_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto grn = std::make_shared<op::v0::GRN>(in, bias);
        return std::make_shared<ov::Model>(NodeVector{grn}, ParameterVector{in});
    }
};

TEST_P(ReferenceGrnLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<GrnParams> generateGrnParams(const element::Type& type) {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<GrnParams> grnParams{
        // bias 1e-6 // 2D // 3D // 4D
        GrnParams(1e-6f,
                  PartialShape{3, 4},
                  type,
                  std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                  std::vector<T>{0.182574,
                                 0.365148,
                                 0.547723,
                                 0.730297,
                                 0.379049,
                                 0.454859,
                                 0.530669,
                                 0.606478,
                                 0.426162,
                                 0.473514,
                                 0.520865,
                                 0.568217}),
        GrnParams(1e-6f,
                  PartialShape{2, 3, 4},
                  type,
                  std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                  std::vector<T>{0.0966737, 0.169031, 0.224231, 0.267261, 0.483368, 0.507093, 0.523205, 0.534522,
                                 0.870063,  0.845154, 0.822179, 0.801784, 0.433574, 0.441836, 0.449215, 0.455842,
                                 0.566982,  0.568075, 0.569005, 0.569803, 0.700389, 0.694314, 0.688796, 0.683763}),
        GrnParams(1e-6f,
                  PartialShape{1, 2, 3, 4},
                  type,
                  std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                  std::vector<T>{0.0766965, 0.141421, 0.196116, 0.242536, 0.282166, 0.316228, 0.345705, 0.371391,
                                 0.393919,  0.413803, 0.431455, 0.447214, 0.997055, 0.989949, 0.980581, 0.970143,
                                 0.959365,  0.948683, 0.938343, 0.928477, 0.919145, 0.910366, 0.902134, 0.894427}),
        GrnParams(1e-6f,
                  PartialShape{2, 2, 3, 4},
                  type,
                  std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48},
                  std::vector<T>{0.0766965, 0.141421, 0.196116, 0.242536, 0.282166, 0.316228, 0.345705, 0.371391,
                                 0.393919,  0.413803, 0.431455, 0.447214, 0.997055, 0.989949, 0.980581, 0.970143,
                                 0.959365,  0.948683, 0.938343, 0.928477, 0.919145, 0.910366, 0.902134, 0.894427,
                                 0.559857,  0.564684, 0.56921,  0.573462, 0.577465, 0.581238, 0.584802, 0.588172,
                                 0.591364,  0.594391, 0.597266, 0.6,      0.828589, 0.825307, 0.822192, 0.819232,
                                 0.816416,  0.813733, 0.811176, 0.808736, 0.806405, 0.804176, 0.802043, 0.8}),
        // bias 100.25 // 2D // 3D // 4D
        GrnParams(100.25f,
                  PartialShape{3, 4},
                  type,
                  std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                  std::vector<T>{0.0876216,
                                 0.175243,
                                 0.262865,
                                 0.350486,
                                 0.301923,
                                 0.362308,
                                 0.422693,
                                 0.483077,
                                 0.385076,
                                 0.427863,
                                 0.470649,
                                 0.513435}),
        GrnParams(100.25f,
                  PartialShape{2, 3, 4},
                  type,
                  std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                  std::vector<T>{0.0694629, 0.129032, 0.179525, 0.222137, 0.347314, 0.387097, 0.418891, 0.444273,
                                 0.625166,  0.645161, 0.658258, 0.66641,  0.41125,  0.421303, 0.430287, 0.438356,
                                 0.537789,  0.541675, 0.54503,  0.547945, 0.664327, 0.662047, 0.659774, 0.657534}),
        GrnParams(100.25f,
                  PartialShape{1, 2, 3, 4},
                  type,
                  std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
                  std::vector<T>{0.0608299, 0.115422, 0.164091, 0.207321, 0.245662, 0.279675, 0.309889, 0.336786,
                                 0.360795,  0.38229,  0.401596, 0.418994, 0.790789, 0.807954, 0.820457, 0.829283,
                                 0.835252,  0.839026, 0.841128, 0.841965, 0.841854, 0.841037, 0.839701, 0.837989f}),
        GrnParams(100.25f,
                  PartialShape{2, 2, 3, 4},
                  type,
                  std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48},
                  std::vector<T>{0.0608299, 0.115422, 0.164091, 0.207321, 0.245662, 0.279675, 0.309889, 0.336786,
                                 0.360795,  0.38229,  0.401596, 0.418994, 0.790789, 0.807954, 0.820457, 0.829283,
                                 0.835252,  0.839026, 0.841128, 0.841965, 0.841854, 0.841037, 0.839701, 0.837989,
                                 0.546293,  0.551788, 0.556938, 0.561772, 0.566319, 0.570601, 0.574641, 0.578458,
                                 0.582069,  0.585489, 0.588734, 0.591816, 0.808514, 0.80646,  0.804466, 0.802532,
                                 0.800658,  0.798842, 0.797083, 0.795379, 0.79373,  0.792133, 0.790586, 0.789088})};
    return grnParams;
}

std::vector<GrnParams> generateGrnCombinedParams() {
    const std::vector<std::vector<GrnParams>> grnTypeParams{generateGrnParams<element::Type_t::bf16>(element::bf16),
                                                            generateGrnParams<element::Type_t::f16>(element::f16),
                                                            generateGrnParams<element::Type_t::f32>(element::f32)};
    std::vector<GrnParams> combinedParams;
    std::for_each(grnTypeParams.begin(), grnTypeParams.end(), [&](std::vector<GrnParams> params) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    });
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_GRN_With_Hardcoded_Refs,
                         ReferenceGrnLayerTest,
                         ::testing::ValuesIn(generateGrnCombinedParams()),
                         ReferenceGrnLayerTest::getTestCaseName);
}  // namespace
