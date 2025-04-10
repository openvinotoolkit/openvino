// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/elu.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct EluParams {
    template <class IT>
    EluParams(const ov::PartialShape& shape,
              const ov::element::Type& iType,
              const std::vector<IT>& iValues,
              const std::vector<IT>& oValues,
              const double alpha)
        : alpha(alpha),
          pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(shape.get_shape(), iType, iValues)),
          refData(CreateTensor(shape.get_shape(), iType, oValues)) {}

    double alpha = 0;

    ov::PartialShape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceEluLayerTest : public testing::TestWithParam<EluParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType, params.alpha);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<EluParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "alpha=" << param.alpha;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& input_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type,
                                                 const double alpha) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto Elu = std::make_shared<op::v0::Elu>(in, alpha);
        return std::make_shared<ov::Model>(NodeVector{Elu}, ParameterVector{in});
    }
};

TEST_P(ReferenceEluLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<EluParams> generateEluFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<EluParams> eluParams{
        EluParams(ov::PartialShape{3, 2},
                  IN_ET,
                  std::vector<T>{-2.f, 3.f, -2.f, 1.f, -1.f, 0.f},
                  std::vector<T>{-0.432332358f, 3.f, -0.432332358f, 1.f, -0.316060279f, 0.f},
                  0.5f),
        EluParams(ov::PartialShape{3, 2},
                  IN_ET,
                  std::vector<T>{-2.f, 3.f, -2.f, 1.f, -1.f, 0.f},
                  std::vector<T>{0.864664717f, 3.f, 0.864664717f, 1.f, 0.632120559f, 0.f},
                  -1.f)};
    return eluParams;
}

template <element::Type_t IN_ET>
std::vector<EluParams> generateEluIntParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<EluParams> eluParams{EluParams(ov::PartialShape{3, 2},
                                               IN_ET,
                                               std::vector<T>{-2, 3, -2, 1, -1, 0},
                                               std::vector<T>{0, 3, 0, 1, 0, 0},
                                               0.5f),
                                     EluParams(ov::PartialShape{3, 2},
                                               IN_ET,
                                               std::vector<T>{-2, 3, -2, 1, -1, 0},
                                               std::vector<T>{0, 3, 0, 1, 0, 0},
                                               -1.f)};
    return eluParams;
}

template <element::Type_t IN_ET>
std::vector<EluParams> generateEluUintParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<EluParams> eluParams{EluParams(ov::PartialShape{3, 2},
                                               IN_ET,
                                               std::vector<T>{5, 4, 3, 2, 1, 0},
                                               std::vector<T>{5, 4, 3, 2, 1, 0},
                                               0.5f),
                                     EluParams(ov::PartialShape{3, 2},
                                               IN_ET,
                                               std::vector<T>{5, 4, 3, 2, 1, 0},
                                               std::vector<T>{5, 4, 3, 2, 1, 0},
                                               -1.f)};
    return eluParams;
}
std::vector<EluParams> generateEluCombinedParams() {
    const std::vector<std::vector<EluParams>> eluTypeParams{generateEluFloatParams<element::Type_t::f32>(),
                                                            generateEluFloatParams<element::Type_t::f16>(),
                                                            generateEluFloatParams<element::Type_t::bf16>(),
                                                            generateEluIntParams<element::Type_t::i8>(),
                                                            generateEluIntParams<element::Type_t::i16>(),
                                                            generateEluIntParams<element::Type_t::i32>(),
                                                            generateEluIntParams<element::Type_t::i64>(),
                                                            generateEluUintParams<element::Type_t::u8>(),
                                                            generateEluUintParams<element::Type_t::u16>(),
                                                            generateEluUintParams<element::Type_t::u32>(),
                                                            generateEluUintParams<element::Type_t::u64>()};
    std::vector<EluParams> combinedParams;

    for (const auto& params : eluTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Elu_With_Hardcoded_Refs,
                         ReferenceEluLayerTest,
                         testing::ValuesIn(generateEluCombinedParams()),
                         ReferenceEluLayerTest::getTestCaseName);

}  // namespace
