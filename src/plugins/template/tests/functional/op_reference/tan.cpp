// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tan.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {
struct TanParams {
    template <class IT>
    TanParams(const ov::PartialShape& shape,
              const ov::element::Type& iType,
              const std::vector<IT>& iValues,
              const std::vector<IT>& oValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          refData(CreateTensor(iType, oValues)) {}
    ov::PartialShape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceTanLayerTest : public testing::TestWithParam<TanParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.pshape, params.inType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<TanParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& input_shape, const element::Type& input_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto tan = std::make_shared<op::v0::Tan>(in);
        return std::make_shared<ov::Model>(tan, ParameterVector{in});
    }
};

TEST_P(ReferenceTanLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

std::vector<TanParams> generateTanCombinedParams() {
    std::vector<TanParams> combinedParams{
        TanParams(ov::PartialShape{5},
                  ov::element::i32,
                  std::vector<int32_t>{-2, -1, 0, 1, 2},
                  std::vector<int32_t>{2, -2, 0, 2, -2}),
        TanParams(ov::PartialShape{5},
                  ov::element::i64,
                  std::vector<int64_t>{-2, -1, 0, 1, 2},
                  std::vector<int64_t>{2, -2, 0, 2, -2}),
        TanParams(ov::PartialShape{5},
                  ov::element::u32,
                  std::vector<uint32_t>{1, 2, 3, 4, 5},
                  std::vector<uint32_t>{2, 0xFFFFFFFF - 1, 0, 1, 0xFFFFFFFF - 2}),
        TanParams(ov::PartialShape{5},
                  ov::element::u64,
                  std::vector<uint64_t>{1, 2, 3, 4, 5},
                  std::vector<uint64_t>{2, 0xFFFFFFFFFFFFFFFF - 1, 0, 1, 0xFFFFFFFFFFFFFFFF - 2}),
        TanParams(ov::PartialShape{11},
                  ov::element::f32,
                  std::vector<float>{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f},
                  std::vector<float>{0.00000000f,
                                     0.25534192f,
                                     -0.25534192f,
                                     0.54630249f,
                                     -0.54630249f,
                                     1.55740772f,
                                     -1.55740772f,
                                     -2.18503986f,
                                     2.18503986f,
                                     1.15782128f,
                                     -1.15782128f}),
        TanParams(ov::PartialShape{11},
                  ov::element::f16,
                  std::vector<float16>{0.f, 0.25f, -0.25f, 0.5f, -0.5f, 1.f, -1.f, 2.f, -2.f, 4.f, -4.f},
                  std::vector<float16>{0.00000000f,
                                       0.25534192f,
                                       -0.25534192f,
                                       0.54630249f,
                                       -0.54630249f,
                                       1.55740772f,
                                       -1.55740772f,
                                       -2.18503986f,
                                       2.18503986f,
                                       1.15782128f,
                                       -1.15782128f})};
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_TAN_With_Hardcoded_Refs,
                         ReferenceTanLayerTest,
                         ::testing::ValuesIn(generateTanCombinedParams()),
                         ReferenceTanLayerTest::getTestCaseName);
}  // namespace
