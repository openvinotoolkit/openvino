// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sign.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

struct SignParams {
    template <class IT, class OT>
    SignParams(const PartialShape& shape,
               const element::Type& iType,
               const element::Type& oType,
               const std::vector<IT>& iValues,
               const std::vector<OT>& oValues)
        : pshape(shape),
          inType(iType),
          outType(oType),
          inputData(CreateTensor(iType, iValues)),
          refData(CreateTensor(oType, oValues)) {}
    PartialShape pshape;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceSignLayerTest : public testing::TestWithParam<SignParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<SignParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& input_shape, const element::Type& input_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto sign = std::make_shared<op::v0::Sign>(in);
        return std::make_shared<ov::Model>(NodeVector{sign}, ParameterVector{in});
    }
};

TEST_P(ReferenceSignLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Sign_With_Hardcoded_Refs,
    ReferenceSignLayerTest,
    ::testing::Values(
        SignParams(PartialShape{6},
                   element::f32,
                   element::f32,
                   std::vector<float>{1, -2, 0, -4.8f, 4.8f, -0.0f},
                   std::vector<float>{1, -1, 0, -1, 1, 0}),
        SignParams(PartialShape{7},
                   element::f32,
                   element::f32,
                   std::vector<float>{1, -2, 0, std::numeric_limits<float>::quiet_NaN(), -4.8f, 4.8f, -0.0f},
                   std::vector<float>{1, -1, 0, std::numeric_limits<float>::quiet_NaN(), -1, 1, 0}),
        SignParams(PartialShape{6},
                   element::f16,
                   element::f16,
                   std::vector<float16>{1, -2, 0, -4.8f, 4.8f, -0.0f},
                   std::vector<float16>{1, -1, 0, -1, 1, 0}),
        SignParams(PartialShape{7},
                   element::f16,
                   element::f16,
                   std::vector<float16>{1, -2, 0, std::numeric_limits<float16>::quiet_NaN(), -4.8f, 4.8f, -0.0f},
                   std::vector<float16>{1, -1, 0, std::numeric_limits<float16>::quiet_NaN(), -1, 1, 0}),
        SignParams(PartialShape{6},
                   element::u64,
                   element::u64,
                   std::vector<uint64_t>{1, 2, 0, 4, 4, 0},
                   std::vector<uint64_t>{1, 1, 0, 1, 1, 0}),
        SignParams(PartialShape{6},
                   element::u32,
                   element::u32,
                   std::vector<uint32_t>{1, 2, 0, 4, 4, 0},
                   std::vector<uint32_t>{1, 1, 0, 1, 1, 0}),
        SignParams(PartialShape{6},
                   element::i32,
                   element::i32,
                   std::vector<int32_t>{1, -2, 0, -4, 4, -0},
                   std::vector<int32_t>{1, -1, 0, -1, 1, 0}),
        SignParams(PartialShape{6},
                   element::i64,
                   element::i64,
                   std::vector<int64_t>{1, -2, 0, -4, 4, -0},
                   std::vector<int64_t>{1, -1, 0, -1, 1, 0})),
    ReferenceSignLayerTest::getTestCaseName);
