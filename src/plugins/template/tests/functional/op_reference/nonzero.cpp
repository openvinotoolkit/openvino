// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <random>

#include "base_reference_test.hpp"
#include "openvino/op/non_zero.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct NonZeroParams {
    template <class IT, class OT>
    NonZeroParams(const PartialShape& dynamicShape,
                  const PartialShape& inputShape,
                  const element::Type& inType,
                  const element::Type& refType,
                  const std::vector<IT>& inputData,
                  const std::vector<OT>& refData,
                  const std::string& test_name = "")
        : dynamicShape(dynamicShape),
          inputShape(inputShape),
          inType(inType),
          refType(refType),
          inputData(CreateTensor(inType, inputData)),
          testcaseName(test_name) {
        const auto input_rank = inputShape.get_shape().size();
        const auto non_zero_num = refData.size() / input_rank;
        this->refData = CreateTensor(Shape{input_rank, non_zero_num}, refType, refData);
    }

    PartialShape dynamicShape;
    PartialShape inputShape;
    element::Type inType;
    element::Type refType;
    ov::Tensor inputData;
    ov::Tensor refData;
    std::string testcaseName;
};

class ReferenceNonZeroLayerTest : public testing::TestWithParam<NonZeroParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.dynamicShape, params.inType, params.refType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<NonZeroParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "dShape=" << param.dynamicShape << "_";
        result << "iShape=" << param.inputShape << "_";
        result << "iType=" << param.inType << "_";
        if (param.testcaseName != "") {
            result << "oType=" << param.refType << "_";
            result << param.testcaseName;
        } else {
            result << "oType=" << param.refType;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& input_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto NonZero = std::make_shared<op::v3::NonZero>(in, output_type);
        return std::make_shared<Model>(NodeVector{NonZero}, ParameterVector{in});
    }
};

TEST_P(ReferenceNonZeroLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke_NonZero_With_Hardcoded_Refs,
                         ReferenceNonZeroLayerTest,
                         ::testing::Values(NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::i32,
                                                         element::i32,
                                                         std::vector<int32_t>{1, 1, 1, 1, 1, 1},
                                                         std::vector<int32_t>{0, 0, 1, 1, 2, 2, 0, 1, 0, 1, 0, 1},
                                                         "all_1s"),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::i32,
                                                         element::i32,
                                                         std::vector<int32_t>{0, 0, 0, 0, 0, 0},
                                                         std::vector<int32_t>{},
                                                         "all_0s"),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::i32,
                                                         element::i64,
                                                         std::vector<int32_t>{1, 1, 1, 1, 1, 1},
                                                         std::vector<int64_t>{0, 0, 1, 1, 2, 2, 0, 1, 0, 1, 0, 1},
                                                         "all_1s_int64"),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::i32,
                                                         element::i64,
                                                         std::vector<int32_t>{0, 0, 0, 0, 0, 0},
                                                         std::vector<int64_t>{},
                                                         "all_0s_int64"),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::boolean,
                                                         element::i32,
                                                         std::vector<char>{false, false, false, false, true, true},
                                                         std::vector<int32_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::i8,
                                                         element::i32,
                                                         std::vector<int8_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int32_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::i16,
                                                         element::i32,
                                                         std::vector<int16_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int32_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::i32,
                                                         element::i32,
                                                         std::vector<int32_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int32_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::i64,
                                                         element::i32,
                                                         std::vector<int64_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int32_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::u8,
                                                         element::i32,
                                                         std::vector<uint8_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int32_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::u16,
                                                         element::i32,
                                                         std::vector<uint16_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int32_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::u32,
                                                         element::i32,
                                                         std::vector<uint32_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int32_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::u64,
                                                         element::i32,
                                                         std::vector<uint64_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int32_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::bf16,
                                                         element::i32,
                                                         std::vector<bfloat16>{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 3.0f},
                                                         std::vector<int32_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::f16,
                                                         element::i32,
                                                         std::vector<float16>{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 3.0f},
                                                         std::vector<int32_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::f32,
                                                         element::i32,
                                                         std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 3.0f},
                                                         std::vector<int32_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::f64,
                                                         element::i32,
                                                         std::vector<double>{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 3.0f},
                                                         std::vector<int32_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::boolean,
                                                         element::i64,
                                                         std::vector<char>{false, false, false, false, true, true},
                                                         std::vector<int64_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::i8,
                                                         element::i64,
                                                         std::vector<int8_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int64_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::i16,
                                                         element::i64,
                                                         std::vector<int16_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int64_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::i32,
                                                         element::i64,
                                                         std::vector<int32_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int64_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::i64,
                                                         element::i64,
                                                         std::vector<int64_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int64_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::u8,
                                                         element::i64,
                                                         std::vector<uint8_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int64_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::u16,
                                                         element::i64,
                                                         std::vector<uint16_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int64_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::u32,
                                                         element::i64,
                                                         std::vector<uint32_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int64_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::u64,
                                                         element::i64,
                                                         std::vector<uint64_t>{0, 0, 0, 0, 1, 3},
                                                         std::vector<int64_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::bf16,
                                                         element::i64,
                                                         std::vector<bfloat16>{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 3.0f},
                                                         std::vector<int64_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::f16,
                                                         element::i64,
                                                         std::vector<float16>{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 3.0f},
                                                         std::vector<int64_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::f32,
                                                         element::i64,
                                                         std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 3.0f},
                                                         std::vector<int64_t>{2, 2, 0, 1}),
                                           NonZeroParams(PartialShape{3, 2},
                                                         PartialShape{3, 2},
                                                         element::f64,
                                                         element::i64,
                                                         std::vector<double>{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 3.0f},
                                                         std::vector<int64_t>{2, 2, 0, 1})),
                         ReferenceNonZeroLayerTest::getTestCaseName);
}  // namespace
