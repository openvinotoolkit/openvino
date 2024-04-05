// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct TransposeParams {
    TransposeParams(const PartialShape& dynamicDataShape,
                    const reference_tests::Tensor& dataTensor,
                    const reference_tests::Tensor& axisTensor,
                    const reference_tests::Tensor& expectedTensor,
                    const std::string& testcaseName,
                    const std::pair<std::string, std::string>& expectedException = {})
        : dynamicDataShape(dynamicDataShape),
          dataTensor(dataTensor),
          axisTensor(axisTensor),
          expectedTensor(expectedTensor),
          testcaseName(testcaseName),
          expectedException(expectedException) {}

    PartialShape dynamicDataShape;
    reference_tests::Tensor dataTensor;
    reference_tests::Tensor axisTensor;
    reference_tests::Tensor expectedTensor;
    std::string testcaseName;
    std::pair<std::string, std::string> expectedException;
};

class ReferenceTransposeLayerTest : public testing::TestWithParam<TransposeParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        if (params.dynamicDataShape.is_static()) {
            inputData = {params.dataTensor.data};
        } else {
            inputData = {params.dataTensor.data, params.axisTensor.data};
        }
        refOutData = {params.expectedTensor.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<TransposeParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "ddShape=" << param.dynamicDataShape;
        result << "_dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_aType=" << param.axisTensor.type;
        result << "_aShape=" << param.axisTensor.shape;
        result << "_eType=" << param.expectedTensor.type;
        result << "_eShape=" << param.expectedTensor.shape;
        result << "_=" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const TransposeParams& params) {
        std::shared_ptr<Model> function;
        if (params.dynamicDataShape.is_static()) {
            const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, params.dataTensor.shape);
            const auto axis = std::make_shared<op::v0::Constant>(params.axisTensor.type,
                                                                 params.axisTensor.shape,
                                                                 params.axisTensor.data.data());
            const auto axisI64 = std::make_shared<op::v0::Convert>(axis, element::i64);
            const auto transpose = std::make_shared<op::v1::Transpose>(data, axisI64);
            function = std::make_shared<ov::Model>(NodeVector{transpose}, ParameterVector{data});
        } else {
            const auto data = std::make_shared<op::v0::Parameter>(params.dataTensor.type, PartialShape::dynamic());
            const auto axis =
                std::make_shared<op::v0::Parameter>(params.axisTensor.type, PartialShape{Dimension::dynamic()});
            const auto axisI64 = std::make_shared<op::v0::Convert>(axis, element::i64);
            const auto transpose = std::make_shared<op::v1::Transpose>(data, axisI64);
            function = std::make_shared<ov::Model>(NodeVector{transpose}, ParameterVector{data, axis});
        }
        return function;
    }
};

TEST_P(ReferenceTransposeLayerTest, CompareWithRefs) {
    const auto& params = GetParam();
    if (params.expectedException.first.empty()) {
        Exec();
    } else {
        try {
            Exec();
            FAIL() << params.expectedException.second;
        } catch (const ov::Exception& error) {
            EXPECT_HAS_SUBSTRING(error.what(), params.expectedException.first);
        } catch (const std::exception& error) {
            FAIL() << "Failed for unexpected reason: " << error.what();
        } catch (...) {
            FAIL() << "Failed for unknown reason";
        }
    }
}

template <element::Type_t IN_ET>
std::vector<TransposeParams> generateTransposeParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    std::vector<TransposeParams> transposeParams{
        // transpose_basic
        TransposeParams(PartialShape::dynamic(),
                        reference_tests::Tensor(IN_ET, {2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}),
                        reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{0, 1}),
                        reference_tests::Tensor(IN_ET, {2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}),
                        "transpose_basic_1"),
        TransposeParams(PartialShape::dynamic(),
                        reference_tests::Tensor(IN_ET, {2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6}),
                        reference_tests::Tensor(element::i64, {2}, std::vector<int64_t>{1, 0}),
                        reference_tests::Tensor(IN_ET, {3, 2}, std::vector<T>{1, 4, 2, 5, 3, 6}),
                        "transpose_basic_2"),
        TransposeParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(IN_ET, {2, 2, 3}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
            reference_tests::Tensor(element::i64, {3}, std::vector<int64_t>{2, 1, 0}),
            reference_tests::Tensor(IN_ET, {3, 2, 2}, std::vector<T>{1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12}),
            "transpose_basic_3"),
        // transpose_axes_constant
        TransposeParams(
            {},
            reference_tests::Tensor(IN_ET, {2, 1, 3, 4}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,
                                                                        9,  10, 11, 12, 13, 14, 15, 16,
                                                                        17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor(element::i64, {4}, std::vector<int64_t>{2, 3, 0, 1}),
            reference_tests::Tensor(IN_ET, {3, 4, 2, 1}, std::vector<T>{1, 13, 2, 14, 3, 15, 4,  16, 5,  17, 6,  18,
                                                                        7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24}),
            "transpose_axes_constant"),
        // transpose_axes_empty_constant
        TransposeParams(
            {},
            reference_tests::Tensor(IN_ET, {2, 1, 3, 4}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,
                                                                        9,  10, 11, 12, 13, 14, 15, 16,
                                                                        17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor(element::i64, {0}, std::vector<int64_t>{}),
            reference_tests::Tensor(IN_ET, {4, 3, 1, 2}, std::vector<T>{1, 13, 5, 17, 9,  21, 2, 14, 6, 18, 10, 22,
                                                                        3, 15, 7, 19, 11, 23, 4, 16, 8, 20, 12, 24}),
            "transpose_axes_empty_constant"),
        // transpose_axes_parameter_static_shapes
        TransposeParams(
            {},
            reference_tests::Tensor(IN_ET, {2, 1, 3, 4}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,
                                                                        9,  10, 11, 12, 13, 14, 15, 16,
                                                                        17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor(element::i64, {4}, std::vector<int64_t>{2, 3, 0, 1}),
            reference_tests::Tensor(IN_ET, {3, 4, 2, 1}, std::vector<T>{1, 13, 2, 14, 3, 15, 4,  16, 5,  17, 6,  18,
                                                                        7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24}),
            "transpose_axes_parameter_static_shapes"),
        // transpose_axes_parameter_dynamic_shapes
        TransposeParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(IN_ET, {2, 1, 3, 4}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,
                                                                        9,  10, 11, 12, 13, 14, 15, 16,
                                                                        17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor(element::i64, {4}, std::vector<int64_t>{2, 3, 0, 1}),
            reference_tests::Tensor(IN_ET, {3, 4, 2, 1}, std::vector<T>{1, 13, 2, 14, 3, 15, 4,  16, 5,  17, 6,  18,
                                                                        7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24}),
            "transpose_axes_parameter_dynamic_shapes"),
        // transpose_int_data_axes_constant
        TransposeParams(
            {},
            reference_tests::Tensor(IN_ET, {2, 1, 3, 4}, std::vector<T>{1,  2,  3,  4,  5,  6,  7,  8,
                                                                        9,  10, 11, 12, 13, 14, 15, 16,
                                                                        17, 18, 19, 20, 21, 22, 23, 24}),
            reference_tests::Tensor(element::i64, {4}, std::vector<int64_t>{2, 3, 0, 1}),
            reference_tests::Tensor(IN_ET, {3, 4, 2, 1}, std::vector<T>{1, 13, 2, 14, 3, 15, 4,  16, 5,  17, 6,  18,
                                                                        7, 19, 8, 20, 9, 21, 10, 22, 11, 23, 12, 24}),
            "transpose_int_data_axes_constant"),
    };
    return transposeParams;
}

template <element::Type_t IN_ET>
std::vector<TransposeParams> generateThrowingTransposeParams() {
    using T = typename element_type_traits<IN_ET>::value_type;
    return std::vector<TransposeParams>{
        TransposeParams(PartialShape::dynamic(),
                        reference_tests::Tensor(IN_ET, {2, 3, 1}, std::vector<T>{1, 2, 3, 4, 5, 6}),
                        reference_tests::Tensor(element::i64, {3}, std::vector<int64_t>{2, 1, 2}),
                        reference_tests::Tensor(IN_ET, {2, 3, 1}, std::vector<T>{1, 2, 3, 4, 5, 6}),
                        "duplicated_axes_values",
                        {"not valid for input shape", "Duplicated axes values not detected"}),
        TransposeParams(PartialShape::dynamic(),
                        reference_tests::Tensor(IN_ET, {2, 3, 1}, std::vector<T>{1, 2, 3, 4, 5, 6}),
                        reference_tests::Tensor(element::i64, {3}, std::vector<int64_t>{0, 1, 3}),
                        reference_tests::Tensor(IN_ET, {2, 3, 1}, std::vector<T>{1, 2, 3, 4, 5, 6}),
                        "out_of_shape_axes_values",
                        {"not valid for input shape", "Out of shape axes not detected"}),
        TransposeParams(PartialShape::dynamic(),
                        reference_tests::Tensor(IN_ET, {2, 3, 1}, std::vector<T>{1, 2, 3, 4, 5, 6}),
                        reference_tests::Tensor(element::i64, {3}, std::vector<int64_t>{-1, -2, -3}),
                        reference_tests::Tensor(IN_ET, {2, 3, 1}, std::vector<T>{1, 4, 2, 5, 3, 6}),
                        "negative_axes_values",
                        {"not valid for input shape", "Negative axes for Transpose were not supported before"}),
    };
}

std::vector<TransposeParams> generateTransposeCombinedParams() {
    const std::vector<std::vector<TransposeParams>> transposeTypeParams{
        generateTransposeParams<element::Type_t::i8>(),
        generateTransposeParams<element::Type_t::i16>(),
        generateTransposeParams<element::Type_t::i32>(),
        generateTransposeParams<element::Type_t::i64>(),
        generateTransposeParams<element::Type_t::u8>(),
        generateTransposeParams<element::Type_t::u16>(),
        generateTransposeParams<element::Type_t::u32>(),
        generateTransposeParams<element::Type_t::u64>(),
        generateTransposeParams<element::Type_t::f16>(),
        generateTransposeParams<element::Type_t::f32>(),
        generateThrowingTransposeParams<element::Type_t::f32>(),
        generateThrowingTransposeParams<element::Type_t::i32>(),
    };
    std::vector<TransposeParams> combinedParams;

    for (const auto& params : transposeTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Transpose_With_Hardcoded_Refs,
                         ReferenceTransposeLayerTest,
                         testing::ValuesIn(generateTransposeCombinedParams()),
                         ReferenceTransposeLayerTest::getTestCaseName);
}  // namespace
