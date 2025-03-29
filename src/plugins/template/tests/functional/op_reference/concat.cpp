// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/concat.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/subtract.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ConcatParams {
    ConcatParams(const PartialShape& dynamicShape,
                 const reference_tests::Tensor& A,
                 const reference_tests::Tensor& B,
                 const reference_tests::Tensor& C,
                 const int32_t axis,
                 const reference_tests::Tensor& expected,
                 const std::string& testcaseName = "")
        : dynamicShape(dynamicShape),
          A(A),
          B(B),
          C(C),
          axis(axis),
          expected(expected),
          testcaseName(testcaseName) {}

    PartialShape dynamicShape;
    reference_tests::Tensor A;
    reference_tests::Tensor B;
    reference_tests::Tensor C;
    int32_t axis;
    reference_tests::Tensor expected;
    std::string testcaseName;
};

class ReferenceConcatTest : public testing::TestWithParam<ConcatParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.A.data, params.B.data, params.C.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConcatParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dynShape=" << param.dynamicShape;
        result << "_aType=" << param.A.type;
        result << "_aShape=" << param.A.shape;
        result << "_bType=" << param.B.type;
        result << "_bShape=" << param.B.shape;
        result << "_cType=" << param.C.type;
        result << "_cShape=" << param.C.shape;
        result << "_axis=" << param.axis;
        result << "_eType=" << param.expected.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expected.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_rShape=" << param.expected.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ConcatParams& params) {
        std::shared_ptr<op::v0::Parameter> A, B, C;
        if (params.dynamicShape.is_dynamic()) {
            A = std::make_shared<op::v0::Parameter>(params.A.type, params.dynamicShape);
            B = std::make_shared<op::v0::Parameter>(params.B.type, params.dynamicShape);
            C = std::make_shared<op::v0::Parameter>(params.C.type, params.dynamicShape);
        } else {
            A = std::make_shared<op::v0::Parameter>(params.A.type, params.A.shape);
            B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);
            C = std::make_shared<op::v0::Parameter>(params.C.type, params.C.shape);
        }
        auto f = std::make_shared<Model>(std::make_shared<op::v0::Concat>(NodeVector{A, B, C}, params.axis),
                                         ParameterVector{A, B, C});
        return f;
    }
};

TEST_P(ReferenceConcatTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ConcatParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ConcatParams> params{
        ConcatParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(ET, {2, 2}, std::vector<T>{2, 4, 8, 16}),
            reference_tests::Tensor(ET, {2, 3}, std::vector<T>{1, 2, 4, 8, 16, 32}),
            reference_tests::Tensor(ET, {2, 3}, std::vector<T>{2, 3, 5, 7, 11, 13}),
            -1,
            reference_tests::Tensor(ET, {2, 8}, std::vector<T>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}),
            "concat_negative_axis"),
        ConcatParams(
            {},
            reference_tests::Tensor(ET, {2, 2}, std::vector<T>{2, 4, 8, 16}),
            reference_tests::Tensor(ET, {2, 3}, std::vector<T>{1, 2, 4, 8, 16, 32}),
            reference_tests::Tensor(ET, {2, 3}, std::vector<T>{2, 3, 5, 7, 11, 13}),
            1,
            reference_tests::Tensor(ET, {2, 8}, std::vector<T>{2, 4, 1, 2, 4, 2, 3, 5, 8, 16, 8, 16, 32, 7, 11, 13}),
            "concat_matrix_colwise"),
        ConcatParams(
            {},
            reference_tests::Tensor(ET, {2, 2}, std::vector<T>{2, 4, 8, 16}),
            reference_tests::Tensor(ET, {3, 2}, std::vector<T>{1, 2, 4, 8, 16, 32}),
            reference_tests::Tensor(ET, {3, 2}, std::vector<T>{2, 3, 5, 7, 11, 13}),
            0,
            reference_tests::Tensor(ET, {8, 2}, std::vector<T>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 2, 3, 5, 7, 11, 13}),
            "concat_matrix_rowwise"),
        ConcatParams({},
                     reference_tests::Tensor(ET, {4}, std::vector<T>{2, 4, 8, 16}),
                     reference_tests::Tensor(ET, {6}, std::vector<T>{1, 2, 4, 8, 16, 32}),
                     reference_tests::Tensor(ET, {2}, std::vector<T>{18, 19}),
                     0,
                     reference_tests::Tensor(ET, {12}, std::vector<T>{2, 4, 8, 16, 1, 2, 4, 8, 16, 32, 18, 19}),
                     "concat_vector"),
        ConcatParams({},
                     reference_tests::Tensor(ET, {1, 1, 1, 1}, std::vector<T>{1}),
                     reference_tests::Tensor(ET, {1, 1, 1, 1}, std::vector<T>{2}),
                     reference_tests::Tensor(ET, {1, 1, 1, 1}, std::vector<T>{3}),
                     0,
                     reference_tests::Tensor(ET, {3, 1, 1, 1}, std::vector<T>{1, 2, 3}),
                     "concat_4d_tensor"),
        ConcatParams({},
                     reference_tests::Tensor(ET, {1, 1}, std::vector<T>{1}),
                     reference_tests::Tensor(ET, {1, 1}, std::vector<T>{2}),
                     reference_tests::Tensor(ET, {1, 1}, std::vector<T>{3}),
                     0,
                     reference_tests::Tensor(ET, {3, 1}, std::vector<T>{1, 2, 3}),
                     "concat_2d_tensor"),
    };
    return params;
}

std::vector<ConcatParams> generateStringParams() {
    const auto ET = ov::element::string;
    using T = typename element_type_traits<ov::element::string>::value_type;
    std::vector<ConcatParams> params{
        ConcatParams(PartialShape::dynamic(),
                     reference_tests::Tensor(ET, {2}, std::vector<T>{"  ", "..."}),
                     reference_tests::Tensor(ET, {2}, std::vector<T>{"Ab cd", "1"}),
                     reference_tests::Tensor(ET, {0}, std::vector<T>{}),
                     0,
                     reference_tests::Tensor(ET, {4}, std::vector<T>{"  ", "...", "Ab cd", "1"}),
                     "concat_string_2_1D_ax_0"),
        ConcatParams(PartialShape::dynamic(),
                     reference_tests::Tensor(ET, {2}, std::vector<T>{"  ", "..."}),
                     reference_tests::Tensor(ET, {1}, std::vector<T>{"Ab cd"}),
                     reference_tests::Tensor(ET, {3}, std::vector<T>{"1.2", "; ", " \n "}),
                     0,
                     reference_tests::Tensor(ET, {6}, std::vector<T>{"  ", "...", "Ab cd", "1.2", "; ", " \n "}),
                     "concat_string_3_1D_ax_0"),
        ConcatParams(PartialShape::dynamic(),
                     reference_tests::Tensor(ET, {1, 2}, std::vector<T>{"  ", "..."}),
                     reference_tests::Tensor(ET, {1, 1}, std::vector<T>{"Ab cd"}),
                     reference_tests::Tensor(ET, {1, 3}, std::vector<T>{"1.2", "; ", " \n "}),
                     1,
                     reference_tests::Tensor(ET, {1, 6}, std::vector<T>{"  ", "...", "Ab cd", "1.2", "; ", " \n "}),
                     "concat_string_3_2D_ax_1"),
        ConcatParams(PartialShape::dynamic(),
                     reference_tests::Tensor(ET, {2, 1}, std::vector<T>{"  ", "..."}),
                     reference_tests::Tensor(ET, {1, 1}, std::vector<T>{"Ab cd"}),
                     reference_tests::Tensor(ET, {3, 1}, std::vector<T>{"1.2", "; ", " \n "}),
                     0,
                     reference_tests::Tensor(ET, {6, 1}, std::vector<T>{"  ", "...", "Ab cd", "1.2", "; ", " \n "}),
                     "concat_string_3_2D_ax_0"),
        ConcatParams(
            PartialShape::dynamic(),
            reference_tests::Tensor(ET, {1, 2, 1, 2}, std::vector<T>{{"1,2", "3;4;", "- 5 -", "...."}}),
            reference_tests::Tensor(ET, {1, 2, 1, 3}, std::vector<T>{"a", ",b", "c. ", "d d ", " e ", "F"}),
            reference_tests::Tensor(ET, {1, 2, 1, 4}, std::vector<T>{"defg ", ".", "3 4", ":", " \0", "_", "\n", ";"}),
            3,
            reference_tests::Tensor(ET,
                                    {1, 2, 1, 9},
                                    std::vector<T>{"1,2",
                                                   "3;4;",
                                                   "a",
                                                   ",b",
                                                   "c. ",
                                                   "defg ",
                                                   ".",
                                                   "3 4",
                                                   ":",
                                                   "- 5 -",
                                                   "....",
                                                   "d d ",
                                                   " e ",
                                                   "F",
                                                   " ",
                                                   "_",
                                                   "\n",
                                                   ";"}),
            "concat_string_3_4D_ax_3"),
    };
    return params;
}

std::vector<ConcatParams> generateCombinedParams() {
    const std::vector<std::vector<ConcatParams>> generatedParams{
        generateParams<element::Type_t::i8>(),
        generateParams<element::Type_t::i16>(),
        generateParams<element::Type_t::i32>(),
        generateParams<element::Type_t::i64>(),
        generateParams<element::Type_t::u8>(),
        generateParams<element::Type_t::u16>(),
        generateParams<element::Type_t::u32>(),
        generateParams<element::Type_t::u64>(),
        generateParams<element::Type_t::bf16>(),
        generateParams<element::Type_t::f16>(),
        generateParams<element::Type_t::f32>(),
        generateParams<element::Type_t::f64>(),
        generateStringParams(),
    };
    std::vector<ConcatParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Concat_With_Hardcoded_Refs,
                         ReferenceConcatTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceConcatTest::getTestCaseName);

//// concat_vector_params, concat_vector_large

struct ConcatParamsVectorLarge {
    ConcatParamsVectorLarge(const uint32_t numInputs, const std::string& testcaseName = "")
        : numInputs(numInputs),
          testcaseName(testcaseName) {}

    uint32_t numInputs;
    std::string testcaseName;
};

class ReferenceConcatTestVectorLarge : public testing::TestWithParam<ConcatParamsVectorLarge>,
                                       public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();

        Shape shape_a{1};
        NodeVector inputs;
        ParameterVector inputs_param;
        for (uint32_t i = 0; i < params.numInputs; i++) {
            auto A = std::make_shared<op::v0::Parameter>(element::f32, shape_a);
            inputs_param.push_back(A);
            inputs.push_back(A);
        }
        function = std::make_shared<Model>(std::make_shared<op::v0::Concat>(inputs, 0), inputs_param);

        std::vector<float> ref_result;
        for (uint32_t i = 0; i < params.numInputs; i++) {
            auto a = CreateTensor(shape_a, element::f32, std::vector<float>{static_cast<float>(i)});
            ref_result.push_back(static_cast<float>(i));
            inputData.push_back(a);
        }
        refOutData = {CreateTensor(Shape{params.numInputs}, element::f32, ref_result)};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConcatParamsVectorLarge>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        if (param.testcaseName != "") {
            result << "numInputs=" << param.numInputs;
            result << "_=" << param.testcaseName;
        } else {
            result << "numInputs=" << param.numInputs;
        }
        return result.str();
    }
};

TEST_P(ReferenceConcatTestVectorLarge, CompareWithRefs) {
    Exec();
}

std::vector<ConcatParamsVectorLarge> generateParamsVectorLarge() {
    std::vector<ConcatParamsVectorLarge> params{
        ConcatParamsVectorLarge(100, "concat_vector_large_100"),
        ConcatParamsVectorLarge(128, "concat_vector_large_128"),
        ConcatParamsVectorLarge(999, "concat_vector_large_999"),
    };
    return params;
}

std::vector<ConcatParamsVectorLarge> generateCombinedParamsVectorLarge() {
    const std::vector<std::vector<ConcatParamsVectorLarge>> generatedParams{
        generateParamsVectorLarge(),
    };
    std::vector<ConcatParamsVectorLarge> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Concat_With_Hardcoded_Refs,
                         ReferenceConcatTestVectorLarge,
                         testing::ValuesIn(generateCombinedParamsVectorLarge()),
                         ReferenceConcatTestVectorLarge::getTestCaseName);

//// concat_in_place_2d_tensor

struct ConcatParamsInPlace2dTensor {
    ConcatParamsInPlace2dTensor(const reference_tests::Tensor& A,
                                const reference_tests::Tensor& B,
                                const reference_tests::Tensor& C,
                                const reference_tests::Tensor& D,
                                const int32_t axis,
                                const reference_tests::Tensor& expected,
                                const std::string& testcaseName = "")
        : A(A),
          B(B),
          C(C),
          D(D),
          axis(axis),
          expected(expected),
          testcaseName(testcaseName) {}

    reference_tests::Tensor A;
    reference_tests::Tensor B;
    reference_tests::Tensor C;
    reference_tests::Tensor D;
    int32_t axis;
    reference_tests::Tensor expected;
    std::string testcaseName;
};

class ReferenceConcatTestInPlace2dTensor : public testing::TestWithParam<ConcatParamsInPlace2dTensor>,
                                           public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.A.data, params.B.data, params.C.data, params.D.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConcatParamsInPlace2dTensor>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "aType=" << param.A.type;
        result << "_aShape=" << param.A.shape;
        result << "_bType=" << param.B.type;
        result << "_bShape=" << param.B.shape;
        result << "_cType=" << param.C.type;
        result << "_cShape=" << param.C.shape;
        result << "_dType=" << param.D.type;
        result << "_dShape=" << param.D.shape;
        result << "_axis=" << param.axis;
        result << "_eType=" << param.expected.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expected.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_rShape=" << param.expected.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ConcatParamsInPlace2dTensor& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.A.type, params.A.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);
        const auto add1 = std::make_shared<op::v1::Add>(A, B);
        const auto C = std::make_shared<op::v0::Parameter>(params.C.type, params.C.shape);
        const auto D = std::make_shared<op::v0::Parameter>(params.D.type, params.D.shape);
        const auto add2 = std::make_shared<op::v1::Add>(C, D);
        const auto subtract = std::make_shared<op::v1::Subtract>(C, A);
        const auto f =
            std::make_shared<Model>(std::make_shared<op::v0::Concat>(NodeVector{add1, add2, subtract}, params.axis),
                                    ParameterVector{A, B, C, D});
        return f;
    }
};

TEST_P(ReferenceConcatTestInPlace2dTensor, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ConcatParamsInPlace2dTensor> generateParamsInPlace2dTensor() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ConcatParamsInPlace2dTensor> params{
        ConcatParamsInPlace2dTensor(reference_tests::Tensor(ET, {1, 1}, std::vector<T>{1}),
                                    reference_tests::Tensor(ET, {1, 1}, std::vector<T>{2}),
                                    reference_tests::Tensor(ET, {1, 1}, std::vector<T>{3}),
                                    reference_tests::Tensor(ET, {1, 1}, std::vector<T>{4}),
                                    0,
                                    reference_tests::Tensor(ET, {3, 1}, std::vector<T>{3, 7, 2}),
                                    "concat_in_place_2d_tensor"),
    };
    return params;
}

std::vector<ConcatParamsInPlace2dTensor> generateCombinedParamsInPlace2dTensor() {
    const std::vector<std::vector<ConcatParamsInPlace2dTensor>> generatedParams{
        generateParamsInPlace2dTensor<element::Type_t::i32>(),
        generateParamsInPlace2dTensor<element::Type_t::i64>(),
        generateParamsInPlace2dTensor<element::Type_t::u32>(),
        generateParamsInPlace2dTensor<element::Type_t::u64>(),
        generateParamsInPlace2dTensor<element::Type_t::bf16>(),
        generateParamsInPlace2dTensor<element::Type_t::f16>(),
        generateParamsInPlace2dTensor<element::Type_t::f32>(),
    };
    std::vector<ConcatParamsInPlace2dTensor> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Concat_With_Hardcoded_Refs,
                         ReferenceConcatTestInPlace2dTensor,
                         testing::ValuesIn(generateCombinedParamsInPlace2dTensor()),
                         ReferenceConcatTestInPlace2dTensor::getTestCaseName);

//// concat_in_place_propagate_2d_tensor

struct ConcatParamsInPlacePropagate2dTensor {
    ConcatParamsInPlacePropagate2dTensor(const reference_tests::Tensor& A,
                                         const reference_tests::Tensor& B,
                                         const reference_tests::Tensor& C,
                                         const reference_tests::Tensor& D,
                                         const int32_t axis,
                                         const reference_tests::Tensor& expected,
                                         const std::string& testcaseName = "")
        : A(A),
          B(B),
          C(C),
          D(D),
          axis(axis),
          expected(expected),
          testcaseName(testcaseName) {}

    reference_tests::Tensor A;
    reference_tests::Tensor B;
    reference_tests::Tensor C;
    reference_tests::Tensor D;
    int32_t axis;
    reference_tests::Tensor expected;
    std::string testcaseName;
};

class ReferenceConcatTestInPlacePropagate2dTensor : public testing::TestWithParam<ConcatParamsInPlacePropagate2dTensor>,
                                                    public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.A.data, params.B.data, params.C.data, params.D.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConcatParamsInPlacePropagate2dTensor>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "aType=" << param.A.type;
        result << "_aShape=" << param.A.shape;
        result << "_bType=" << param.B.type;
        result << "_bShape=" << param.B.shape;
        result << "_cType=" << param.C.type;
        result << "_cShape=" << param.C.shape;
        result << "_dType=" << param.D.type;
        result << "_dShape=" << param.D.shape;
        result << "_axis=" << param.axis;
        result << "_eType=" << param.expected.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expected.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_rShape=" << param.expected.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ConcatParamsInPlacePropagate2dTensor& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.A.type, params.A.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);
        const auto add1 = std::make_shared<op::v1::Add>(A, B);
        const auto C = std::make_shared<op::v0::Parameter>(params.C.type, params.C.shape);
        const auto D = std::make_shared<op::v0::Parameter>(params.D.type, params.D.shape);
        const auto add2 = std::make_shared<op::v1::Add>(C, D);
        const auto concat1 = std::make_shared<op::v0::Concat>(NodeVector{add1, add2}, params.axis);
        const auto subtract = std::make_shared<op::v1::Subtract>(C, A);
        const auto f =
            std::make_shared<Model>(std::make_shared<op::v0::Concat>(NodeVector{concat1, subtract}, params.axis),
                                    ParameterVector{A, B, C, D});
        return f;
    }
};

TEST_P(ReferenceConcatTestInPlacePropagate2dTensor, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ConcatParamsInPlacePropagate2dTensor> generateParamsInPlacePropagate2dTensor() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ConcatParamsInPlacePropagate2dTensor> params{
        ConcatParamsInPlacePropagate2dTensor(reference_tests::Tensor(ET, {1, 1}, std::vector<T>{1}),
                                             reference_tests::Tensor(ET, {1, 1}, std::vector<T>{2}),
                                             reference_tests::Tensor(ET, {1, 1}, std::vector<T>{3}),
                                             reference_tests::Tensor(ET, {1, 1}, std::vector<T>{4}),
                                             0,
                                             reference_tests::Tensor(ET, {3, 1}, std::vector<T>{3, 7, 2}),
                                             "concat_in_place_2d_tensor"),
    };
    return params;
}

std::vector<ConcatParamsInPlacePropagate2dTensor> generateCombinedParamsInPlacePropagate2dTensor() {
    const std::vector<std::vector<ConcatParamsInPlacePropagate2dTensor>> generatedParams{
        generateParamsInPlacePropagate2dTensor<element::Type_t::i32>(),
        generateParamsInPlacePropagate2dTensor<element::Type_t::i64>(),
        generateParamsInPlacePropagate2dTensor<element::Type_t::u32>(),
        generateParamsInPlacePropagate2dTensor<element::Type_t::u64>(),
        generateParamsInPlacePropagate2dTensor<element::Type_t::bf16>(),
        generateParamsInPlacePropagate2dTensor<element::Type_t::f16>(),
        generateParamsInPlacePropagate2dTensor<element::Type_t::f32>(),
    };
    std::vector<ConcatParamsInPlacePropagate2dTensor> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Concat_With_Hardcoded_Refs,
                         ReferenceConcatTestInPlacePropagate2dTensor,
                         testing::ValuesIn(generateCombinedParamsInPlacePropagate2dTensor()),
                         ReferenceConcatTestInPlacePropagate2dTensor::getTestCaseName);

//// concat_in_place_tree_1

struct ConcatParamsInPlaceTree1 {
    ConcatParamsInPlaceTree1(const reference_tests::Tensor& A,
                             const reference_tests::Tensor& B,
                             const int32_t axis,
                             const reference_tests::Tensor& expected,
                             const std::string& testcaseName = "")
        : A(A),
          B(B),
          axis(axis),
          expected(expected),
          testcaseName(testcaseName) {}

    reference_tests::Tensor A;
    reference_tests::Tensor B;
    int32_t axis;
    reference_tests::Tensor expected;
    std::string testcaseName;
};

class ReferenceConcatTestInPlaceTree1 : public testing::TestWithParam<ConcatParamsInPlaceTree1>,
                                        public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.A.data, params.B.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConcatParamsInPlaceTree1>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "aType=" << param.A.type;
        result << "_aShape=" << param.A.shape;
        result << "_bType=" << param.B.type;
        result << "_bShape=" << param.B.shape;
        result << "_axis=" << param.axis;
        result << "_eType=" << param.expected.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expected.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_rShape=" << param.expected.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ConcatParamsInPlaceTree1& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.A.type, params.A.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);
        const auto add1 = std::make_shared<op::v1::Add>(A, B);
        const auto add2 = std::make_shared<op::v1::Add>(A, B);
        const auto concat = std::make_shared<op::v0::Concat>(NodeVector{add1, add2}, params.axis);
        const auto f = std::make_shared<Model>(std::make_shared<op::v1::Add>(concat, concat), ParameterVector{A, B});
        return f;
    }
};

TEST_P(ReferenceConcatTestInPlaceTree1, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ConcatParamsInPlaceTree1> generateParamsInPlaceTree1() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ConcatParamsInPlaceTree1> params{
        ConcatParamsInPlaceTree1(reference_tests::Tensor(ET, {1, 2, 2}, std::vector<T>{1, 1, 1, 1}),
                                 reference_tests::Tensor(ET, {1, 2, 2}, std::vector<T>{1, 1, 1, 1}),
                                 1,
                                 reference_tests::Tensor(ET, {1, 4, 2}, std::vector<T>{4, 4, 4, 4, 4, 4, 4, 4}),
                                 "concat_in_place_tree_1"),
    };
    return params;
}

std::vector<ConcatParamsInPlaceTree1> generateCombinedParamsInPlaceTree1() {
    const std::vector<std::vector<ConcatParamsInPlaceTree1>> generatedParams{
        generateParamsInPlaceTree1<element::Type_t::i32>(),
        generateParamsInPlaceTree1<element::Type_t::i64>(),
        generateParamsInPlaceTree1<element::Type_t::u32>(),
        generateParamsInPlaceTree1<element::Type_t::u64>(),
        generateParamsInPlaceTree1<element::Type_t::bf16>(),
        generateParamsInPlaceTree1<element::Type_t::f16>(),
        generateParamsInPlaceTree1<element::Type_t::f32>(),
    };
    std::vector<ConcatParamsInPlaceTree1> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Concat_With_Hardcoded_Refs,
                         ReferenceConcatTestInPlaceTree1,
                         testing::ValuesIn(generateCombinedParamsInPlaceTree1()),
                         ReferenceConcatTestInPlaceTree1::getTestCaseName);

//// concat_in_place_tree_2

struct ConcatParamsInPlaceTree2 {
    ConcatParamsInPlaceTree2(const reference_tests::Tensor& A,
                             const reference_tests::Tensor& B,
                             const int32_t axis,
                             const reference_tests::Tensor& expected,
                             const std::string& testcaseName = "")
        : A(A),
          B(B),
          axis(axis),
          expected(expected),
          testcaseName(testcaseName) {}

    reference_tests::Tensor A;
    reference_tests::Tensor B;
    int32_t axis;
    reference_tests::Tensor expected;
    std::string testcaseName;
};

class ReferenceConcatTestInPlaceTree2 : public testing::TestWithParam<ConcatParamsInPlaceTree2>,
                                        public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.A.data, params.B.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConcatParamsInPlaceTree2>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "aType=" << param.A.type;
        result << "_aShape=" << param.A.shape;
        result << "_bType=" << param.B.type;
        result << "_bShape=" << param.B.shape;
        result << "_axis=" << param.axis;
        result << "_eType=" << param.expected.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expected.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_rShape=" << param.expected.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ConcatParamsInPlaceTree2& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.A.type, params.A.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);
        const auto add1 = std::make_shared<op::v1::Add>(A, B);
        const auto add2 = std::make_shared<op::v1::Add>(A, B);
        const auto concat1 = std::make_shared<op::v0::Concat>(NodeVector{add1, add2}, params.axis);
        const auto concat2 = std::make_shared<op::v0::Concat>(NodeVector{add1, add2}, params.axis);
        const auto concat12 = std::make_shared<op::v0::Concat>(NodeVector{concat1, concat2}, params.axis);
        const auto f =
            std::make_shared<Model>(std::make_shared<op::v1::Add>(concat12, concat12), ParameterVector{A, B});
        return f;
    }
};

TEST_P(ReferenceConcatTestInPlaceTree2, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ConcatParamsInPlaceTree2> generateParamsInPlaceTree2() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ConcatParamsInPlaceTree2> params{
        ConcatParamsInPlaceTree2(
            reference_tests::Tensor(ET, {1, 2, 2}, std::vector<T>{1, 1, 1, 1}),
            reference_tests::Tensor(ET, {1, 2, 2}, std::vector<T>{1, 1, 1, 1}),
            1,
            reference_tests::Tensor(ET, {1, 8, 2}, std::vector<T>{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}),
            "concat_in_place_tree_2"),
    };
    return params;
}

std::vector<ConcatParamsInPlaceTree2> generateCombinedParamsInPlaceTree2() {
    const std::vector<std::vector<ConcatParamsInPlaceTree2>> generatedParams{
        generateParamsInPlaceTree2<element::Type_t::i32>(),
        generateParamsInPlaceTree2<element::Type_t::i64>(),
        generateParamsInPlaceTree2<element::Type_t::u32>(),
        generateParamsInPlaceTree2<element::Type_t::u64>(),
        generateParamsInPlaceTree2<element::Type_t::bf16>(),
        generateParamsInPlaceTree2<element::Type_t::f16>(),
        generateParamsInPlaceTree2<element::Type_t::f32>(),
    };
    std::vector<ConcatParamsInPlaceTree2> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Concat_With_Hardcoded_Refs,
                         ReferenceConcatTestInPlaceTree2,
                         testing::ValuesIn(generateCombinedParamsInPlaceTree2()),
                         ReferenceConcatTestInPlaceTree2::getTestCaseName);

//// concat_in_place_tree_3

struct ConcatParamsInPlaceTree3 {
    ConcatParamsInPlaceTree3(const reference_tests::Tensor& A,
                             const reference_tests::Tensor& B,
                             const int32_t axis,
                             const reference_tests::Tensor& expected,
                             const std::string& testcaseName = "")
        : A(A),
          B(B),
          axis(axis),
          expected(expected),
          testcaseName(testcaseName) {}

    reference_tests::Tensor A;
    reference_tests::Tensor B;
    int32_t axis;
    reference_tests::Tensor expected;
    std::string testcaseName;
};

class ReferenceConcatTestInPlaceTree3 : public testing::TestWithParam<ConcatParamsInPlaceTree3>,
                                        public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.A.data, params.B.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConcatParamsInPlaceTree3>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "aType=" << param.A.type;
        result << "_aShape=" << param.A.shape;
        result << "_bType=" << param.B.type;
        result << "_bShape=" << param.B.shape;
        result << "_axis=" << param.axis;
        result << "_eType=" << param.expected.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expected.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_rShape=" << param.expected.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ConcatParamsInPlaceTree3& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.A.type, params.A.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);
        const auto concat1 = std::make_shared<op::v0::Concat>(NodeVector{A, B}, params.axis);
        const auto concat2 = std::make_shared<op::v0::Concat>(NodeVector{A, B}, params.axis);
        const auto concat3 = std::make_shared<op::v0::Concat>(NodeVector{A, B}, params.axis);
        const auto concat4 = std::make_shared<op::v0::Concat>(NodeVector{A, B}, params.axis);
        const auto concat12 = std::make_shared<op::v0::Concat>(NodeVector{concat1, concat2}, params.axis);
        const auto concat34 = std::make_shared<op::v0::Concat>(NodeVector{concat3, concat4}, params.axis);
        const auto concat14 = std::make_shared<op::v0::Concat>(NodeVector{concat12, concat34}, params.axis);
        const auto f =
            std::make_shared<Model>(std::make_shared<op::v1::Add>(concat14, concat14), ParameterVector{A, B});
        return f;
    }
};

TEST_P(ReferenceConcatTestInPlaceTree3, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ConcatParamsInPlaceTree3> generateParamsInPlaceTree3() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ConcatParamsInPlaceTree3> params{
        ConcatParamsInPlaceTree3(
            reference_tests::Tensor(ET, {1, 2, 2}, std::vector<T>{1, 1, 1, 1}),
            reference_tests::Tensor(ET, {1, 2, 2}, std::vector<T>{1, 1, 1, 1}),
            1,
            reference_tests::Tensor(ET, {1, 16, 2}, std::vector<T>{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                                                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}),
            "concat_in_place_tree_3"),
    };
    return params;
}

std::vector<ConcatParamsInPlaceTree3> generateCombinedParamsInPlaceTree3() {
    const std::vector<std::vector<ConcatParamsInPlaceTree3>> generatedParams{
        generateParamsInPlaceTree3<element::Type_t::i32>(),
        generateParamsInPlaceTree3<element::Type_t::i64>(),
        generateParamsInPlaceTree3<element::Type_t::u32>(),
        generateParamsInPlaceTree3<element::Type_t::u64>(),
        generateParamsInPlaceTree3<element::Type_t::bf16>(),
        generateParamsInPlaceTree3<element::Type_t::f16>(),
        generateParamsInPlaceTree3<element::Type_t::f32>(),
    };
    std::vector<ConcatParamsInPlaceTree3> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Concat_With_Hardcoded_Refs,
                         ReferenceConcatTestInPlaceTree3,
                         testing::ValuesIn(generateCombinedParamsInPlaceTree3()),
                         ReferenceConcatTestInPlaceTree3::getTestCaseName);

//// concat_in_place_add_concat

struct ConcatParamsInPlaceAddConcat {
    ConcatParamsInPlaceAddConcat(const reference_tests::Tensor& A,
                                 const reference_tests::Tensor& B,
                                 const int32_t axis,
                                 const reference_tests::Tensor& expected,
                                 const std::string& testcaseName = "")
        : A(A),
          B(B),
          axis(axis),
          expected(expected),
          testcaseName(testcaseName) {}

    reference_tests::Tensor A;
    reference_tests::Tensor B;
    int32_t axis;
    reference_tests::Tensor expected;
    std::string testcaseName;
};

class ReferenceConcatTestInPlaceAddConcat : public testing::TestWithParam<ConcatParamsInPlaceAddConcat>,
                                            public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.A.data, params.B.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConcatParamsInPlaceAddConcat>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "aType=" << param.A.type;
        result << "_aShape=" << param.A.shape;
        result << "_bType=" << param.B.type;
        result << "_bShape=" << param.B.shape;
        result << "_axis=" << param.axis;
        result << "_eType=" << param.expected.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expected.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_rShape=" << param.expected.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ConcatParamsInPlaceAddConcat& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.A.type, params.A.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);
        const auto add1 = std::make_shared<op::v1::Add>(A, B);
        const auto add2 = std::make_shared<op::v1::Add>(add1, add1);
        const auto concat = std::make_shared<op::v0::Concat>(NodeVector{add1, add2}, params.axis);
        const auto add3 = std::make_shared<op::v1::Add>(concat, concat);
        const auto f = std::make_shared<Model>(add3, ParameterVector{A, B});
        return f;
    }
};

TEST_P(ReferenceConcatTestInPlaceAddConcat, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ConcatParamsInPlaceAddConcat> generateParamsInPlaceAddConcat() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ConcatParamsInPlaceAddConcat> params{
        ConcatParamsInPlaceAddConcat(reference_tests::Tensor(ET, {2, 2}, std::vector<T>{1, 1, 1, 1}),
                                     reference_tests::Tensor(ET, {2, 2}, std::vector<T>{1, 1, 1, 1}),
                                     0,
                                     reference_tests::Tensor(ET, {4, 2}, std::vector<T>{4, 4, 4, 4, 8, 8, 8, 8}),
                                     "concat_in_place_add_concat"),
    };
    return params;
}

std::vector<ConcatParamsInPlaceAddConcat> generateCombinedParamsInPlaceAddConcat() {
    const std::vector<std::vector<ConcatParamsInPlaceAddConcat>> generatedParams{
        generateParamsInPlaceAddConcat<element::Type_t::i32>(),
        generateParamsInPlaceAddConcat<element::Type_t::i64>(),
        generateParamsInPlaceAddConcat<element::Type_t::u32>(),
        generateParamsInPlaceAddConcat<element::Type_t::u64>(),
        generateParamsInPlaceAddConcat<element::Type_t::bf16>(),
        generateParamsInPlaceAddConcat<element::Type_t::f16>(),
        generateParamsInPlaceAddConcat<element::Type_t::f32>(),
    };
    std::vector<ConcatParamsInPlaceAddConcat> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Concat_With_Hardcoded_Refs,
                         ReferenceConcatTestInPlaceAddConcat,
                         testing::ValuesIn(generateCombinedParamsInPlaceAddConcat()),
                         ReferenceConcatTestInPlaceAddConcat::getTestCaseName);

//// concat_in_place_add_concat_2

struct ConcatParamsInPlaceAddConcat2 {
    ConcatParamsInPlaceAddConcat2(const reference_tests::Tensor& A,
                                  const reference_tests::Tensor& B,
                                  const int32_t axis,
                                  const reference_tests::Tensor& expected,
                                  const std::string& testcaseName = "")
        : A(A),
          B(B),
          axis(axis),
          expected(expected),
          testcaseName(testcaseName) {}

    reference_tests::Tensor A;
    reference_tests::Tensor B;
    int32_t axis;
    reference_tests::Tensor expected;
    std::string testcaseName;
};

class ReferenceConcatTestInPlaceAddConcat2 : public testing::TestWithParam<ConcatParamsInPlaceAddConcat2>,
                                             public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.A.data, params.B.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConcatParamsInPlaceAddConcat2>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "aType=" << param.A.type;
        result << "_aShape=" << param.A.shape;
        result << "_bType=" << param.B.type;
        result << "_bShape=" << param.B.shape;
        result << "_axis=" << param.axis;
        result << "_eType=" << param.expected.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expected.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_rShape=" << param.expected.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ConcatParamsInPlaceAddConcat2& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.A.type, params.A.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);
        const auto add1 = std::make_shared<op::v1::Add>(A, B);
        const auto add2 = std::make_shared<op::v1::Add>(A, B);
        const auto add3 = std::make_shared<op::v1::Add>(A, B);
        const auto add4 = std::make_shared<op::v1::Add>(A, B);
        const auto add5 = std::make_shared<op::v1::Add>(A, B);
        const auto concat1 = std::make_shared<op::v0::Concat>(NodeVector{add1, add2, add3}, params.axis);
        const auto concat2 = std::make_shared<op::v0::Concat>(NodeVector{add4, add2, add5}, params.axis);
        const auto add6 = std::make_shared<op::v1::Add>(concat1, concat2);
        const auto f = std::make_shared<Model>(add6, ParameterVector{A, B});
        return f;
    }
};

TEST_P(ReferenceConcatTestInPlaceAddConcat2, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ConcatParamsInPlaceAddConcat2> generateParamsInPlaceAddConcat2() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ConcatParamsInPlaceAddConcat2> params{
        ConcatParamsInPlaceAddConcat2(
            reference_tests::Tensor(ET, {1, 2, 2}, std::vector<T>{1, 1, 1, 1}),
            reference_tests::Tensor(ET, {1, 2, 2}, std::vector<T>{1, 1, 1, 1}),
            1,
            reference_tests::Tensor(ET, {1, 6, 2}, std::vector<T>{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}),
            "concat_in_place_add_concat_2"),
    };
    return params;
}

std::vector<ConcatParamsInPlaceAddConcat2> generateCombinedParamsInPlaceAddConcat2() {
    const std::vector<std::vector<ConcatParamsInPlaceAddConcat2>> generatedParams{
        generateParamsInPlaceAddConcat2<element::Type_t::i32>(),
        generateParamsInPlaceAddConcat2<element::Type_t::i64>(),
        generateParamsInPlaceAddConcat2<element::Type_t::u32>(),
        generateParamsInPlaceAddConcat2<element::Type_t::u64>(),
        generateParamsInPlaceAddConcat2<element::Type_t::bf16>(),
        generateParamsInPlaceAddConcat2<element::Type_t::f16>(),
        generateParamsInPlaceAddConcat2<element::Type_t::f32>(),
    };
    std::vector<ConcatParamsInPlaceAddConcat2> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Concat_With_Hardcoded_Refs,
                         ReferenceConcatTestInPlaceAddConcat2,
                         testing::ValuesIn(generateCombinedParamsInPlaceAddConcat2()),
                         ReferenceConcatTestInPlaceAddConcat2::getTestCaseName);

//// concat_5d

struct ConcatParams5d {
    ConcatParams5d(const reference_tests::Tensor& A,
                   const reference_tests::Tensor& B,
                   const reference_tests::Tensor& C,
                   const int32_t axis,
                   const reference_tests::Tensor& expected,
                   const std::string& testcaseName = "")
        : A(A),
          B(B),
          C(C),
          axis(axis),
          expected(expected),
          testcaseName(testcaseName) {}

    reference_tests::Tensor A;
    reference_tests::Tensor B;
    reference_tests::Tensor C;
    int32_t axis;
    reference_tests::Tensor expected;
    std::string testcaseName;
};

class ReferenceConcatTest5d : public testing::TestWithParam<ConcatParams5d>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.A.data, params.B.data, params.C.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConcatParams5d>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "aType=" << param.A.type;
        result << "_aShape=" << param.A.shape;
        result << "_bType=" << param.B.type;
        result << "_bShape=" << param.B.shape;
        result << "_cType=" << param.C.type;
        result << "_cShape=" << param.C.shape;
        result << "_axis=" << param.axis;
        result << "_eType=" << param.expected.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expected.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_rShape=" << param.expected.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ConcatParams5d& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.A.type, params.A.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);
        const auto C = std::make_shared<op::v0::Parameter>(params.C.type, params.C.shape);
        const auto concat = std::make_shared<op::v0::Concat>(NodeVector{A, B, C}, params.axis);
        const auto f = std::make_shared<Model>(concat, ParameterVector{A, B, C});
        return f;
    }
};

TEST_P(ReferenceConcatTest5d, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ConcatParams5d> generateParams5d() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ConcatParams5d> params{
        ConcatParams5d(
            reference_tests::Tensor(ET,
                                    {2, 3, 4, 3, 2},
                                    []() -> std::vector<T> {
                                        std::vector<T> data(2 * 3 * 4 * 3 * 2);
                                        for (int i = 0; i < 2 * 3 * 4 * 3 * 2; i++) {
                                            data[i] = static_cast<T>(i + 1);
                                        }
                                        return data;
                                    }()),
            reference_tests::Tensor(ET,
                                    {2, 3, 3, 3, 2},
                                    []() -> std::vector<T> {
                                        std::vector<T> data(2 * 3 * 3 * 3 * 2);
                                        for (int i = 0; i < 2 * 3 * 3 * 3 * 2; i++) {
                                            data[i] = 1000 + static_cast<T>(i + 1);
                                        }
                                        return data;
                                    }()),
            reference_tests::Tensor(ET,
                                    {2, 3, 2, 3, 2},
                                    []() -> std::vector<T> {
                                        std::vector<T> data(2 * 3 * 2 * 3 * 2);
                                        for (int i = 0; i < 2 * 3 * 2 * 3 * 2; i++) {
                                            data[i] = 2000 + static_cast<T>(i + 1);
                                        }
                                        return data;
                                    }()),
            2,
            reference_tests::Tensor(
                ET,
                {2, 3, 9, 3, 2},
                std::vector<T>{
                    1.,    2.,    3.,    4.,    5.,    6.,    7.,    8.,    9.,    10.,   11.,   12.,   13.,   14.,
                    15.,   16.,   17.,   18.,   19.,   20.,   21.,   22.,   23.,   24.,   1001., 1002., 1003., 1004.,
                    1005., 1006., 1007., 1008., 1009., 1010., 1011., 1012., 1013., 1014., 1015., 1016., 1017., 1018.,
                    2001., 2002., 2003., 2004., 2005., 2006., 2007., 2008., 2009., 2010., 2011., 2012., 25.,   26.,
                    27.,   28.,   29.,   30.,   31.,   32.,   33.,   34.,   35.,   36.,   37.,   38.,   39.,   40.,
                    41.,   42.,   43.,   44.,   45.,   46.,   47.,   48.,   1019., 1020., 1021., 1022., 1023., 1024.,
                    1025., 1026., 1027., 1028., 1029., 1030., 1031., 1032., 1033., 1034., 1035., 1036., 2013., 2014.,
                    2015., 2016., 2017., 2018., 2019., 2020., 2021., 2022., 2023., 2024., 49.,   50.,   51.,   52.,
                    53.,   54.,   55.,   56.,   57.,   58.,   59.,   60.,   61.,   62.,   63.,   64.,   65.,   66.,
                    67.,   68.,   69.,   70.,   71.,   72.,   1037., 1038., 1039., 1040., 1041., 1042., 1043., 1044.,
                    1045., 1046., 1047., 1048., 1049., 1050., 1051., 1052., 1053., 1054., 2025., 2026., 2027., 2028.,
                    2029., 2030., 2031., 2032., 2033., 2034., 2035., 2036., 73.,   74.,   75.,   76.,   77.,   78.,
                    79.,   80.,   81.,   82.,   83.,   84.,   85.,   86.,   87.,   88.,   89.,   90.,   91.,   92.,
                    93.,   94.,   95.,   96.,   1055., 1056., 1057., 1058., 1059., 1060., 1061., 1062., 1063., 1064.,
                    1065., 1066., 1067., 1068., 1069., 1070., 1071., 1072., 2037., 2038., 2039., 2040., 2041., 2042.,
                    2043., 2044., 2045., 2046., 2047., 2048., 97.,   98.,   99.,   100.,  101.,  102.,  103.,  104.,
                    105.,  106.,  107.,  108.,  109.,  110.,  111.,  112.,  113.,  114.,  115.,  116.,  117.,  118.,
                    119.,  120.,  1073., 1074., 1075., 1076., 1077., 1078., 1079., 1080., 1081., 1082., 1083., 1084.,
                    1085., 1086., 1087., 1088., 1089., 1090., 2049., 2050., 2051., 2052., 2053., 2054., 2055., 2056.,
                    2057., 2058., 2059., 2060., 121.,  122.,  123.,  124.,  125.,  126.,  127.,  128.,  129.,  130.,
                    131.,  132.,  133.,  134.,  135.,  136.,  137.,  138.,  139.,  140.,  141.,  142.,  143.,  144.,
                    1091., 1092., 1093., 1094., 1095., 1096., 1097., 1098., 1099., 1100., 1101., 1102., 1103., 1104.,
                    1105., 1106., 1107., 1108., 2061., 2062., 2063., 2064., 2065., 2066., 2067., 2068., 2069., 2070.,
                    2071., 2072.}),
            "concat_5d"),
    };
    return params;
}

std::vector<ConcatParams5d> generateCombinedParams5d() {
    const std::vector<std::vector<ConcatParams5d>> generatedParams{
        generateParams5d<element::Type_t::bf16>(),
        generateParams5d<element::Type_t::f16>(),
        generateParams5d<element::Type_t::f32>(),
    };
    std::vector<ConcatParams5d> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Concat_With_Hardcoded_Refs,
                         ReferenceConcatTest5d,
                         testing::ValuesIn(generateCombinedParams5d()),
                         ReferenceConcatTest5d::getTestCaseName);

//// concat_zero_length_1d_last

struct ConcatParamsZeroLength1dLast {
    ConcatParamsZeroLength1dLast(const reference_tests::Tensor& A,
                                 const reference_tests::Tensor& B,
                                 const int32_t axis,
                                 const reference_tests::Tensor& expected,
                                 const std::string& testcaseName = "")
        : A(A),
          B(B),
          axis(axis),
          expected(expected),
          testcaseName(testcaseName) {}

    reference_tests::Tensor A;
    reference_tests::Tensor B;
    int32_t axis;
    reference_tests::Tensor expected;
    std::string testcaseName;
};

class ReferenceConcatTestZeroLength1dLast : public testing::TestWithParam<ConcatParamsZeroLength1dLast>,
                                            public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.A.data, params.B.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConcatParamsZeroLength1dLast>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "aType=" << param.A.type;
        result << "_aShape=" << param.A.shape;
        result << "_bType=" << param.B.type;
        result << "_bShape=" << param.B.shape;
        result << "_axis=" << param.axis;
        result << "_eType=" << param.expected.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expected.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_rShape=" << param.expected.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ConcatParamsZeroLength1dLast& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.A.type, params.A.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);
        const auto concat = std::make_shared<op::v0::Concat>(NodeVector{A, B}, params.axis);
        const auto f = std::make_shared<Model>(concat, ParameterVector{A, B});
        return f;
    }
};

TEST_P(ReferenceConcatTestZeroLength1dLast, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ConcatParamsZeroLength1dLast> generateParamsZeroLength1dLast() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ConcatParamsZeroLength1dLast> params{
        ConcatParamsZeroLength1dLast(reference_tests::Tensor(ET, {4}, std::vector<T>{1, 2, 3, 4}),
                                     reference_tests::Tensor(ET, {0}, std::vector<T>{0}),
                                     0,
                                     reference_tests::Tensor(ET, {4}, std::vector<T>{1, 2, 3, 4}),
                                     "concat_zero_length_1d_last"),
    };
    return params;
}

std::vector<ConcatParamsZeroLength1dLast> generateCombinedParamsZeroLength1dLast() {
    const std::vector<std::vector<ConcatParamsZeroLength1dLast>> generatedParams{
        generateParamsZeroLength1dLast<element::Type_t::i32>(),
        generateParamsZeroLength1dLast<element::Type_t::i64>(),
        generateParamsZeroLength1dLast<element::Type_t::u32>(),
        generateParamsZeroLength1dLast<element::Type_t::u64>(),
        generateParamsZeroLength1dLast<element::Type_t::bf16>(),
        generateParamsZeroLength1dLast<element::Type_t::f16>(),
        generateParamsZeroLength1dLast<element::Type_t::f32>(),
    };
    std::vector<ConcatParamsZeroLength1dLast> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Concat_With_Hardcoded_Refs,
                         ReferenceConcatTestZeroLength1dLast,
                         testing::ValuesIn(generateCombinedParamsZeroLength1dLast()),
                         ReferenceConcatTestZeroLength1dLast::getTestCaseName);

//// concat_zero_length_1d_middle

struct ConcatParamsZeroLength1dMiddle {
    ConcatParamsZeroLength1dMiddle(const reference_tests::Tensor& A,
                                   const reference_tests::Tensor& B,
                                   const reference_tests::Tensor& C,
                                   const int32_t axis,
                                   const reference_tests::Tensor& expected,
                                   const std::string& testcaseName = "")
        : A(A),
          B(B),
          C(C),
          axis(axis),
          expected(expected),
          testcaseName(testcaseName) {}

    reference_tests::Tensor A;
    reference_tests::Tensor B;
    reference_tests::Tensor C;
    int32_t axis;
    reference_tests::Tensor expected;
    std::string testcaseName;
};

class ReferenceConcatTestZeroLength1dMiddle : public testing::TestWithParam<ConcatParamsZeroLength1dMiddle>,
                                              public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.A.data, params.B.data, params.C.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConcatParamsZeroLength1dMiddle>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "aType=" << param.A.type;
        result << "_aShape=" << param.A.shape;
        result << "_bType=" << param.B.type;
        result << "_bShape=" << param.B.shape;
        result << "_cType=" << param.C.type;
        result << "_cShape=" << param.C.shape;
        result << "_axis=" << param.axis;
        result << "_eType=" << param.expected.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expected.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_rShape=" << param.expected.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ConcatParamsZeroLength1dMiddle& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.A.type, params.A.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);
        const auto C = std::make_shared<op::v0::Parameter>(params.C.type, params.C.shape);
        const auto concat = std::make_shared<op::v0::Concat>(NodeVector{A, B, C}, params.axis);
        const auto f = std::make_shared<Model>(concat, ParameterVector{A, B, C});
        return f;
    }
};

TEST_P(ReferenceConcatTestZeroLength1dMiddle, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ConcatParamsZeroLength1dMiddle> generateParamsZeroLength1dMiddle() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ConcatParamsZeroLength1dMiddle> params{
        ConcatParamsZeroLength1dMiddle(reference_tests::Tensor(ET, {4}, std::vector<T>{1, 2, 3, 4}),
                                       reference_tests::Tensor(ET, {0}, std::vector<T>{0}),
                                       reference_tests::Tensor(ET, {4}, std::vector<T>{5, 6, 7, 8}),
                                       0,
                                       reference_tests::Tensor(ET, {8}, std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8}),
                                       "concat_zero_length_1d_middle"),
    };
    return params;
}

std::vector<ConcatParamsZeroLength1dMiddle> generateCombinedParamsZeroLength1dMiddle() {
    const std::vector<std::vector<ConcatParamsZeroLength1dMiddle>> generatedParams{
        generateParamsZeroLength1dMiddle<element::Type_t::i32>(),
        generateParamsZeroLength1dMiddle<element::Type_t::i64>(),
        generateParamsZeroLength1dMiddle<element::Type_t::u32>(),
        generateParamsZeroLength1dMiddle<element::Type_t::u64>(),
        generateParamsZeroLength1dMiddle<element::Type_t::bf16>(),
        generateParamsZeroLength1dMiddle<element::Type_t::f16>(),
        generateParamsZeroLength1dMiddle<element::Type_t::f32>(),
    };
    std::vector<ConcatParamsZeroLength1dMiddle> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Concat_With_Hardcoded_Refs,
                         ReferenceConcatTestZeroLength1dMiddle,
                         testing::ValuesIn(generateCombinedParamsZeroLength1dMiddle()),
                         ReferenceConcatTestZeroLength1dMiddle::getTestCaseName);

//// concat_zero_zero

struct ConcatParamsZeroZero {
    ConcatParamsZeroZero(const reference_tests::Tensor& A,
                         const int32_t axis,
                         const reference_tests::Tensor& expected,
                         const std::string& testcaseName = "")
        : A(A),
          axis(axis),
          expected(expected),
          testcaseName(testcaseName) {}

    reference_tests::Tensor A;
    int32_t axis;
    reference_tests::Tensor expected;
    std::string testcaseName;
};

class ReferenceConcatTestZeroZero : public testing::TestWithParam<ConcatParamsZeroZero>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.A.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConcatParamsZeroZero>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "aType=" << param.A.type;
        result << "_aShape=" << param.A.shape;
        result << "_axis=" << param.axis;
        result << "_eType=" << param.expected.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expected.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_rShape=" << param.expected.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ConcatParamsZeroZero& params) {
        const auto constant_1 = std::make_shared<op::v0::Constant>(params.A.type, params.A.shape, params.A.data.data());
        const auto concat_1 = std::make_shared<op::v0::Concat>(NodeVector{constant_1, constant_1}, params.axis);
        const auto f = std::make_shared<Model>(concat_1, ParameterVector{});
        return f;
    }
};

TEST_P(ReferenceConcatTestZeroZero, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ConcatParamsZeroZero> generateParamsZeroZero() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ConcatParamsZeroZero> params{
        ConcatParamsZeroZero(reference_tests::Tensor(ET, {0}, std::vector<T>{1}),
                             0,
                             reference_tests::Tensor(ET, {0}, std::vector<T>{}),
                             "concat_zero_zero"),
    };
    return params;
}

std::vector<ConcatParamsZeroZero> generateCombinedParamsZeroZero() {
    const std::vector<std::vector<ConcatParamsZeroZero>> generatedParams{
        generateParamsZeroZero<element::Type_t::i32>(),
        generateParamsZeroZero<element::Type_t::i64>(),
        generateParamsZeroZero<element::Type_t::u32>(),
        generateParamsZeroZero<element::Type_t::u64>(),
        generateParamsZeroZero<element::Type_t::bf16>(),
        generateParamsZeroZero<element::Type_t::f16>(),
        generateParamsZeroZero<element::Type_t::f32>(),
    };
    std::vector<ConcatParamsZeroZero> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Concat_With_Hardcoded_Refs,
                         ReferenceConcatTestZeroZero,
                         testing::ValuesIn(generateCombinedParamsZeroZero()),
                         ReferenceConcatTestZeroZero::getTestCaseName);

//// concat_zero_length_4d_middle

struct ConcatParamsZeroLength4dMiddle {
    ConcatParamsZeroLength4dMiddle(const reference_tests::Tensor& A,
                                   const reference_tests::Tensor& B,
                                   const reference_tests::Tensor& C,
                                   const int32_t axis,
                                   const reference_tests::Tensor& expected,
                                   const std::string& testcaseName = "")
        : A(A),
          B(B),
          C(C),
          axis(axis),
          expected(expected),
          testcaseName(testcaseName) {}

    reference_tests::Tensor A;
    reference_tests::Tensor B;
    reference_tests::Tensor C;
    int32_t axis;
    reference_tests::Tensor expected;
    std::string testcaseName;
};

class ReferenceConcatTestZeroLength4dMiddle : public testing::TestWithParam<ConcatParamsZeroLength4dMiddle>,
                                              public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.A.data, params.B.data, params.C.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConcatParamsZeroLength4dMiddle>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "aType=" << param.A.type;
        result << "_aShape=" << param.A.shape;
        result << "_bType=" << param.B.type;
        result << "_bShape=" << param.B.shape;
        result << "_cType=" << param.C.type;
        result << "_cShape=" << param.C.shape;
        result << "_axis=" << param.axis;
        result << "_eType=" << param.expected.type;
        if (param.testcaseName != "") {
            result << "_eShape=" << param.expected.shape;
            result << "_=" << param.testcaseName;
        } else {
            result << "_rShape=" << param.expected.shape;
        }
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ConcatParamsZeroLength4dMiddle& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.A.type, params.A.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.B.type, params.B.shape);
        const auto C = std::make_shared<op::v0::Parameter>(params.C.type, params.C.shape);
        const auto concat = std::make_shared<op::v0::Concat>(NodeVector{A, B, C}, params.axis);
        const auto f = std::make_shared<Model>(concat, ParameterVector{A, B, C});
        return f;
    }
};

TEST_P(ReferenceConcatTestZeroLength4dMiddle, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<ConcatParamsZeroLength4dMiddle> generateParamsZeroLength4dMiddle() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<ConcatParamsZeroLength4dMiddle> params{
        ConcatParamsZeroLength4dMiddle(
            reference_tests::Tensor(ET, {2, 2, 1, 1}, std::vector<T>{1, 2, 3, 4}),
            reference_tests::Tensor(ET, {2, 2, 0, 1}, std::vector<T>{0}),
            reference_tests::Tensor(ET, {2, 2, 1, 1}, std::vector<T>{5, 6, 7, 8}),
            2,
            reference_tests::Tensor(ET, {2, 2, 2, 1}, std::vector<T>{1, 5, 2, 6, 3, 7, 4, 8}),
            "concat_zero_length_4d_middle"),
    };
    return params;
}

std::vector<ConcatParamsZeroLength4dMiddle> generateCombinedParamsZeroLength4dMiddle() {
    const std::vector<std::vector<ConcatParamsZeroLength4dMiddle>> generatedParams{
        generateParamsZeroLength4dMiddle<element::Type_t::i32>(),
        generateParamsZeroLength4dMiddle<element::Type_t::i64>(),
        generateParamsZeroLength4dMiddle<element::Type_t::u32>(),
        generateParamsZeroLength4dMiddle<element::Type_t::u64>(),
        generateParamsZeroLength4dMiddle<element::Type_t::bf16>(),
        generateParamsZeroLength4dMiddle<element::Type_t::f16>(),
        generateParamsZeroLength4dMiddle<element::Type_t::f32>(),
    };
    std::vector<ConcatParamsZeroLength4dMiddle> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Concat_With_Hardcoded_Refs,
                         ReferenceConcatTestZeroLength4dMiddle,
                         testing::ValuesIn(generateCombinedParamsZeroLength4dMiddle()),
                         ReferenceConcatTestZeroLength4dMiddle::getTestCaseName);

}  // namespace
