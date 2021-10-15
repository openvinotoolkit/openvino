// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <random>
#include "openvino/op/constant.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/equal.hpp"
#include "base_reference_test.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ConstantParams {
    template <class IT, class OT>
    ConstantParams(const PartialShape& inputShape,
                   const element::Type& inType, const element::Type& refType,
                   const std::vector<IT>& inputData, const std::vector<OT>& refData,
                   const std::string& test_name = "")
        : inputShape(inputShape),
          inType(inType),
          refType(refType),
          inputData(CreateTensor(inType, inputData)),
          refData(CreateTensor(refType, refData)),
          testcaseName(test_name) {}

    PartialShape inputShape;
    element::Type inType;
    element::Type refType;
    runtime::Tensor inputData;
    runtime::Tensor refData;
    std::string testcaseName;
};

class ReferenceConstantLayerTest : public testing::TestWithParam<ConstantParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        if (params.testcaseName == "tensor_constant"                ||
                params.testcaseName == "scalar_constant_float32"    ||
                params.testcaseName == "scalar_constant_int64"      ||
                params.testcaseName == "tensor_constant_float32"    ||
                params.testcaseName == "tensor_constant_int64"      ||
                params.testcaseName == "constant_equality_u4_2x2x3" ||
                params.testcaseName == "constant_equality_u4_1x3"   ||
                params.testcaseName == "constant_equality_u1_1x10"  ||
                params.testcaseName == "constant_equality_i4_2x2x3" ||
                params.testcaseName == "constant_equality_i4_1x3") {
            function = CreateFunction_Default(params.inputShape, params.inType, params.inputData);
            inputData = {};
            refOutData = {params.refData};
        } else if (params.testcaseName == "tensor_2constant") {
            function = CreateFunction_2Constant(params.inputShape, params.inType, params.inputData);
            inputData = {};
            refOutData = {params.refData, params.refData};
        } else if (params.testcaseName == "tensor_constant_with_op") {
            function = CreateFunction_With_Op(params.inputShape, params.inType, params.inputData);
            inputData = {};
            refOutData = {params.refData};
        } else if (params.testcaseName == "constant_multi_use") {
            function = CreateFunction_Multi_Use(params.inputShape, params.inType, params.inputData);
            inputData = {};
            refOutData = {params.refData};
        } else if (params.testcaseName == "constant_equality_bool") {
            function = CreateFunction_Equality_Bool(params.inputShape, params.inType, params.inputData);
            inputData = {};
            refOutData = {params.refData};
        } else {
            IE_THROW() << "This test is not supported";
        }
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ConstantParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
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
    static std::shared_ptr<Function> CreateFunction_Default(const PartialShape& input_shape, const element::Type& input_type,
                                                    const runtime::Tensor& constant_tensor) {
        auto A = op::v0::Constant::create(input_type, input_shape.to_shape(), constant_tensor.data());
        return std::make_shared<Function>(A, ParameterVector{});
    }

    static std::shared_ptr<Function> CreateFunction_2Constant(const PartialShape& input_shape, const element::Type& input_type,
                                                    const runtime::Tensor& constant_tensor) {
        auto A = op::v0::Constant::create(input_type, input_shape.to_shape(), constant_tensor.data());
        auto B = op::v0::Constant::create(input_type, input_shape.to_shape(), constant_tensor.data());
        return std::make_shared<Function>(NodeVector{A, B}, ParameterVector{});
    }

    static std::shared_ptr<Function> CreateFunction_With_Op(const PartialShape& input_shape, const element::Type& input_type,
                                                    const runtime::Tensor& constant_tensor) {
        auto A = op::v0::Constant::create(input_type, input_shape.to_shape(), constant_tensor.data());
        return std::make_shared<Function>(std::make_shared<op::v0::Abs>(A), ParameterVector{});
    }

    static std::shared_ptr<Function> CreateFunction_Multi_Use(const PartialShape& input_shape, const element::Type& input_type,
                                                    const runtime::Tensor& constant_tensor) {
        const auto A = std::make_shared<op::v0::Constant>(
            input_type,
            input_shape.to_shape(),
            std::vector<std::string>{std::to_string(*reinterpret_cast<int*>(constant_tensor.data()))});
        return std::make_shared<Function>(A, ParameterVector {});
    }

    static std::shared_ptr<Function> CreateFunction_Equality_Bool(const PartialShape& input_shape, const element::Type& input_type,
                                                    const runtime::Tensor& constant_tensor) {
        auto A = op::v0::Constant::create(input_type, input_shape.to_shape(), constant_tensor.data());
        auto B = op::v0::Constant::create(input_type, input_shape.to_shape(), {true, true, true, true});
        return std::make_shared<Function>(std::make_shared<op::v1::Equal>(A, B), ParameterVector{});
    }
};

TEST_P(ReferenceConstantLayerTest, CompareWithHardcodedRefs) {
    Exec();
    if (this->GetParam().testcaseName == "constant_multi_use") {
        Infer();
        Validate();
    }
}

template <element::Type_t IET, element::Type_t OET>
ConstantParams generateConstantParams(const PartialShape& staticShape,
                                      const std::vector<typename element_type_traits<IET>::value_type>& input,
                                      const std::vector<typename element_type_traits<OET>::value_type>& expected,
                                      const std::string& test_name = "") {
    return ConstantParams(staticShape, IET, OET, input, expected, test_name);
}

std::vector<ConstantParams> generateConstantCombinedParams() {
    const std::vector<ConstantParams> constantTypeParams {
        generateConstantParams<element::Type_t::f32, element::Type_t::f32>(
            {2, 2, 2},
            {1, 2, 3, 4, 5, 6, 7, 8},
            {1, 2, 3, 4, 5, 6, 7, 8},
            "tensor_constant"),
        generateConstantParams<element::Type_t::f32, element::Type_t::f32>(
            {2, 2, 2},
            {1, 2, 3, 4, 5, 6, 7, 8},
            {1, 2, 3, 4, 5, 6, 7, 8},
            "tensor_2constant"),
        generateConstantParams<element::Type_t::f32, element::Type_t::f32>(
            {2, 2, 2},
            {-1, 2, 3, -4, 5, -6, -7, 8},
            {1, 2, 3, 4, 5, 6, 7, 8},
            "tensor_constant_with_op"),
        generateConstantParams<element::Type_t::i32, element::Type_t::i32>(
            {},
            {388},
            {388},
            "constant_multi_use"),
        generateConstantParams<element::Type_t::f32, element::Type_t::f32>(
            {},
            {4.75},
            {4.75f},
            "scalar_constant_float32"),
        generateConstantParams<element::Type_t::i64, element::Type_t::i64>(
            {},
            {0x4000000000000001},
            {0x4000000000000001},
            "scalar_constant_int64"),
        generateConstantParams<element::Type_t::f32, element::Type_t::f32>(
            {2, 2},
            {4.75, 4.5, -5.25, 0.0},
            {4.75f, 4.5f, -5.25f, 0.0f},
            "tensor_constant_float32"),
        generateConstantParams<element::Type_t::i64, element::Type_t::i64>(
            {2},
            {0x4000000000000001, 0x4000000000000002},
            {0x4000000000000001, 0x4000000000000002},
            "tensor_constant_int64"),
        generateConstantParams<element::Type_t::boolean, element::Type_t::boolean>(
            {4},
            {true, false, true, false},
            {true, false, true, false},
            "constant_equality_bool"),
        generateConstantParams<element::Type_t::u4, element::Type_t::u4>(
            {2, 2, 3},
            {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xF, 0xF},
            {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xF, 0xF},
            "constant_equality_u4_2x2x3"),
        generateConstantParams<element::Type_t::u4, element::Type_t::u4>(
            {1, 3},
            {0x1, 0x2, 0x3},
            {0x1, 0x2, 0x3},
            "constant_equality_u4_1x3"),
        generateConstantParams<element::Type_t::u1, element::Type_t::u1>(
            {1, 10},
            {0x0, 0x0, 0x0, 0x1, 0x0, 0x0, 0x1, 0x0, 0x0, 0x0},
            {0x0, 0x0, 0x0, 0x1, 0x0, 0x0, 0x1, 0x0, 0x0, 0x0},
            "constant_equality_u1_1x10"),
        generateConstantParams<element::Type_t::i4, element::Type_t::i4>(
            {2, 2, 3},
            {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xF, 0xF},
            {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xF, 0xF},
            "constant_equality_i4_2x2x3"),
        generateConstantParams<element::Type_t::i4, element::Type_t::i4>(
            {1, 3},
            {0x1, 0x2, 0x3},
            {0x1, 0x2, 0x3},
            "constant_equality_i4_1x3"),
    };
    return constantTypeParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Constant_With_Hardcoded_Refs, ReferenceConstantLayerTest,
    testing::ValuesIn(generateConstantCombinedParams()), ReferenceConstantLayerTest::getTestCaseName);

} // namespace
