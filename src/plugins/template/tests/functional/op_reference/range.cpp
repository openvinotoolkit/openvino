// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/range.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct RangeParams {
    template <class IT>
    RangeParams(const Shape& oShape,
                const element::Type& oType,
                const element::Type& nodeType,
                const std::vector<IT>& oValues,
                float start,
                float stop,
                float step)
        : outShape(oShape),
          outType(oType),
          nodeType(nodeType),
          outData(CreateTensor(oShape, oType, oValues)),
          start(start),
          stop(stop),
          step(step) {}

    Shape outShape;
    element::Type outType;
    element::Type nodeType;
    ov::Tensor outData;
    float start;
    float stop;
    float step;
};

class ReferenceRangeV0LayerTest : public testing::TestWithParam<RangeParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function =
            CreateFunction(params.outShape, params.outType, params.nodeType, params.start, params.stop, params.step);
        refOutData = {params.outData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<RangeParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "oShape=" << param.outShape << "_";
        result << "oType=" << param.outType << "_";
        result << "nType=" << param.nodeType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& oshape,
                                                 const element::Type& otype,
                                                 const element::Type& ntype,
                                                 float fstart,
                                                 float fstop,
                                                 float fstep) {
        auto start = std::make_shared<op::v0::Constant>(ntype, Shape{}, fstart);
        auto stop = std::make_shared<op::v0::Constant>(ntype, Shape{}, fstop);
        auto step = std::make_shared<op::v0::Constant>(ntype, Shape{}, fstep);
        auto range = std::make_shared<op::v0::Range>(start, stop, step);
        return std::make_shared<Model>(NodeVector{range}, ParameterVector{});
    }
};

class ReferenceRangeV4LayerTest : public testing::TestWithParam<RangeParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function =
            CreateFunction(params.outShape, params.outType, params.nodeType, params.start, params.stop, params.step);
        refOutData = {params.outData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<RangeParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "oShape=" << param.outShape << "_";
        result << "oType=" << param.outType << "_";
        result << "nType=" << param.nodeType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& oshape,
                                                 const element::Type& otype,
                                                 const element::Type& ntype,
                                                 float fstart,
                                                 float fstop,
                                                 float fstep) {
        auto start = std::make_shared<op::v0::Constant>(ntype, Shape{}, fstart);
        auto stop = std::make_shared<op::v0::Constant>(ntype, Shape{}, fstop);
        auto step = std::make_shared<op::v0::Constant>(ntype, Shape{}, fstep);
        auto range = std::make_shared<op::v4::Range>(start, stop, step, otype);
        return std::make_shared<Model>(NodeVector{range}, ParameterVector{});
    }
};

TEST_P(ReferenceRangeV0LayerTest, RangeWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceRangeV4LayerTest, RangeWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<RangeParams> generateParamsForRangeV0Int() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RangeParams> params{RangeParams(Shape{4}, IN_ET, IN_ET, std::vector<T>{-5, -2, 1, 4}, -5, 6, 3),
                                    RangeParams(Shape{2}, IN_ET, IN_ET, std::vector<T>{10, 7}, 10, 5, -3)};
    return params;
}

template <element::Type_t IN_ET>
std::vector<RangeParams> generateParamsForRangeV0UnsignedInt() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RangeParams> params{
        RangeParams(Shape{10}, IN_ET, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 0, 10, 1)};
    return params;
}

template <element::Type_t IN_ET>
std::vector<RangeParams> generateParamsForRangeV0Float() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RangeParams> params{
        RangeParams(Shape{4}, IN_ET, IN_ET, std::vector<T>{0.0f, 0.25f, 0.5f, 0.75}, 0.0f, 1.0f, 0.25f),
        RangeParams(Shape{3}, IN_ET, IN_ET, std::vector<T>{1.0f, 4.f, 7.f}, 1.0f, 10.0f, 3.0f),
        RangeParams(Shape{10},
                    IN_ET,
                    IN_ET,
                    std::vector<T>{-1.0f, -0.8f, -0.6f, -0.4f, -0.2f, 0.0f, 0.2f, 0.4f, 0.6f, 0.8f},
                    -1.0f,
                    0.875f,
                    0.2f),
        RangeParams(Shape{8},
                    IN_ET,
                    IN_ET,
                    std::vector<T>{2.0f, 1.75f, 1.5f, 1.25f, 1.0f, 0.75f, 0.5f, 0.25},
                    2.0f,
                    0.0f,
                    -0.25f)};
    return params;
}

template <element::Type_t IN_ET>
std::vector<RangeParams> generateParamsForRangeV4Int() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RangeParams> params{RangeParams(Shape{4}, IN_ET, IN_ET, std::vector<T>{-5, -2, 1, 4}, -5, 6, 3),
                                    RangeParams(Shape{2}, IN_ET, IN_ET, std::vector<T>{10, 7}, 10, 5, -3)};

    return params;
}

template <element::Type_t IN_ET>
std::vector<RangeParams> generateParamsForRangeV4UnsignedInt() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RangeParams> params{
        RangeParams(Shape{10},
                    IN_ET,
                    element::Type_t::f32,
                    std::vector<T>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                    1.2f,
                    11.3f,
                    1.6f),
        RangeParams(Shape{10}, IN_ET, IN_ET, std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 0, 10, 1)};

    return params;
}

template <element::Type_t IN_ET>
std::vector<RangeParams> generateParamsForRangeV4Float() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<RangeParams> params{
        RangeParams(Shape{4}, IN_ET, IN_ET, std::vector<T>{0.0f, 0.25f, 0.5f, 0.75f}, 0.0f, 1.0f, 0.25f),
        RangeParams(Shape{10},
                    IN_ET,
                    IN_ET,
                    std::vector<T>{-1.0f, -0.8f, -0.6f, -0.4f, -0.2f, 0.0f, 0.2f, 0.4f, 0.6f, 0.8f},
                    -1.0f,
                    0.875f,
                    0.2f),
        RangeParams(Shape{8},
                    IN_ET,
                    IN_ET,
                    std::vector<T>{2.0f, 1.75f, 1.5f, 1.25f, 1.0f, 0.75f, 0.5f, 0.25f},
                    2,
                    0,
                    -0.25)};

    return params;
}

std::vector<RangeParams> generateCombinedParamsForRangeV0() {
    const std::vector<std::vector<RangeParams>> allTypeParams{
        generateParamsForRangeV0Float<element::Type_t::f32>(),
        generateParamsForRangeV0Float<element::Type_t::f16>(),
        generateParamsForRangeV0Float<element::Type_t::bf16>(),
        generateParamsForRangeV0Int<element::Type_t::i64>(),
        generateParamsForRangeV0Int<element::Type_t::i32>(),
        generateParamsForRangeV0Int<element::Type_t::i16>(),
        generateParamsForRangeV0Int<element::Type_t::i8>(),
        generateParamsForRangeV0UnsignedInt<element::Type_t::u64>(),
        generateParamsForRangeV0UnsignedInt<element::Type_t::u32>(),
        generateParamsForRangeV0UnsignedInt<element::Type_t::u16>(),
        generateParamsForRangeV0UnsignedInt<element::Type_t::u8>(),
    };

    std::vector<RangeParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

std::vector<RangeParams> generateCombinedParamsForRangeV4() {
    const std::vector<std::vector<RangeParams>> allTypeParams{
        generateParamsForRangeV4Float<element::Type_t::f32>(),
        generateParamsForRangeV4Float<element::Type_t::f16>(),
        generateParamsForRangeV4Float<element::Type_t::bf16>(),
        generateParamsForRangeV4Int<element::Type_t::i64>(),
        generateParamsForRangeV4Int<element::Type_t::i32>(),
        generateParamsForRangeV4Int<element::Type_t::i8>(),
        generateParamsForRangeV4UnsignedInt<element::Type_t::u64>(),
        generateParamsForRangeV4UnsignedInt<element::Type_t::u32>(),
        generateParamsForRangeV4UnsignedInt<element::Type_t::u8>(),
    };

    std::vector<RangeParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Range_V0_With_Hardcoded_Refs,
                         ReferenceRangeV0LayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForRangeV0()),
                         ReferenceRangeV0LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Range_V4_With_Hardcoded_Refs,
                         ReferenceRangeV4LayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForRangeV4()),
                         ReferenceRangeV4LayerTest::getTestCaseName);

}  // namespace
