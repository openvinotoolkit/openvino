// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/floor.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct FloorParams {
    template <class IT>
    FloorParams(const PartialShape& shape,
                const element::Type& iType,
                const std::vector<IT>& iValues,
                const std::vector<IT>& oValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          refData(CreateTensor(iType, oValues)) {}

    PartialShape pshape;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceFloorLayerTest : public testing::TestWithParam<FloorParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.outType);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<FloorParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& input_shape,
                                                 const element::Type& input_type,
                                                 const element::Type& expected_output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        const auto floor = std::make_shared<op::v0::Floor>(in);
        return std::make_shared<Model>(NodeVector{floor}, ParameterVector{in});
    }
};

TEST_P(ReferenceFloorLayerTest, FloorWithHardcodedRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<FloorParams> generateParamsForFloorFloat() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<FloorParams> params{FloorParams(ov::PartialShape{4},
                                                IN_ET,
                                                std::vector<T>{-2.5f, -2.0f, 0.3f, 4.8f},
                                                std::vector<T>{-3.0f, -2.0f, 0.0f, 4.0f})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<FloorParams> generateParamsForFloorInt64() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<FloorParams> params{FloorParams(ov::PartialShape{3},
                                                IN_ET,
                                                std::vector<T>{0, 1, 0x4000000000000001},
                                                std::vector<T>{0, 1, 0x4000000000000001})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<FloorParams> generateParamsForFloorInt32() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<FloorParams> params{FloorParams(ov::PartialShape{4},
                                                IN_ET,
                                                std::vector<T>{2, 136314888, 0x40000010, 0x40000001},
                                                std::vector<T>{2, 136314888, 0x40000010, 0x40000001})};
    return params;
}

template <element::Type_t IN_ET>
std::vector<FloorParams> generateParamsForFloorInt() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<FloorParams> params{
        FloorParams(ov::PartialShape{4}, IN_ET, std::vector<T>{2, 64, 0x40, 0x01}, std::vector<T>{2, 64, 0x40, 0x01})};
    return params;
}

std::vector<FloorParams> generateCombinedParamsForFloor() {
    const std::vector<std::vector<FloorParams>> allTypeParams{generateParamsForFloorFloat<element::Type_t::f32>(),
                                                              generateParamsForFloorFloat<element::Type_t::f16>(),
                                                              generateParamsForFloorInt64<element::Type_t::i64>(),
                                                              generateParamsForFloorInt32<element::Type_t::i32>(),
                                                              generateParamsForFloorInt<element::Type_t::i16>(),
                                                              generateParamsForFloorInt<element::Type_t::i8>(),
                                                              generateParamsForFloorInt<element::Type_t::u64>(),
                                                              generateParamsForFloorInt<element::Type_t::u32>(),
                                                              generateParamsForFloorInt<element::Type_t::u16>(),
                                                              generateParamsForFloorInt<element::Type_t::u8>()};

    std::vector<FloorParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Floor_With_Hardcoded_Refs,
                         ReferenceFloorLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForFloor()),
                         ReferenceFloorLayerTest::getTestCaseName);

}  // namespace
