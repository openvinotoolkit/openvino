// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/minimum.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace ov;
using namespace reference_tests;

struct MinimumParams {
    template <class IT, class OT>
    MinimumParams(const PartialShape& s,
                  const element::Type& iType,
                  const element::Type& oType,
                  const std::vector<IT>& iValues1,
                  const std::vector<IT>& iValues2,
                  const std::vector<OT>& oValues)
        : pshape(s),
          inType(iType),
          outType(oType),
          inputData1(CreateTensor(iType, iValues1)),
          inputData2(CreateTensor(iType, iValues2)),
          refData(CreateTensor(oType, oValues)) {}
    PartialShape pshape;
    element::Type inType;
    element::Type outType;
    ov::Tensor inputData1;
    ov::Tensor inputData2;
    ov::Tensor refData;
};

class ReferenceMinimumLayerTest : public testing::TestWithParam<MinimumParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType);
        inputData = {params.inputData1, params.inputData2};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<MinimumParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PartialShape& shape, const element::Type& data_type) {
        auto A = std::make_shared<op::v0::Parameter>(data_type, shape);
        auto B = std::make_shared<op::v0::Parameter>(data_type, shape);
        return std::make_shared<ov::Model>(std::make_shared<op::v1::Minimum>(A, B), ParameterVector{A, B});
    }
};

TEST_P(ReferenceMinimumLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Minimum,
    ReferenceMinimumLayerTest,
    ::testing::Values(MinimumParams(PartialShape{8},
                                    element::u8,
                                    element::u8,
                                    std::vector<uint8_t>{1, 8, 8, 17, 5, 5, 2, 3},
                                    std::vector<uint8_t>{1, 2, 4, 8, 0, 2, 1, 200},
                                    std::vector<uint8_t>{1, 2, 4, 8, 0, 2, 1, 3}),
                      MinimumParams(PartialShape{8},
                                    element::u16,
                                    element::u16,
                                    std::vector<uint16_t>{1, 8, 8, 17, 5, 7, 123, 3},
                                    std::vector<uint16_t>{1, 2, 4, 8, 0, 2, 1, 1037},
                                    std::vector<uint16_t>{1, 2, 4, 8, 0, 2, 1, 3}),
                      MinimumParams(PartialShape{8},
                                    element::u32,
                                    element::u32,
                                    std::vector<uint32_t>{1, 8, 8, 17, 5, 5, 2, 1},
                                    std::vector<uint32_t>{1, 2, 4, 8, 0, 2, 1, 222},
                                    std::vector<uint32_t>{1, 2, 4, 8, 0, 2, 1, 1}),
                      MinimumParams(PartialShape{8},
                                    element::u64,
                                    element::u64,
                                    std::vector<uint64_t>{1, 8, 8, 17, 5, 5, 2, 13},
                                    std::vector<uint64_t>{1, 2, 4, 8, 0, 2, 1, 2222},
                                    std::vector<uint64_t>{1, 2, 4, 8, 0, 2, 1, 13}),
                      MinimumParams(PartialShape{8},
                                    element::f32,
                                    element::f32,
                                    std::vector<float>{1, 8, -8, 17, -0.5, 0.5, 2, 1},
                                    std::vector<float>{1, 2, 4, 8, 0, 0, 1, 1.5},
                                    std::vector<float>{1, 2, -8, 8, -.5, 0, 1, 1}),
                      MinimumParams(PartialShape{8},
                                    element::i32,
                                    element::i32,
                                    std::vector<int32_t>{1, 8, -8, 17, -5, 67635216, 2, 1},
                                    std::vector<int32_t>{1, 2, 4, 8, 0, 18448, 1, 6},
                                    std::vector<int32_t>{1, 2, -8, 8, -5, 18448, 1, 1}),
                      MinimumParams(PartialShape{8},
                                    element::i64,
                                    element::i64,
                                    std::vector<int64_t>{1, 8, -8, 17, -5, 67635216, 2, 17179887632},
                                    std::vector<int64_t>{1, 2, 4, 8, 0, 18448, 1, 280592},
                                    std::vector<int64_t>{1, 2, -8, 8, -5, 18448, 1, 280592})),
    ReferenceMinimumLayerTest::getTestCaseName);
