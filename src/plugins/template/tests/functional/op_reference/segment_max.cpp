// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/segment_max.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/constant.hpp"

namespace {
struct SegmentMaxParams {
    SegmentMaxParams(const reference_tests::Tensor& dataTensor,
                 const reference_tests::Tensor& segmentIdsTensor,
                 const reference_tests::Tensor& expectedTensor,
                 const int64_t emptySegmentValue)
        : dataTensor(dataTensor),
          segmentIdsTensor(segmentIdsTensor),
          expectedTensor(expectedTensor),
          emptySegmentValue(emptySegmentValue) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor segmentIdsTensor;
    reference_tests::Tensor expectedTensor;
    int64_t emptySegmentValue;
};

class ReferenceSegmentMaxV16LayerTest : public testing::TestWithParam<SegmentMaxParams>,
                                    public reference_tests::CommonReferenceTest {
protected:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.dataTensor.data, params.segmentIdsTensor.data};
        refOutData = {params.expectedTensor.data};
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<SegmentMaxParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_segmentIdsType=" << param.segmentIdsTensor.type;
        result << "_segmentIdsValues=" << testing::PrintToString(param.segmentIdsTensor.data);
        result << "_segmentIdsShape=" << param.segmentIdsTensor.shape;
        result << "_emptySegmentValue=" << param.emptySegmentValue;
        return result.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const SegmentMaxParams& params) {
        using ov::op::v0::Parameter;
        const auto data = std::make_shared<Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto segmentIds =
            std::make_shared<Parameter>(params.segmentIdsTensor.type, params.segmentIdsTensor.shape);
        const auto segmentMax = std::make_shared<ov::op::v16::SegmentMax>(data,
                                                                  segmentIds,
                                                                  params.emptySegmentValue);
        return std::make_shared<ov::Model>(ov::NodeVector{segmentMax}, ov::ParameterVector{data, segmentIds});
    }
};

TEST_P(ReferenceSegmentMaxV16LayerTest, CompareWithRefs) {
    Exec();
}

template <ov::element::Type_t T, ov::element::Type_t T_idx>
std::vector<SegmentMaxParams> generateSegmentMaxParams() {
    using T_D = typename ov::element_type_traits<T>::value_type;
    using T_I = typename ov::element_type_traits<T_idx>::value_type;
    using reference_tests::Tensor;
    std::vector<SegmentMaxParams> segmentMaxParams{
        // 1D
        SegmentMaxParams(Tensor({4}, T, std::vector<T_I>{1, 2, 3, 4}),
                         Tensor({4}, T_idx, std::vector<T_I>{0, 0, 2, 2}),
                         Tensor({3}, T, std::vector<T_D>{2, 10, 4}),
                         10),
        // 2D
        SegmentMaxParams(Tensor({4, 2}, T, std::vector<T_D>{1, 2, 3, 4, 5, 6, 7, 8}),
                         Tensor({4}, T_idx, std::vector<T_I>{0, 0, 1, 1}),
                         Tensor({2, 2}, T, std::vector<T_D>{3, 4, 7, 8}),
                         20),
        // 3D
        SegmentMaxParams(Tensor({4, 2, 2}, T, std::vector<T_D>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}),
                         Tensor({4}, T_idx, std::vector<T_I>{0, 0, 1, 1}),
                         Tensor({2, 2, 2}, T, std::vector<T_D>{5, 6, 7, 8, 13, 14, 15, 16}),
                         30),
        // empty segments
        SegmentMaxParams(Tensor({4}, T, std::vector<T_D>{1, 2, 3, 4}),
                         Tensor({4}, T_idx, std::vector<T_I>{0, 0, 2, 2}),
                         Tensor({3}, T, std::vector<T_D>{2, 40, 4}),
                         40),
        // single element segments
        SegmentMaxParams(Tensor({4}, T, std::vector<T_D>{1, 2, 3, 4}),
                         Tensor({4}, T_idx, std::vector<T_I>{0, 1, 2, 3}),
                         Tensor({4}, T, std::vector<T_D>{1, 2, 3, 4}),
                         50),
        // all elements in one segment
        SegmentMaxParams(Tensor({4}, T, std::vector<T_D>{1, 2, 3, 4}),
                         Tensor({4}, T_idx, std::vector<T_I>{0, 0, 0, 0}),
                         Tensor({1}, T, std::vector<T_D>{4}),
                         60),
    };
    return segmentMaxParams;
}

std::vector<SegmentMaxParams> generateSegmentMaxV16CombinedParams() {
    using ov::element::Type_t;
    const std::vector<std::vector<SegmentMaxParams>> SegmentMaxTypeParams{
        generateSegmentMaxParams<Type_t::i32, Type_t::i32>(),
        //generateSegmentMaxParams<Type_t::i64, Type_t::i32>(),
        //generateSegmentMaxParams<Type_t::u32, Type_t::i32>(),
        //generateSegmentMaxParams<Type_t::u64, Type_t::i32>(),
        //generateSegmentMaxParams<Type_t::f16, Type_t::i32>(),
        //generateSegmentMaxParams<Type_t::f32, Type_t::i32>(),
        //generateSegmentMaxParams<Type_t::i32, Type_t::i64>(),
        //generateSegmentMaxParams<Type_t::i64, Type_t::i64>(),
        //generateSegmentMaxParams<Type_t::u32, Type_t::i64>(),
        //generateSegmentMaxParams<Type_t::u64, Type_t::i64>(),
        //generateSegmentMaxParams<Type_t::f16, Type_t::i64>(),
        //generateSegmentMaxParams<Type_t::f32, Type_t::i64>(),
    };

    std::vector<SegmentMaxParams> combinedParams;
    for (const auto& params : SegmentMaxTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_SegmentMax_With_Hardcoded_Refs,
                         ReferenceSegmentMaxV16LayerTest,
                         testing::ValuesIn(generateSegmentMaxV16CombinedParams()),
                         ReferenceSegmentMaxV16LayerTest::getTestCaseName);

}  // namespace
