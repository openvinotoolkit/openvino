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
                     const reference_tests::Tensor& numSegmentsTensor,
                     const reference_tests::Tensor& expectedTensor,
                     const reference_tests::Tensor& expectedTensorNumSegments,
                     const ov::op::FillMode fillMode)
        : dataTensor(dataTensor),
          segmentIdsTensor(segmentIdsTensor),
          numSegmentsTensor(numSegmentsTensor),
          expectedTensor(expectedTensor),
          expectedTensorNumSegments(expectedTensorNumSegments),
          fillMode(fillMode) {}

    reference_tests::Tensor dataTensor;
    reference_tests::Tensor segmentIdsTensor;
    reference_tests::Tensor numSegmentsTensor;
    reference_tests::Tensor expectedTensor;
    reference_tests::Tensor expectedTensorNumSegments;
    ov::op::FillMode fillMode;
};

class ReferenceSegmentMaxV16LayerTest : public testing::TestWithParam<std::tuple<SegmentMaxParams, bool>>,
                                        public reference_tests::CommonReferenceTest {
protected:
    void SetUp() override {
        const auto& params = std::get<0>(GetParam());
        const bool useNumSegments = std::get<1>(GetParam());
        function = CreateFunction(params, useNumSegments);
        inputData = {params.dataTensor.data, params.segmentIdsTensor.data};
        if (useNumSegments) {
            inputData.push_back(params.numSegmentsTensor.data);
            refOutData.push_back(params.expectedTensorNumSegments.data);
        } else {
            refOutData = {params.expectedTensor.data};
        }
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<std::tuple<SegmentMaxParams, bool>>& obj) {
        auto param = std::get<0>(obj.param);
        bool useNumSegments = std::get<1>(obj.param);
        std::ostringstream result;
        result << "dType=" << param.dataTensor.type;
        result << "_dShape=" << param.dataTensor.shape;
        result << "_segmentIdsType=" << param.segmentIdsTensor.type;
        result << "_segmentIdsValues=" << testing::PrintToString(param.segmentIdsTensor.data);
        result << "_segmentIdsShape=" << param.segmentIdsTensor.shape;
        result << "_fillMode=" << param.fillMode;
        result << "_useNumSegments=" << useNumSegments;
        return result.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const SegmentMaxParams& params, bool useNumSegments) {
        using ov::op::v0::Parameter;
        const auto data = std::make_shared<Parameter>(params.dataTensor.type, params.dataTensor.shape);
        const auto segmentIds =
            std::make_shared<Parameter>(params.segmentIdsTensor.type, params.segmentIdsTensor.shape);
        std::shared_ptr<ov::op::v16::SegmentMax> segmentMax;
        ov::ParameterVector parameters = {data, segmentIds};
        if (useNumSegments) {
            const auto numSegments =
                std::make_shared<Parameter>(params.numSegmentsTensor.type, params.numSegmentsTensor.shape);
            parameters.emplace_back(numSegments);
            segmentMax = std::make_shared<ov::op::v16::SegmentMax>(data, segmentIds, numSegments, params.fillMode);
        } else {
            segmentMax = std::make_shared<ov::op::v16::SegmentMax>(data, segmentIds, params.fillMode);
        }
        return std::make_shared<ov::Model>(ov::NodeVector{segmentMax}, parameters);
    }
};

TEST_P(ReferenceSegmentMaxV16LayerTest, CompareWithRefs) {
    Exec();
}

template <ov::element::Type_t T, ov::element::Type_t T_idx>
std::vector<SegmentMaxParams> generateSegmentMaxParams(ov::op::FillMode fillMode) {
    using T_D = typename ov::element_type_traits<T>::value_type;
    using T_I = typename ov::element_type_traits<T_idx>::value_type;
    using reference_tests::Tensor;
    const auto empty_segment_value = fillMode == ov::op::FillMode::ZERO ? T_D(0) : std::numeric_limits<T_D>::lowest();
    std::vector<SegmentMaxParams> segmentMaxParams{
        // 1D
        SegmentMaxParams(Tensor({4}, T, std::vector<T_D>{1, 2, 3, 4}),                 // data
                         Tensor({4}, T_idx, std::vector<T_I>{0, 0, 2, 2}),             // segmentIds
                         Tensor({}, T_idx, std::vector<T_I>{3}),                       // numSegments
                         Tensor({3}, T, std::vector<T_D>{2, empty_segment_value, 4}),  // expectedResult
                         Tensor({3}, T, std::vector<T_D>{2, empty_segment_value, 4}),  // expectedResultNumSegments
                         fillMode),
        // 2D
        SegmentMaxParams(Tensor({4, 2}, T, std::vector<T_D>{8, 7, 6, 5, 4, 3, 2, 1}),  // data
                         Tensor({4}, T_idx, std::vector<T_I>{0, 0, 1, 1}),             // segmentIds
                         Tensor({}, T_idx, std::vector<T_I>{2}),                       // numSegments
                         Tensor({2, 2}, T, std::vector<T_D>{8, 7, 4, 3}),              // expectedResult
                         Tensor({2, 2}, T, std::vector<T_D>{8, 7, 4, 3}),              // expectedResultNumSegments
                         fillMode),
        // 3D
        SegmentMaxParams(
            Tensor({4, 2, 2}, T, std::vector<T_D>{1, 2, 3, 4, 5, 6, 7, 8, 16, 15, 14, 13, 12, 11, 10, 9}),  // data
            Tensor({4}, T_idx, std::vector<T_I>{0, 0, 1, 1}),                    // segmentIds
            Tensor({}, T_idx, std::vector<T_I>{2}),                              // numSegments
            Tensor({2, 2, 2}, T, std::vector<T_D>{5, 6, 7, 8, 16, 15, 14, 13}),  // expectedResult
            Tensor({2, 2, 2}, T, std::vector<T_D>{5, 6, 7, 8, 16, 15, 14, 13}),  // expectedResultNumSegments
            fillMode),
        // empty segments
        SegmentMaxParams(Tensor({4}, T, std::vector<T_D>{1, 2, 3, 4}),                 // data
                         Tensor({4}, T_idx, std::vector<T_I>{0, 0, 2, 2}),             // segmentIds
                         Tensor({}, T_idx, std::vector<T_I>{3}),                       // numSegments
                         Tensor({3}, T, std::vector<T_D>{2, empty_segment_value, 4}),  // expectedResult
                         Tensor({3}, T, std::vector<T_D>{2, empty_segment_value, 4}),  // expectedResultNumSegments
                         fillMode),
        // single element segments
        SegmentMaxParams(Tensor({4}, T, std::vector<T_D>{1, 2, 3, 4}),      // data
                         Tensor({4}, T_idx, std::vector<T_I>{0, 1, 2, 3}),  // segmentIds
                         Tensor({}, T_idx, std::vector<T_I>{4}),            // numSegments
                         Tensor({4}, T, std::vector<T_D>{1, 2, 3, 4}),      // expectedResult
                         Tensor({4}, T, std::vector<T_D>{1, 2, 3, 4}),      // expectedResultNumSegments
                         fillMode),
        // all elements in one segment
        SegmentMaxParams(Tensor({4}, T, std::vector<T_D>{1, 2, 3, 4}),      // data
                         Tensor({4}, T_idx, std::vector<T_I>{0, 0, 0, 0}),  // segmentIds
                         Tensor({}, T_idx, std::vector<T_I>{1}),            // numSegments
                         Tensor({1}, T, std::vector<T_D>{4}),               // expectedResult
                         Tensor({1}, T, std::vector<T_D>{4}),               // expectedResultNumSegments
                         fillMode),
        // numSegments < max(segmentIds) + 1
        SegmentMaxParams(Tensor({4}, T, std::vector<T_D>{1, 2, 3, 4}),      // data
                         Tensor({4}, T_idx, std::vector<T_I>{0, 1, 2, 3}),  // segmentIds
                         Tensor({}, T_idx, std::vector<T_I>{2}),            // numSegments
                         Tensor({4}, T, std::vector<T_D>{1, 2, 3, 4}),      // expectedResult
                         Tensor({2}, T, std::vector<T_D>{1, 2}),            // expectedResultNumSegments
                         fillMode),
        // numSegments > max(segmentIds) + 1
        SegmentMaxParams(Tensor({4}, T, std::vector<T_D>{1, 2, 3, 4}),      // data
                         Tensor({4}, T_idx, std::vector<T_I>{0, 1, 2, 3}),  // segmentIds
                         Tensor({}, T_idx, std::vector<T_I>{6}),            // numSegments
                         Tensor({4}, T, std::vector<T_D>{1, 2, 3, 4}),      // expectedResult
                         Tensor(                                            // expectedResultNumSegments
                             {6},
                             T,
                             std::vector<T_D>{1, 2, 3, 4, empty_segment_value, empty_segment_value}),
                         fillMode),
    };
    return segmentMaxParams;
}

std::vector<std::tuple<SegmentMaxParams, bool>> generateSegmentMaxV16CombinedParams() {
    using ov::element::Type_t;
    const std::vector<std::vector<SegmentMaxParams>> SegmentMaxTypeParams{
        generateSegmentMaxParams<Type_t::i32, Type_t::i32>(ov::op::FillMode::ZERO),
        generateSegmentMaxParams<Type_t::i32, Type_t::i32>(ov::op::FillMode::LOWEST),
        generateSegmentMaxParams<Type_t::i64, Type_t::i32>(ov::op::FillMode::ZERO),
        generateSegmentMaxParams<Type_t::i64, Type_t::i32>(ov::op::FillMode::LOWEST),
        generateSegmentMaxParams<Type_t::u32, Type_t::i32>(ov::op::FillMode::ZERO),
        generateSegmentMaxParams<Type_t::u32, Type_t::i32>(ov::op::FillMode::LOWEST),
        generateSegmentMaxParams<Type_t::u64, Type_t::i32>(ov::op::FillMode::ZERO),
        generateSegmentMaxParams<Type_t::u64, Type_t::i32>(ov::op::FillMode::LOWEST),
        generateSegmentMaxParams<Type_t::f16, Type_t::i32>(ov::op::FillMode::ZERO),
        generateSegmentMaxParams<Type_t::f16, Type_t::i32>(ov::op::FillMode::LOWEST),
        generateSegmentMaxParams<Type_t::f32, Type_t::i32>(ov::op::FillMode::ZERO),
        generateSegmentMaxParams<Type_t::f32, Type_t::i32>(ov::op::FillMode::LOWEST),
        generateSegmentMaxParams<Type_t::i32, Type_t::i64>(ov::op::FillMode::ZERO),
        generateSegmentMaxParams<Type_t::i32, Type_t::i64>(ov::op::FillMode::LOWEST),
        generateSegmentMaxParams<Type_t::i64, Type_t::i64>(ov::op::FillMode::ZERO),
        generateSegmentMaxParams<Type_t::i64, Type_t::i64>(ov::op::FillMode::LOWEST),
        generateSegmentMaxParams<Type_t::u32, Type_t::i64>(ov::op::FillMode::ZERO),
        generateSegmentMaxParams<Type_t::u32, Type_t::i64>(ov::op::FillMode::LOWEST),
        generateSegmentMaxParams<Type_t::u64, Type_t::i64>(ov::op::FillMode::ZERO),
        generateSegmentMaxParams<Type_t::u64, Type_t::i64>(ov::op::FillMode::LOWEST),
        generateSegmentMaxParams<Type_t::f16, Type_t::i64>(ov::op::FillMode::ZERO),
        generateSegmentMaxParams<Type_t::f16, Type_t::i64>(ov::op::FillMode::LOWEST),
        generateSegmentMaxParams<Type_t::f32, Type_t::i64>(ov::op::FillMode::ZERO),
        generateSegmentMaxParams<Type_t::f32, Type_t::i64>(ov::op::FillMode::LOWEST),
    };

    // expand cases with useNumSegments bool
    std::vector<std::tuple<SegmentMaxParams, bool>> combinedParams;
    for (const auto& params : SegmentMaxTypeParams) {
        for (const auto& param : params) {
            combinedParams.emplace_back(param, true);
            combinedParams.emplace_back(param, false);
        }
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_SegmentMax_With_Hardcoded_Refs,
                         ReferenceSegmentMaxV16LayerTest,
                         testing::ValuesIn(generateSegmentMaxV16CombinedParams()),
                         ReferenceSegmentMaxV16LayerTest::getTestCaseName);
}  // namespace
