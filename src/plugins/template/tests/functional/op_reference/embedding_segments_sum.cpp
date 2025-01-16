// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/embedding_segments_sum.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

struct EmbeddingSegmentsSumParams {
    template <class IT>
    EmbeddingSegmentsSumParams(const ov::Shape& iShape,
                               const ov::element::Type& iType,
                               const std::vector<IT>& iValues,
                               const ov::Shape& oShape,
                               const ov::element::Type& oType,
                               const std::vector<IT>& oValues,
                               const std::shared_ptr<ov::op::v0::Constant>& indices,
                               const std::shared_ptr<ov::op::v0::Constant>& segment_ids,
                               const std::shared_ptr<ov::op::v0::Constant>& num_segments,
                               const std::shared_ptr<ov::op::v0::Constant>& default_index = nullptr,
                               const std::shared_ptr<ov::op::v0::Constant>& per_sample_weights = nullptr)
        : _iShape(iShape),
          _iType(iType),
          _iData(CreateTensor(iShape, iType, iValues)),
          _refShape(oShape),
          _refType(oType),
          _refData(CreateTensor(oShape, oType, oValues)) {
        _segmentIds = segment_ids;
        _indices = indices;
        _numSegments = num_segments;
        _defaultIndex = default_index;
        _perSampleWeights = per_sample_weights;
    }

    ov::Shape _iShape;
    ov::element::Type _iType;
    ov::Tensor _iData;

    ov::Shape _refShape;
    ov::element::Type _refType;
    ov::Tensor _refData;

    std::shared_ptr<ov::op::v0::Constant> _indices;
    std::shared_ptr<ov::op::v0::Constant> _segmentIds;
    std::shared_ptr<ov::op::v0::Constant> _numSegments;
    std::shared_ptr<ov::op::v0::Constant> _defaultIndex;      // Optional, default filled zero.
    std::shared_ptr<ov::op::v0::Constant> _perSampleWeights;  // Optional, default is tensor of ones.
};

class ReferenceEmbeddingSegmentsSumLayerTest : public testing::TestWithParam<EmbeddingSegmentsSumParams>,
                                               public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params._iShape,
                                  params._iType,
                                  params._indices,
                                  params._segmentIds,
                                  params._numSegments,
                                  params._defaultIndex,
                                  params._perSampleWeights);
        inputData = {params._iData};
        refOutData = {params._refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<EmbeddingSegmentsSumParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "_iShape=" << param._iShape << "_";
        result << "_iType=" << param._iType << "_";
        result << "_refShape=" << param._refShape << "_";
        result << "_refType=" << param._refType;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type& input_type,
                                                 const std::shared_ptr<ov::op::v0::Constant> indices,
                                                 const std::shared_ptr<ov::op::v0::Constant> segment_ids,
                                                 const std::shared_ptr<ov::op::v0::Constant> num_segments,
                                                 const std::shared_ptr<ov::op::v0::Constant> default_index,
                                                 const std::shared_ptr<ov::op::v0::Constant> per_sample_weights) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);

        if (default_index) {
            if (per_sample_weights) {
                const auto ess = std::make_shared<op::v3::EmbeddingSegmentsSum>(in,
                                                                                indices,
                                                                                segment_ids,
                                                                                num_segments,
                                                                                default_index,
                                                                                per_sample_weights);
                return std::make_shared<Model>(NodeVector{ess}, ParameterVector{in});
            } else {
                const auto ess = std::make_shared<op::v3::EmbeddingSegmentsSum>(in,
                                                                                indices,
                                                                                segment_ids,
                                                                                num_segments,
                                                                                default_index);
                return std::make_shared<Model>(NodeVector{ess}, ParameterVector{in});
            }
        } else {
            const auto ess = std::make_shared<op::v3::EmbeddingSegmentsSum>(in, indices, segment_ids, num_segments);
            return std::make_shared<Model>(NodeVector{ess}, ParameterVector{in});
        }
    }
};

TEST_P(ReferenceEmbeddingSegmentsSumLayerTest, CompareWithRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_EmbeddingSegmentsSum_With_Hardcoded_Refs,
    ReferenceEmbeddingSegmentsSumLayerTest,
    ::testing::Values(
        EmbeddingSegmentsSumParams(
            ov::Shape{5, 2},
            ov::element::f32,
            std::vector<float>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
            ov::Shape{3, 2},
            ov::element::f32,
            {-1.05f, -1.2f, -0.2f, -0.6f, -0.1f, 0.4f},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 0, 2, 2}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{3}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{0}),
            std::make_shared<ov::op::v0::Constant>(element::f32,
                                                   ov::Shape({4}),
                                                   std::vector<float>{0.5, 0.5, 0.5, 0.5})),
        EmbeddingSegmentsSumParams(
            ov::Shape{5, 2},
            ov::element::f64,
            std::vector<double>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
            ov::Shape{3, 2},
            ov::element::f64,
            std::vector<double>{-2.1, -2.4, -0.2, -0.6, -0.2, 0.8},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 0, 2, 2}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{3}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{0})),
        EmbeddingSegmentsSumParams(
            ov::Shape{5, 2},
            ov::element::i32,
            std::vector<int32_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::i32,
            std::vector<int32_t>{-6, -4, 0, 0, 2, 18},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 0, 2, 2}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{3})),
        EmbeddingSegmentsSumParams(
            ov::Shape{5, 2},
            ov::element::u32,
            std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::u32,
            std::vector<uint32_t>{6, 8, 3, 4, 16, 18},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 0, 2, 2}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{3}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{1})),
        EmbeddingSegmentsSumParams(
            ov::Shape{5, 2},
            ov::element::f16,
            std::vector<float16>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
            ov::Shape{3, 2},
            ov::element::f16,
            std::vector<float16>{-2.1, -2.4, 0, 0, -0.2, 0.8},
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({4}), std::vector<int64_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({4}), std::vector<int64_t>{0, 0, 2, 2}),
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape(), std::vector<int64_t>{3})),
        EmbeddingSegmentsSumParams(
            ov::Shape{5, 2},
            ov::element::i64,
            std::vector<int64_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::i64,
            std::vector<int64_t>{-6, -4, -1, 2, 2, 18},
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({4}), std::vector<int64_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({4}), std::vector<int64_t>{0, 0, 2, 2}),
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape(), std::vector<int64_t>{3}),
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape(), std::vector<int64_t>{0})),
        EmbeddingSegmentsSumParams(
            ov::Shape{5, 2},
            ov::element::i8,
            std::vector<int8_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::i8,
            std::vector<int8_t>{-12, -8, -1, 2, 4, 36},
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({4}), std::vector<int64_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({4}), std::vector<int64_t>{0, 0, 2, 2}),
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape(), std::vector<int64_t>{3}),
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape(), std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(element::i8, ov::Shape({4}), std::vector<int8_t>{2, 2, 2, 2})),
        EmbeddingSegmentsSumParams(
            ov::Shape{5, 2},
            ov::element::u8,
            std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::u8,
            std::vector<uint8_t>{6, 8, 1, 2, 16, 18},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 0, 2, 2}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{3}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{0}))),
    ReferenceEmbeddingSegmentsSumLayerTest::getTestCaseName);
