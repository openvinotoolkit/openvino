// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <limits>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <tuple>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ngraph;
using namespace InferenceEngine;

struct EmbeddingSegmentsSumParams {
    template <class IT>
    EmbeddingSegmentsSumParams(const ngraph::PartialShape& iShape,
                               const ngraph::element::Type& iType,
                               const std::vector<IT>& iValues,
                               const ngraph::PartialShape& oShape,
                               const ngraph::element::Type& oType,
                               const std::vector<IT>& oValues,
                               const std::vector<int32_t>& segment_ids,
                               const std::vector<int32_t>& indices,
                               const int32_t& num_segments,
                               const int32_t& default_index = -1,
                               const std::vector<float>& per_sample_weights = std::vector<float>())
        : _iShape(iShape),
          _iType(iType),
          _iData(CreateBlob(iType, iValues)),
          _refShape(oShape),
          _refType(oType),
          _refData(CreateBlob(oType, oValues)) {
        _segmentIds = segment_ids;
        _indices = indices;
        _numSegments = num_segments;
        _defaultIndex = default_index;
        _perSampleWeights = per_sample_weights;
    }

    ngraph::PartialShape _iShape;
    ngraph::element::Type _iType;
    InferenceEngine::Blob::Ptr _iData;

    ngraph::PartialShape _refShape;
    ngraph::element::Type _refType;
    InferenceEngine::Blob::Ptr _refData;

    std::vector<int32_t> _segmentIds;
    std::vector<int32_t> _indices;
    int32_t _numSegments = 0;
    int32_t _defaultIndex = -1;            // Optional, default filled zero.
    std::vector<float> _perSampleWeights;  // Optional, default is tensor of ones.
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
    static std::shared_ptr<Function> CreateFunction(const PartialShape& input_shape,
                                                    const element::Type& input_type,
                                                    const std::vector<int32_t>& segment_ids,
                                                    const std::vector<int32_t>& indices,
                                                    const int32_t& num_segments,
                                                    const int32_t& default_index,
                                                    const std::vector<float>& per_sample_weights) {
        const auto in = std::make_shared<op::Parameter>(input_type, input_shape);

        std::vector<size_t> i_shape = {indices.size()};
        auto indicesNode = std::make_shared<ngraph::opset1::Constant>(element::i32, i_shape, indices);
        std::vector<size_t> o_shape = {segment_ids.size()};
        auto segmentIdNode = std::make_shared<ngraph::opset1::Constant>(element::i32, o_shape, segment_ids);
        std::vector<size_t> shape_0 = {};
        auto segmentNumNode = std::make_shared<ngraph::opset1::Constant>(element::i32, shape_0, num_segments);

        if (default_index >= 0) {
            auto defIdxNode = std::make_shared<ngraph::opset1::Constant>(element::i32, shape_0, default_index);

            if (per_sample_weights.size() > 0) {
                auto weightsNode =
                    std::make_shared<ngraph::opset1::Constant>(element::f32, i_shape, per_sample_weights);

                const auto ess = std::make_shared<op::v3::EmbeddingSegmentsSum>(in,
                                                                                indicesNode,
                                                                                segmentIdNode,
                                                                                segmentNumNode,
                                                                                defIdxNode,
                                                                                weightsNode);
                return std::make_shared<Function>(NodeVector{ess}, ParameterVector{in});
            } else {
                const auto ess = std::make_shared<op::v3::EmbeddingSegmentsSum>(in,
                                                                                indicesNode,
                                                                                segmentIdNode,
                                                                                segmentNumNode,
                                                                                defIdxNode);
                return std::make_shared<Function>(NodeVector{ess}, ParameterVector{in});
            }
        } else {
            const auto ess =
                std::make_shared<op::v3::EmbeddingSegmentsSum>(in, indicesNode, segmentIdNode, segmentNumNode);
            return std::make_shared<Function>(NodeVector{ess}, ParameterVector{in});
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
        EmbeddingSegmentsSumParams(ngraph::PartialShape{5, 2},
                                   ngraph::element::f32,
                                   std::vector<float>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
                                   ngraph::PartialShape{3, 2},
                                   ngraph::element::f32,
                                   {-1.05f, -1.2f, -0.2f, -0.6f, -0.1f, 0.4f},
                                   {0, 2, 3, 4},
                                   {0, 0, 2, 2},
                                   3,
                                   0,
                                   {0.5, 0.5, 0.5, 0.5}),
        EmbeddingSegmentsSumParams(ngraph::PartialShape{5, 2},
                                   ngraph::element::f64,
                                   std::vector<double>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
                                   ngraph::PartialShape{3, 2},
                                   ngraph::element::f64,
                                   std::vector<double>{-2.1, -2.4, -0.2, -0.6, -0.2, 0.8},
                                   {0, 2, 3, 4},
                                   {0, 0, 2, 2},
                                   3,
                                   0),
        EmbeddingSegmentsSumParams(ngraph::PartialShape{5, 2},
                                   ngraph::element::i32,
                                   std::vector<int32_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
                                   ngraph::PartialShape{3, 2},
                                   ngraph::element::i32,
                                   std::vector<int32_t>{-6, -4, 0, 0, 2, 18},
                                   {0, 2, 3, 4},
                                   {0, 0, 2, 2},
                                   3),
        EmbeddingSegmentsSumParams(ngraph::PartialShape{5, 2},
                                   ngraph::element::u32,
                                   std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                   ngraph::PartialShape{3, 2},
                                   ngraph::element::u32,
                                   std::vector<uint32_t>{6, 8, 3, 4, 16, 18},
                                   {0, 2, 3, 4},
                                   {0, 0, 2, 2},
                                   3,
                                   1),
        EmbeddingSegmentsSumParams(ngraph::PartialShape{5, 2},
                                   ngraph::element::f16,
                                   std::vector<float16>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
                                   ngraph::PartialShape{3, 2},
                                   ngraph::element::f16,
                                   std::vector<float16>{-2.1, -2.4, 0, 0, -0.2, 0.8},
                                   {0, 2, 3, 4},
                                   {0, 0, 2, 2},
                                   3),
        EmbeddingSegmentsSumParams(ngraph::PartialShape{5, 2},
                                   ngraph::element::i64,
                                   std::vector<int64_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
                                   ngraph::PartialShape{3, 2},
                                   ngraph::element::i64,
                                   std::vector<int64_t>{-6, -4, -1, 2, 2, 18},
                                   {0, 2, 3, 4},
                                   {0, 0, 2, 2},
                                   3,
                                   0),
        EmbeddingSegmentsSumParams(ngraph::PartialShape{5, 2},
                                   ngraph::element::i8,
                                   std::vector<int8_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
                                   ngraph::PartialShape{3, 2},
                                   ngraph::element::i8,
                                   std::vector<int8_t>{-6, -4, -1, 2, 2, 18},
                                   {0, 2, 3, 4},
                                   {0, 0, 2, 2},
                                   3,
                                   0),
        EmbeddingSegmentsSumParams(ngraph::PartialShape{5, 2},
                                   ngraph::element::u8,
                                   std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                   ngraph::PartialShape{3, 2},
                                   ngraph::element::u8,
                                   std::vector<uint8_t>{6, 8, 1, 2, 16, 18},
                                   {0, 2, 3, 4},
                                   {0, 0, 2, 2},
                                   3,
                                   0)),
    ReferenceEmbeddingSegmentsSumLayerTest::getTestCaseName);
