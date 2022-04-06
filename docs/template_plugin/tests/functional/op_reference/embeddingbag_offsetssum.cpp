// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <shared_test_classes/base/layer_test_utils.hpp>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;
using namespace InferenceEngine;

struct EmbeddingBagOffsetsSumParams {
    template <class IT>
    EmbeddingBagOffsetsSumParams(const ov::PartialShape& iShape,
                                 const ov::element::Type& iType,
                                 const std::vector<IT>& iValues,
                                 const ov::PartialShape& oShape,
                                 const ov::element::Type& oType,
                                 const std::vector<IT>& oValues,
                                 const std::shared_ptr<ngraph::opset1::Constant>& indices,
                                 const std::shared_ptr<ngraph::opset1::Constant>& offsets,
                                 const std::shared_ptr<ngraph::opset1::Constant>& default_index = nullptr,
                                 const std::shared_ptr<ngraph::opset1::Constant>& per_sample_weights = nullptr)
        : _iShape(iShape),
          _iType(iType),
          _iData(CreateTensor(iType, iValues)),
          _refShape(oShape),
          _refType(oType),
          _refData(CreateTensor(oType, oValues)) {
        _indices = indices;
        _offsets = offsets;
        _defaultIndex = default_index;
        _perSampleWeights = per_sample_weights;
    }
    ov::PartialShape _iShape;
    ov::element::Type _iType;
    ov::Tensor _iData;

    ov::PartialShape _refShape;
    ov::element::Type _refType;
    ov::Tensor _refData;

    std::shared_ptr<ngraph::opset1::Constant> _indices;
    std::shared_ptr<ngraph::opset1::Constant> _offsets;
    std::shared_ptr<ngraph::opset1::Constant> _defaultIndex;      // Optional, default filled zero.
    std::shared_ptr<ngraph::opset1::Constant> _perSampleWeights;  // Optional, default is tensor of ones.
};

class ReferenceEmbeddingBagOffsetsSumLayerTest : public testing::TestWithParam<EmbeddingBagOffsetsSumParams>,
                                                 public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params._iShape,
                                  params._iType,
                                  params._indices,
                                  params._offsets,
                                  params._defaultIndex,
                                  params._perSampleWeights);
        inputData = {params._iData};
        refOutData = {params._refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<EmbeddingBagOffsetsSumParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "_iShape=" << param._iShape << "_";
        result << "_iType=" << param._iType << "_";
        result << "_refShape=" << param._refShape << "_";
        result << "_refType=" << param._refType;
        return result.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(
        const PartialShape& input_shape,
        const element::Type& input_type,
        const std::shared_ptr<ngraph::opset1::Constant> indices,
        const std::shared_ptr<ngraph::opset1::Constant> offsets,
        const std::shared_ptr<ngraph::opset1::Constant> default_index,
        const std::shared_ptr<ngraph::opset1::Constant> per_sample_weights) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);

        if (default_index) {
            if (per_sample_weights) {
                const auto ess = std::make_shared<op::v3::EmbeddingBagOffsetsSum>(in,
                                                                                  indices,
                                                                                  offsets,
                                                                                  default_index,
                                                                                  per_sample_weights);
                return std::make_shared<Model>(NodeVector{ess}, ParameterVector{in});
            } else {
                const auto ess = std::make_shared<op::v3::EmbeddingBagOffsetsSum>(in, indices, offsets, default_index);
                return std::make_shared<Model>(NodeVector{ess}, ParameterVector{in});
            }
        } else {
            const auto ess = std::make_shared<op::v3::EmbeddingBagOffsetsSum>(in, indices, offsets);
            return std::make_shared<Model>(NodeVector{ess}, ParameterVector{in});
        }
    }
};

TEST_P(ReferenceEmbeddingBagOffsetsSumLayerTest, CompareWithRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_EmbeddingBagOffsetsSum_With_Hardcoded_Refs,
    ReferenceEmbeddingBagOffsetsSumLayerTest,
    ::testing::Values(
        EmbeddingBagOffsetsSumParams(
            ov::PartialShape{5, 2},
            ov::element::f32,
            std::vector<float>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
            ov::PartialShape{3, 2},
            ov::element::f32,
            {-1.05f, -1.2f, -0.2f, -0.6f, -0.1f, 0.4f},
            std::make_shared<ngraph::opset1::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ngraph::opset1::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2}),
            std::make_shared<ngraph::opset1::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{0}),
            std::make_shared<ngraph::opset1::Constant>(element::f32,
                                                       ov::Shape({4}),
                                                       std::vector<float>{0.5, 0.5, 0.5, 0.5})),
        EmbeddingBagOffsetsSumParams(
            ov::PartialShape{5, 2},
            ov::element::f64,
            std::vector<double>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
            ov::PartialShape{3, 2},
            ov::element::f64,
            std::vector<double>{-2.1, -2.4, -0.2, -0.6, -0.2, 0.8},
            std::make_shared<ngraph::opset1::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ngraph::opset1::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2}),
            std::make_shared<ngraph::opset1::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{0})),
        EmbeddingBagOffsetsSumParams(
            ov::PartialShape{5, 2},
            ov::element::i32,
            std::vector<int32_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
            ov::PartialShape{3, 2},
            ov::element::i32,
            std::vector<int32_t>{-6, -4, 0, 0, 2, 18},
            std::make_shared<ngraph::opset1::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ngraph::opset1::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2})),
        EmbeddingBagOffsetsSumParams(
            ov::PartialShape{5, 2},
            ov::element::u32,
            std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
            ov::PartialShape{3, 2},
            ov::element::u32,
            std::vector<uint32_t>{6, 8, 3, 4, 16, 18},
            std::make_shared<ngraph::opset1::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ngraph::opset1::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2}),
            std::make_shared<ngraph::opset1::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{1})),
        EmbeddingBagOffsetsSumParams(
            ov::PartialShape{5, 2},
            ov::element::f16,
            std::vector<float16>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
            ov::PartialShape{3, 2},
            ov::element::f16,
            std::vector<float16>{-2.1, -2.4, 0, 0, -0.2, 0.8},
            std::make_shared<ngraph::opset1::Constant>(element::i64, ov::Shape({4}), std::vector<int64_t>{0, 2, 3, 4}),
            std::make_shared<ngraph::opset1::Constant>(element::i64, ov::Shape({3}), std::vector<int64_t>{0, 2, 2})),
        EmbeddingBagOffsetsSumParams(
            ov::PartialShape{5, 2},
            ov::element::i64,
            std::vector<int64_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
            ov::PartialShape{3, 2},
            ov::element::i64,
            std::vector<int64_t>{-6, -4, -1, 2, 2, 18},
            std::make_shared<ngraph::opset1::Constant>(element::i64, ov::Shape({4}), std::vector<int64_t>{0, 2, 3, 4}),
            std::make_shared<ngraph::opset1::Constant>(element::i64, ov::Shape({3}), std::vector<int64_t>{0, 2, 2}),
            std::make_shared<ngraph::opset1::Constant>(element::i64, ov::Shape(), std::vector<int64_t>{0})),
        EmbeddingBagOffsetsSumParams(
            ov::PartialShape{5, 2},
            ov::element::i8,
            std::vector<int8_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
            ov::PartialShape{3, 2},
            ov::element::i8,
            std::vector<int8_t>{-12, -8, -1, 2, 4, 36},
            std::make_shared<ngraph::opset1::Constant>(element::i64, ov::Shape({4}), std::vector<int64_t>{0, 2, 3, 4}),
            std::make_shared<ngraph::opset1::Constant>(element::i64, ov::Shape({3}), std::vector<int64_t>{0, 2, 2}),
            std::make_shared<ngraph::opset1::Constant>(element::i64, ov::Shape(), std::vector<int64_t>{0}),
            std::make_shared<ngraph::opset1::Constant>(element::i8, ov::Shape({4}), std::vector<int8_t>{2, 2, 2, 2})),
        EmbeddingBagOffsetsSumParams(
            ov::PartialShape{5, 2},
            ov::element::u8,
            std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
            ov::PartialShape{3, 2},
            ov::element::u8,
            std::vector<uint8_t>{6, 8, 1, 2, 16, 18},
            std::make_shared<ngraph::opset1::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ngraph::opset1::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2}),
            std::make_shared<ngraph::opset1::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{0}))),
    ReferenceEmbeddingBagOffsetsSumLayerTest::getTestCaseName);
