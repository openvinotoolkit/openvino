// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/embeddingbag_offsets.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

struct EmbeddingBagOffsetsParams {
    template <class IT>
    EmbeddingBagOffsetsParams(const ov::Shape& iShape,
                              const ov::element::Type& iType,
                              const std::vector<IT>& iValues,
                              const ov::Shape& oShape,
                              const ov::element::Type& oType,
                              const std::vector<IT>& oValues,
                              const std::shared_ptr<ov::op::v0::Constant>& indices,
                              const std::shared_ptr<ov::op::v0::Constant>& offsets,
                              const ov::op::v15::EmbeddingBagOffsets::Reduction reduction,
                              const std::shared_ptr<ov::op::v0::Constant>& default_index = nullptr,
                              const std::shared_ptr<ov::op::v0::Constant>& per_sample_weights = nullptr)
        : _iShape(iShape),
          _iType(iType),
          _iData(CreateTensor(iShape, iType, iValues)),
          _refShape(oShape),
          _refType(oType),
          _refData(CreateTensor(oShape, oType, oValues)) {
        _indices = indices;
        _offsets = offsets;
        _reduction = reduction;
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
    std::shared_ptr<ov::op::v0::Constant> _offsets;
    ov::op::v15::EmbeddingBagOffsets::Reduction _reduction;
    std::shared_ptr<ov::op::v0::Constant> _defaultIndex;      // Optional, default filled zero.
    std::shared_ptr<ov::op::v0::Constant> _perSampleWeights;  // Optional, default is tensor of ones.
};

class ReferenceEmbeddingBagOffsetsLayerTest : public testing::TestWithParam<EmbeddingBagOffsetsParams>,
                                              public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params._iShape,
                                  params._iType,
                                  params._indices,
                                  params._offsets,
                                  params._reduction,
                                  params._defaultIndex,
                                  params._perSampleWeights);
        inputData = {params._iData};
        refOutData = {params._refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<EmbeddingBagOffsetsParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "_iShape=" << param._iShape << "_";
        result << "_iType=" << param._iType << "_";
        result << "_refShape=" << param._refShape << "_";
        result << "_refType=" << param._refType << "_";
        result << "_reduction=" << param._reduction;
        return result.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const Shape& input_shape,
                                                     const element::Type& input_type,
                                                     const std::shared_ptr<ov::op::v0::Constant> indices,
                                                     const std::shared_ptr<ov::op::v0::Constant> offsets,
                                                     const ov::op::v15::EmbeddingBagOffsets::Reduction reduction,
                                                     const std::shared_ptr<ov::op::v0::Constant> default_index,
                                                     const std::shared_ptr<ov::op::v0::Constant> per_sample_weights) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);

        if (default_index) {
            if (per_sample_weights) {
                const auto ess = std::make_shared<op::v15::EmbeddingBagOffsets>(in,
                                                                                indices,
                                                                                offsets,
                                                                                default_index,
                                                                                per_sample_weights,
                                                                                reduction);
                return std::make_shared<Model>(NodeVector{ess}, ParameterVector{in});
            } else {
                const auto ess =
                    std::make_shared<op::v15::EmbeddingBagOffsets>(in, indices, offsets, default_index, reduction);
                return std::make_shared<Model>(NodeVector{ess}, ParameterVector{in});
            }
        } else {
            const auto ess = std::make_shared<op::v15::EmbeddingBagOffsets>(in, indices, offsets, reduction);
            return std::make_shared<Model>(NodeVector{ess}, ParameterVector{in});
        }
    }
};

TEST_P(ReferenceEmbeddingBagOffsetsLayerTest, CompareWithRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_EmbeddingBagOffsets_With_Hardcoded_Refs,
    ReferenceEmbeddingBagOffsetsLayerTest,
    ::testing::Values(
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::f32,
            std::vector<float>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
            ov::Shape{3, 2},
            ov::element::f32,
            {-1.05f, -1.2f, -0.2f, -0.6f, -0.1f, 0.4f},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::SUM,
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{0}),
            std::make_shared<ov::op::v0::Constant>(element::f32,
                                                   ov::Shape({4}),
                                                   std::vector<float>{0.5, 0.5, 0.5, 0.5})),
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::f64,
            std::vector<double>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
            ov::Shape{3, 2},
            ov::element::f64,
            std::vector<double>{-2.1, -2.4, -0.2, -0.6, -0.2, 0.8},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::SUM,
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{0})),
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::i32,
            std::vector<int32_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::i32,
            std::vector<int32_t>{-6, -4, 0, 0, 2, 18},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::SUM),
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::u32,
            std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::u32,
            std::vector<uint32_t>{6, 8, 3, 4, 16, 18},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::SUM,
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{1})),
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::f16,
            std::vector<float16>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
            ov::Shape{3, 2},
            ov::element::f16,
            std::vector<float16>{-2.1, -2.4, 0, 0, -0.2, 0.8},
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({4}), std::vector<int64_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({3}), std::vector<int64_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::SUM),
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::i64,
            std::vector<int64_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::i64,
            std::vector<int64_t>{-6, -4, -1, 2, 2, 18},
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({4}), std::vector<int64_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({3}), std::vector<int64_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::SUM,
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape(), std::vector<int64_t>{0})),
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::i8,
            std::vector<int8_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::i8,
            std::vector<int8_t>{-12, -8, -1, 2, 4, 36},
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({4}), std::vector<int64_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({3}), std::vector<int64_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::SUM,
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape(), std::vector<int64_t>{0}),
            std::make_shared<ov::op::v0::Constant>(element::i8, ov::Shape({4}), std::vector<int8_t>{2, 2, 2, 2})),
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::u8,
            std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::u8,
            std::vector<uint8_t>{6, 8, 1, 2, 16, 18},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::SUM,
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{0})),
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::i16,
            std::vector<int16_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::i16,
            std::vector<int16_t>{-6, -4, 0, 0, 2, 18},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::SUM,
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{-1})),
        EmbeddingBagOffsetsParams(
            ov::Shape{6, 2},
            ov::element::i32,
            std::vector<int32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
            ov::Shape{3, 2},
            ov::element::i32,
            std::vector<int32_t>{12, 16, 0, 0, 32, 36},
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({4}), std::vector<int64_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({3}), std::vector<int64_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::SUM,
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape(), std::vector<int32_t>{-1}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{2, 2, 2, 2})),
        // Reduction MEAN
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::f64,
            std::vector<double>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
            ov::Shape{3, 2},
            ov::element::f64,
            std::vector<double>{-1.05, -1.2, -0.2, -0.6, -0.1, 0.4},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::MEAN,
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{0})),
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::i32,
            std::vector<int32_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::i32,
            std::vector<int32_t>{-3, -2, 0, 0, 1, 9},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::MEAN),
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::u32,
            std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::u32,
            std::vector<uint32_t>{3, 4, 3, 4, 8, 9},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::MEAN,
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{1})),
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::f16,
            std::vector<float16>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
            ov::Shape{3, 2},
            ov::element::f16,
            std::vector<float16>{-1.05, -1.2, 0, 0, -0.1, 0.4},
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({4}), std::vector<int64_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({3}), std::vector<int64_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::MEAN),
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::i64,
            std::vector<int64_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::i64,
            std::vector<int64_t>{-3, -2, -1, 2, 1, 9},
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({4}), std::vector<int64_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape({3}), std::vector<int64_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::MEAN,
            std::make_shared<ov::op::v0::Constant>(element::i64, ov::Shape(), std::vector<int64_t>{0})),
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::u8,
            std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::u8,
            std::vector<uint8_t>{3, 4, 1, 2, 8, 9},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::MEAN,
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{0})),
        EmbeddingBagOffsetsParams(
            ov::Shape{5, 2},
            ov::element::f32,
            std::vector<float>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
            ov::Shape{3, 2},
            ov::element::f32,
            std::vector<float>{-3, -2, 0, 0, 1, 9},
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({4}), std::vector<int32_t>{0, 2, 3, 4}),
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape({3}), std::vector<int32_t>{0, 2, 2}),
            ov::op::v15::EmbeddingBagOffsets::Reduction::MEAN,
            std::make_shared<ov::op::v0::Constant>(element::i32, ov::Shape(), std::vector<int32_t>{-1}))),
    ReferenceEmbeddingBagOffsetsLayerTest::getTestCaseName);
