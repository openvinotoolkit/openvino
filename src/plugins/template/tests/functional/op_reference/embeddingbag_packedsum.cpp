// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/embeddingbag_packedsum.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

struct EmbeddingBagPackedSumParams {
    template <class IT>
    EmbeddingBagPackedSumParams(const ov::Shape& iShape,
                                const ov::element::Type& iType,
                                const std::vector<IT>& iValues,
                                const ov::Shape& oShape,
                                const ov::element::Type& oType,
                                const std::vector<IT>& oValues,
                                const std::shared_ptr<ov::op::v0::Constant>& indices,
                                const std::shared_ptr<ov::op::v0::Constant>& per_sample_weights = nullptr)
        : _iShape(iShape),
          _iType(iType),
          _iData(CreateTensor(iShape, iType, iValues)),
          _refShape(oShape),
          _refType(oType),
          _refData(CreateTensor(oShape, oType, oValues)) {
        _indices = indices;
        _perSampleWeights = per_sample_weights;
    }
    ov::Shape _iShape;
    ov::element::Type _iType;
    ov::Tensor _iData;

    ov::Shape _refShape;
    ov::element::Type _refType;
    ov::Tensor _refData;

    std::shared_ptr<ov::op::v0::Constant> _indices;
    std::shared_ptr<ov::op::v0::Constant> _perSampleWeights;  // Optional, default is tensor of ones.
};

class ReferenceEmbeddingBagPackedSumLayerTest : public testing::TestWithParam<EmbeddingBagPackedSumParams>,
                                                public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params._iShape, params._iType, params._indices, params._perSampleWeights);
        inputData = {params._iData};
        refOutData = {params._refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<EmbeddingBagPackedSumParams>& obj) {
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
                                                 const std::shared_ptr<ov::op::v0::Constant> per_sample_weights) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);

        if (per_sample_weights) {
            const auto ess = std::make_shared<op::v3::EmbeddingBagPackedSum>(in, indices, per_sample_weights);
            return std::make_shared<Model>(NodeVector{ess}, ParameterVector{in});
        } else {
            const auto ess = std::make_shared<op::v3::EmbeddingBagPackedSum>(in, indices);
            return std::make_shared<Model>(NodeVector{ess}, ParameterVector{in});
        }
    }
};

TEST_P(ReferenceEmbeddingBagPackedSumLayerTest, CompareWithRefs) {
    Exec();
}

template <class T>
inline std::shared_ptr<ov::op::v0::Constant> CreateConstant(const std::vector<std::vector<T>>& val,
                                                            const ov::element::Type& element_type) {
    if (val.size() > 0) {
        ov::Shape i_shape({val.size(), val[0].size()});

        size_t i_size = ov::shape_size(i_shape);
        std::vector<T> i_values(i_size);

        for (size_t i = 0; i < i_shape[0]; i++) {
            for (size_t j = 0; j < i_shape[1]; j++) {
                i_values[i * i_shape[1] + j] = val[i][j];
            }
        }

        return std::make_shared<ov::op::v0::Constant>(element_type, i_shape, i_values);
    } else {
        return std::make_shared<ov::op::v0::Constant>(element_type, ov::Shape(), std::vector<T>());
    }
}

INSTANTIATE_TEST_SUITE_P(
    smoke_EmbeddingBagPackedSum_With_Hardcoded_Refs,
    ReferenceEmbeddingBagPackedSumLayerTest,
    ::testing::Values(
        EmbeddingBagPackedSumParams(ov::Shape{5, 2},
                                    ov::element::f32,
                                    std::vector<float>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
                                    ov::Shape{3, 2},
                                    ov::element::f32,
                                    std::vector<float>{-1.05f, -1.2f, -1.f, -1.1f, -0.1f, 0.4f},
                                    CreateConstant<int32_t>({{0, 2}, {1, 2}, {3, 4}}, element::i32),
                                    CreateConstant<float>({{0.5, 0.5}, {0.5, 0.5}, {0.5, 0.5}}, element::f32)),
        EmbeddingBagPackedSumParams(ov::Shape{5, 2},
                                    ov::element::f64,
                                    std::vector<double>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
                                    ov::Shape{3, 2},
                                    ov::element::f64,
                                    std::vector<double>{-2.1, -2.4, -2.0, -2.2, -0.2, 0.8},
                                    CreateConstant<int32_t>({{0, 2}, {1, 2}, {3, 4}}, element::i32)),
        EmbeddingBagPackedSumParams(ov::Shape{5, 2},
                                    ov::element::i32,
                                    std::vector<int32_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
                                    ov::Shape{3, 2},
                                    ov::element::i32,
                                    std::vector<int32_t>{-6, -4, -2, -2, 2, 18},
                                    CreateConstant<int32_t>({{0, 2}, {1, 2}, {3, 4}}, element::i32)),
        EmbeddingBagPackedSumParams(ov::Shape{5, 2},
                                    ov::element::u32,
                                    std::vector<uint32_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                    ov::Shape{3, 2},
                                    ov::element::u32,
                                    std::vector<uint32_t>{6, 8, 8, 10, 16, 18},
                                    CreateConstant<int32_t>({{0, 2}, {1, 2}, {3, 4}}, element::i32)),
        EmbeddingBagPackedSumParams(ov::Shape{5, 2},
                                    ov::element::f16,
                                    std::vector<float16>{-0.2, -0.6, -0.1, -0.4, -1.9, -1.8, -1., 1.5, 0.8, -0.7},
                                    ov::Shape{3, 2},
                                    ov::element::f16,
                                    std::vector<float16>{-2.1, -2.4, -2.0, -2.2, -0.2, 0.8},
                                    CreateConstant<int64_t>({{0, 2}, {1, 2}, {3, 4}}, element::i64)),
        EmbeddingBagPackedSumParams(ov::Shape{5, 2},
                                    ov::element::i64,
                                    std::vector<int64_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
                                    ov::Shape{3, 2},
                                    ov::element::i64,
                                    std::vector<int64_t>{-6, -4, -2, -2, 2, 18},
                                    CreateConstant<int64_t>({{0, 2}, {1, 2}, {3, 4}}, element::i64)),
        EmbeddingBagPackedSumParams(ov::Shape{5, 2},
                                    ov::element::i8,
                                    std::vector<int8_t>{-1, 2, 3, 4, -5, -6, -7, 8, 9, 10},
                                    ov::Shape{3, 2},
                                    ov::element::i8,
                                    std::vector<int8_t>{-12, -8, -4, -4, 4, 36},
                                    CreateConstant<int64_t>({{0, 2}, {1, 2}, {3, 4}}, element::i64),
                                    CreateConstant<int8_t>({{2, 2}, {2, 2}, {2, 2}}, element::i8)),
        EmbeddingBagPackedSumParams(ov::Shape{5, 2},
                                    ov::element::u8,
                                    std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                    ov::Shape{3, 2},
                                    ov::element::u8,
                                    std::vector<uint8_t>{6, 8, 8, 10, 16, 18},
                                    CreateConstant<int64_t>({{0, 2}, {1, 2}, {3, 4}}, element::i64))),
    ReferenceEmbeddingBagPackedSumLayerTest::getTestCaseName);
