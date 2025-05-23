// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_fill_empty_rows.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/op/constant.hpp"

namespace {
struct SparseFillEmptyRowsParams {
    SparseFillEmptyRowsParams(const reference_tests::Tensor& valuesTensor,
                              const reference_tests::Tensor& denseShapeTensor,
                              const reference_tests::Tensor& indicesTensor,
                              const reference_tests::Tensor& defaultValueTensor,
                              const reference_tests::Tensor& expectedIndicesTensor,
                              const reference_tests::Tensor& expectedValuesTensor,
                              const reference_tests::Tensor& expectedEmptyRowIndicatorTensor)
        : valuesTensor(valuesTensor),
          denseShapeTensor(denseShapeTensor),
          indicesTensor(indicesTensor),
          defaultValueTensor(defaultValueTensor),
          expectedIndicesTensor(expectedIndicesTensor),
          expectedValuesTensor(expectedValuesTensor),
          expectedEmptyRowIndicatorTensor(expectedEmptyRowIndicatorTensor) {}

    reference_tests::Tensor valuesTensor;
    reference_tests::Tensor denseShapeTensor;
    reference_tests::Tensor indicesTensor;
    reference_tests::Tensor defaultValueTensor;
    reference_tests::Tensor expectedIndicesTensor;
    reference_tests::Tensor expectedValuesTensor;
    reference_tests::Tensor expectedEmptyRowIndicatorTensor;
};

class ReferenceSparseFillEmptyRowsV16LayerTest : public testing::TestWithParam<SparseFillEmptyRowsParams>,
                                                 public reference_tests::CommonReferenceTest {
protected:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.valuesTensor.data,
                     params.denseShapeTensor.data,
                     params.indicesTensor.data,
                     params.defaultValueTensor.data};
        refOutData = {params.expectedIndicesTensor.data,
                      params.expectedValuesTensor.data,
                      params.expectedEmptyRowIndicatorTensor.data};
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<SparseFillEmptyRowsParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "valuesType=" << param.valuesTensor.type;
        result << "_valuesShape=" << param.valuesTensor.shape;
        result << "_denseShapeType=" << param.denseShapeTensor.type;
        result << "_denseShapeValues=" << testing::PrintToString(param.denseShapeTensor.data);
        result << "_indicesType=" << param.indicesTensor.type;
        result << "_indicesShape=" << param.indicesTensor.shape;
        result << "_defaultValue=" << testing::PrintToString(param.defaultValueTensor.data);
        return result.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const SparseFillEmptyRowsParams& params) {
        using ov::op::v0::Parameter;

        const auto values = std::make_shared<Parameter>(params.valuesTensor.type, params.valuesTensor.shape);
        const auto dense_shape =
            std::make_shared<Parameter>(params.denseShapeTensor.type, params.denseShapeTensor.shape);
        const auto indices = std::make_shared<Parameter>(params.indicesTensor.type, params.indicesTensor.shape);
        const auto default_value =
            std::make_shared<Parameter>(params.defaultValueTensor.type, params.defaultValueTensor.shape);

        const auto sparseFillEmptyRows =
            std::make_shared<ov::op::v16::SparseFillEmptyRows>(values, dense_shape, indices, default_value);

        return std::make_shared<ov::Model>(ov::OutputVector{sparseFillEmptyRows->output(0),
                                                            sparseFillEmptyRows->output(1),
                                                            sparseFillEmptyRows->output(2)},
                                           ov::ParameterVector{values, dense_shape, indices, default_value});
    }
};

TEST_P(ReferenceSparseFillEmptyRowsV16LayerTest, CompareWithRefs) {
    Exec();
}

template <ov::element::Type_t T, ov::element::Type_t T_idx>
std::vector<SparseFillEmptyRowsParams> generateSparseFillEmptyRowsParams() {
    using T_D = typename ov::element_type_traits<T>::value_type;
    using T_I = typename ov::element_type_traits<T_idx>::value_type;
    using reference_tests::Tensor;

    std::vector<SparseFillEmptyRowsParams> params{
        // No empty rows
        SparseFillEmptyRowsParams(
            Tensor({3}, T, std::vector<T_D>{1, 2, 3}),                        // values
            Tensor({2}, T_idx, std::vector<T_I>{3, 4}),                       // dense_shape
            Tensor({3, 2}, T_idx, std::vector<T_I>{0, 0, 1, 0, 2, 0}),        // indices
            Tensor({}, T, std::vector<T_D>{-1}),                              // default_value
            Tensor({3, 2}, T_idx, std::vector<T_I>{0, 0, 1, 0, 2, 0}),        // expected_indices
            Tensor({3}, T, std::vector<T_D>{1, 2, 3}),                        // expected_values
            Tensor({3}, ov::element::boolean, std::vector<uint8_t>{0, 0, 0})  // expected_empty_row_indicator
            ),

        // One empty row in the middle
        SparseFillEmptyRowsParams(
            Tensor({3}, T, std::vector<T_D>{1, 2, 3}),                           // values
            Tensor({2}, T_idx, std::vector<T_I>{4, 4}),                          // dense_shape
            Tensor({3, 2}, T_idx, std::vector<T_I>{0, 0, 1, 0, 3, 0}),           // indices
            Tensor({}, T, std::vector<T_D>{-1}),                                 // default_value
            Tensor({4, 2}, T_idx, std::vector<T_I>{0, 0, 1, 0, 2, 0, 3, 0}),     // expected_indices
            Tensor({4}, T, std::vector<T_D>{1, 2, -1, 3}),                       // expected_values
            Tensor({4}, ov::element::boolean, std::vector<uint8_t>{0, 0, 1, 0})  // expected_empty_row_indicator
            ),

        // Multiple empty rows
        SparseFillEmptyRowsParams(
            Tensor({2}, T, std::vector<T_D>{1, 2}),                                 // values
            Tensor({2}, T_idx, std::vector<T_I>{5, 3}),                             // dense_shape
            Tensor({2, 2}, T_idx, std::vector<T_I>{0, 0, 4, 0}),                    // indices
            Tensor({}, T, std::vector<T_D>{-1}),                                    // default_value
            Tensor({5, 2}, T_idx, std::vector<T_I>{0, 0, 1, 0, 2, 0, 3, 0, 4, 0}),  // expected_indices
            Tensor({5}, T, std::vector<T_D>{1, -1, -1, -1, 2}),                     // expected_values
            Tensor({5}, ov::element::boolean, std::vector<uint8_t>{0, 1, 1, 1, 0})  // expected_empty_row_indicator
            ),

        // All rows empty
        SparseFillEmptyRowsParams(
            Tensor({0}, T, std::vector<T_D>{}),                               // values
            Tensor({2}, T_idx, std::vector<T_I>{3, 4}),                       // dense_shape
            Tensor({0, 2}, T_idx, std::vector<T_I>{}),                        // indices
            Tensor({}, T, std::vector<T_D>{-1}),                              // default_value
            Tensor({3, 2}, T_idx, std::vector<T_I>{0, 0, 1, 0, 2, 0}),        // expected_indices
            Tensor({3}, T, std::vector<T_D>{-1, -1, -1}),                     // expected_values
            Tensor({3}, ov::element::boolean, std::vector<uint8_t>{1, 1, 1})  // expected_empty_row_indicator
            ),

        // Non-zero column indices for empty rows
        SparseFillEmptyRowsParams(
            Tensor({4}, T, std::vector<T_D>{1, 2, 3, 4}),                           // values
            Tensor({2}, T_idx, std::vector<T_I>{5, 6}),                             // dense_shape
            Tensor({4, 2}, T_idx, std::vector<T_I>{0, 1, 1, 2, 3, 3, 4, 4}),        // indices
            Tensor({}, T, std::vector<T_D>{99}),                                    // default_value
            Tensor({5, 2}, T_idx, std::vector<T_I>{0, 1, 1, 2, 2, 0, 3, 3, 4, 4}),  // expected_indices
            Tensor({5}, T, std::vector<T_D>{1, 2, 99, 3, 4}),                       // expected_values
            Tensor({5}, ov::element::boolean, std::vector<uint8_t>{0, 0, 1, 0, 0})  // expected_empty_row_indicator
            )};

    return params;
}

std::vector<SparseFillEmptyRowsParams> generateSparseFillEmptyRowsV16CombinedParams() {
    using ov::element::Type_t;
    const std::vector<std::vector<SparseFillEmptyRowsParams>> SparseFillEmptyRowsTypeParams{
        generateSparseFillEmptyRowsParams<Type_t::i32, Type_t::i32>(),
        generateSparseFillEmptyRowsParams<Type_t::i64, Type_t::i32>(),
        generateSparseFillEmptyRowsParams<Type_t::f32, Type_t::i32>(),
        generateSparseFillEmptyRowsParams<Type_t::i32, Type_t::i64>(),
        generateSparseFillEmptyRowsParams<Type_t::i64, Type_t::i64>(),
        generateSparseFillEmptyRowsParams<Type_t::f32, Type_t::i64>()};

    std::vector<SparseFillEmptyRowsParams> combinedParams;
    for (const auto& params : SparseFillEmptyRowsTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_SparseFillEmptyRows_With_Hardcoded_Refs,
                         ReferenceSparseFillEmptyRowsV16LayerTest,
                         testing::ValuesIn(generateSparseFillEmptyRowsV16CombinedParams()),
                         ReferenceSparseFillEmptyRowsV16LayerTest::getTestCaseName);
}  // namespace
