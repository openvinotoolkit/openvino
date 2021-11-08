// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "engines_util/test_case.hpp"
#include "engines_util/test_engines.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

namespace {
template <typename dataType>
struct BatchToSpaceParams {
    using Data = test::NDArrayBase<dataType>;
    using BlockShape = test::NDArrayBase<int64_t>;
    using Crops = test::NDArrayBase<int64_t>;

    BatchToSpaceParams(Data in_data, BlockShape block_shape, Crops crops_begin, Crops crops_end, Data expected_output)
        : m_data{std::move(in_data)},
          m_block_shape{std::move(block_shape)},
          m_crops_begin{std::move(crops_begin)},
          m_crops_end{std::move(crops_end)},
          m_expected_output{std::move(expected_output)} {}

    Data m_data;
    BlockShape m_block_shape;
    Crops m_crops_begin;
    Crops m_crops_end;
    Data m_expected_output;
};

template <typename dataType>
static void BatchToSpaceTestExecute(const BatchToSpaceParams<dataType>& params) {
    const auto data = make_shared<op::Parameter>(element::from<dataType>(), params.m_data.get_shape());

    const auto block_shape =
        op::Constant::create(element::i64, params.m_block_shape.get_shape(), params.m_block_shape.get_vector());

    const auto crops_begin =
        op::Constant::create(element::i64, params.m_crops_begin.get_shape(), params.m_crops_begin.get_vector());

    const auto crops_end =
        op::Constant::create(element::i64, params.m_crops_end.get_shape(), params.m_crops_end.get_vector());

    const auto batch_to_space = make_shared<op::v1::BatchToSpace>(data, block_shape, crops_begin, crops_end);

    auto f = make_shared<Function>(batch_to_space, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input(params.m_data.get_vector());
    test_case.add_expected_output(params.m_expected_output.get_vector());
    test_case.run_with_tolerance_as_fp(1e-4f);
}

class BatchToSpaceTestFloat : public testing::TestWithParam<BatchToSpaceParams<float>> {};
}  // namespace

NGRAPH_TEST_P(${BACKEND_NAME}, BatchToSpaceTestFloat, BatchToSpaceTestFloatCases) {
    BatchToSpaceTestExecute(GetParam());
}

const test::NDArray<float, 2> input_with_shape_4x3(
    {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}});

const test::NDArray<int64_t, 1> zero_crops_2d({0, 0});

NGRAPH_INSTANTIATE_TEST_SUITE_P(${BACKEND_NAME},
                                batch_to_space_2d_without_crops,
                                BatchToSpaceTestFloat,
                                testing::Values(BatchToSpaceParams<float>{
                                    input_with_shape_4x3,
                                    test::NDArray<int64_t, 1>({1, 2}),
                                    zero_crops_2d,
                                    zero_crops_2d,
                                    test::NDArray<float, 2>({{1.0f, 7.0f, 2.0f, 8.0f, 3.0f, 9.0f},
                                                             {4.0f, 10.0f, 5.0f, 11.0f, 6.0f, 12.0f}})}));

const test::NDArray<float, 4> input_with_shape_4x1x1x3(
    {{{{1.0f, 2.0f, 3.0f}}}, {{{4.0f, 5.0f, 6.0f}}}, {{{7.0f, 8.0f, 9.0f}}}, {{{10.0f, 11.0f, 12.0f}}}});

const test::NDArray<float, 4> input_with_shape_4x1x2x3({{{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}},
                                                        {{{7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}}},
                                                        {{{13.0f, 14.0f, 15.0f}, {16.0f, 17.0f, 18.0f}}},
                                                        {{{19.0f, 20.0f, 21.0f}, {22.0f, 23.0f, 24.0f}}}});

const test::NDArray<int64_t, 1> zero_crops_4d({0, 0, 0, 0});

NGRAPH_INSTANTIATE_TEST_SUITE_P(
    ${BACKEND_NAME},
    batch_to_space_4d_without_crops,
    BatchToSpaceTestFloat,
    testing::Values(
        BatchToSpaceParams<float>{input_with_shape_4x1x1x3,
                                  test::NDArray<int64_t, 1>({1, 1, 1, 2}),
                                  zero_crops_4d,
                                  zero_crops_4d,
                                  test::NDArray<float, 4>({{{{1.0f, 7.0f, 2.0f, 8.0f, 3.0f, 9.0f}}},
                                                           {{{4.0f, 10.0f, 5.0f, 11.0f, 6.0f, 12.0f}}}})},
        BatchToSpaceParams<float>{input_with_shape_4x1x1x3,
                                  test::NDArray<int64_t, 1>({1, 1, 2, 1}),
                                  zero_crops_4d,
                                  zero_crops_4d,
                                  test::NDArray<float, 4>({{{{1.0f, 2.0f, 3.0f}, {7.0f, 8.0f, 9.0f}}},
                                                           {{{4.0f, 5.0f, 6.0f}, {10.0f, 11.0f, 12.0f}}}})},
        BatchToSpaceParams<float>{input_with_shape_4x1x1x3,
                                  test::NDArray<int64_t, 1>({1, 1, 2, 2}),
                                  zero_crops_4d,
                                  zero_crops_4d,
                                  test::NDArray<float, 4>({{{{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f},
                                                             {7.0f, 10.0f, 8.0f, 11.0f, 9.0f, 12.0f}}}})},
        BatchToSpaceParams<float>{
            input_with_shape_4x1x2x3,
            test::NDArray<int64_t, 1>({1, 1, 1, 2}),
            zero_crops_4d,
            zero_crops_4d,
            test::NDArray<float, 4>(
                {{{{1.0f, 13.0f, 2.0f, 14.0f, 3.0f, 15.0f}, {4.0f, 16.0f, 5.0f, 17.0f, 6.0f, 18.0f}}},
                 {{{7.0f, 19.0f, 8.0f, 20.0f, 9.0f, 21.0f}, {10.0f, 22.0f, 11.0f, 23.0f, 12.0f, 24.0f}}}})},
        BatchToSpaceParams<float>{
            input_with_shape_4x1x2x3,
            test::NDArray<int64_t, 1>({1, 1, 2, 1}),
            zero_crops_4d,
            zero_crops_4d,
            test::NDArray<float, 4>(
                {{{{1.0f, 2.0f, 3.0f}, {13.0f, 14.0f, 15.0f}, {4.0f, 5.0f, 6.0f}, {16.0f, 17.0f, 18.0f}}},
                 {{{7.0f, 8.0f, 9.0f}, {19.0f, 20.0f, 21.0f}, {10.0f, 11.0f, 12.0f}, {22.0f, 23.0f, 24.0f}}}})},
        BatchToSpaceParams<float>{input_with_shape_4x1x2x3,
                                  test::NDArray<int64_t, 1>({1, 1, 2, 2}),
                                  zero_crops_4d,
                                  zero_crops_4d,
                                  test::NDArray<float, 4>({{{{1.0f, 7.0f, 2.0f, 8.0f, 3.0f, 9.0f},
                                                             {13.0f, 19.0f, 14.0f, 20.0f, 15.0f, 21.0f},
                                                             {4.0f, 10.0f, 5.0f, 11.0f, 6.0f, 12.0f},
                                                             {16.0f, 22.0f, 17.0f, 23.0f, 18.0f, 24.0f}}}})}));

NGRAPH_INSTANTIATE_TEST_SUITE_P(
    ${BACKEND_NAME},
    batch_to_space_4d_crops,
    BatchToSpaceTestFloat,
    testing::Values(BatchToSpaceParams<float>{input_with_shape_4x1x2x3,
                                              test::NDArray<int64_t, 1>({1, 1, 2, 2}),
                                              test::NDArray<int64_t, 1>({0, 0, 0, 0}),
                                              test::NDArray<int64_t, 1>({0, 0, 0, 2}),
                                              test::NDArray<float, 4>({{{{1.0f, 7.0f, 2.0f, 8.0f},
                                                                         {13.0f, 19.0f, 14.0f, 20.0f},
                                                                         {4.0f, 10.0f, 5.0f, 11.0f},
                                                                         {16.0f, 22.0f, 17.0f, 23.0f}}}})},
                    BatchToSpaceParams<float>{input_with_shape_4x1x2x3,
                                              test::NDArray<int64_t, 1>({1, 1, 2, 2}),
                                              test::NDArray<int64_t, 1>({0, 0, 0, 2}),
                                              test::NDArray<int64_t, 1>({0, 0, 0, 0}),
                                              test::NDArray<float, 4>({{{{2.0f, 8.0f, 3.0f, 9.0f},
                                                                         {14.0f, 20.0f, 15.0f, 21.0f},
                                                                         {5.0f, 11.0f, 6.0f, 12.0f},
                                                                         {17.0f, 23.0f, 18.0f, 24.0f}}}})},
                    BatchToSpaceParams<float>{input_with_shape_4x1x2x3,
                                              test::NDArray<int64_t, 1>({1, 1, 2, 2}),
                                              test::NDArray<int64_t, 1>({0, 0, 1, 0}),
                                              test::NDArray<int64_t, 1>({0, 0, 1, 0}),
                                              test::NDArray<float, 4>({{{{13.0f, 19.0f, 14.0f, 20.0f, 15.0f, 21.0f},
                                                                         {4.0f, 10.0f, 5.0f, 11.0f, 6.0f, 12.0f}}}})}));
