// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "runtime/backend.hpp"
#include "util/all_close.hpp"
#include "util/all_close_f.hpp"
#include "util/known_element_types.hpp"
#include "util/ndarray.hpp"
#include "util/test_control.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";

namespace
{
    template <typename ValueType>
    struct Params
    {
        using Data = ::ngraph::test::NDArrayBase<ValueType>;
        using Pads = ::ngraph::test::NDArrayBase<int64_t>;

        Params(Data input_data,
               Pads pads_begin,
               Pads pads_end,
               Data expected_output,
               op::PadMode pad_mode,
               ValueType constant_value)
            : input_data{std::move(input_data)}
            , pads_begin{std::move(pads_begin)}
            , pads_end{std::move(pads_end)}
            , expected_output{std::move(expected_output)}
            , pad_mode{pad_mode}
            , use_const_value{true}
            , constant_value{constant_value}
        {
        }

        Params(Data input_data,
               Pads pads_begin,
               Pads pads_end,
               Data expected_output,
               op::PadMode pad_mode)
            : input_data{std::move(input_data)}
            , pads_begin{std::move(pads_begin)}
            , pads_end{std::move(pads_end)}
            , expected_output{std::move(expected_output)}
            , pad_mode{pad_mode}
        {
        }

        Data input_data;
        Pads pads_begin;
        Pads pads_end;
        Data expected_output;
        op::PadMode pad_mode;
        bool use_const_value{false};
        ValueType constant_value{};
    };

    class PadBackendTest : public ::testing::TestWithParam<Params<float>>
    {
    public:
        static void execute_test(const Params<float>& params)
        {
            const auto data =
                make_shared<op::Parameter>(element::f32, params.input_data.get_shape());

            const auto pads_begin = op::Constant::create(
                element::i64, params.pads_begin.get_shape(), params.pads_begin.get_vector());

            const auto pads_end = op::Constant::create(
                element::i64, params.pads_end.get_shape(), params.pads_end.get_vector());

            auto f = [&] {
                if (params.use_const_value)
                {
                    // pad_value should be used only in CONSTANT mode
                    const auto pad_val =
                        op::Constant::create(element::f32, Shape{}, {params.constant_value});

                    return make_shared<Function>(
                        make_shared<op::v1::Pad>(
                            data, pads_begin, pads_end, pad_val, params.pad_mode),
                        ParameterVector{data});
                }

                return make_shared<Function>(
                    make_shared<op::v1::Pad>(data, pads_begin, pads_end, params.pad_mode),
                    ParameterVector{data});
            }();

            auto backend = runtime::Backend::create("${BACKEND_NAME}");

            // Create some tensors for input/output
            auto a = backend->create_tensor(element::f32, params.input_data.get_shape());
            copy_data(a, params.input_data.get_vector());
            auto result = backend->create_tensor(element::f32, params.expected_output.get_shape());

            auto handle = backend->compile(f);
            handle->call_with_validate({result}, {a});
            EXPECT_TRUE(test::all_close_f(params.expected_output.get_vector(),
                                          read_vector<float>(result),
                                          MIN_FLOAT_TOLERANCE_BITS));
        }
    };
} // namespace

NGRAPH_TEST_P(${BACKEND_NAME}, PadBackendTest, PadBackendTestForSpec)
{
    execute_test(GetParam());
}

NGRAPH_INSTANTIATE_TEST_SUITE_P(
    ${BACKEND_NAME},
    pad_1d_constant_const_value_provided,
    PadBackendTest,
    testing::Values(
        Params<float>{test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}),
                      test::NDArray<int64_t, 1>({4}),
                      test::NDArray<int64_t, 1>({5}),
                      test::NDArray<float, 1>(
                          {2112, 2112, 2112, 2112, 1, 2, 3, 4, 5, 6, 2112, 2112, 2112, 2112, 2112}),
                      op::PadMode::CONSTANT,
                      2112},
        Params<float>{test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}),
                      test::NDArray<int64_t, 1>({4}),
                      test::NDArray<int64_t, 1>({0}),
                      test::NDArray<float, 1>({2112, 2112, 2112, 2112, 1, 2, 3, 4, 5, 6}),
                      op::PadMode::CONSTANT,
                      2112},
        Params<float>{test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}),
                      test::NDArray<int64_t, 1>({0}),
                      test::NDArray<int64_t, 1>({3}),
                      test::NDArray<float, 1>({1, 2, 3, 4, 5, 6, 2112, 2112, 2112}),
                      op::PadMode::CONSTANT,
                      2112}));

NGRAPH_INSTANTIATE_TEST_SUITE_P(
    ${BACKEND_NAME},
    pad_1d_constant_use_default_const,
    PadBackendTest,
    testing::Values(
        Params<float>{test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}),
                      test::NDArray<int64_t, 1>({4}),
                      test::NDArray<int64_t, 1>({5}),
                      test::NDArray<float, 1>({0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0}),
                      op::PadMode::CONSTANT},
        Params<float>{test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}),
                      test::NDArray<int64_t, 1>({4}),
                      test::NDArray<int64_t, 1>({0}),
                      test::NDArray<float, 1>({0, 0, 0, 0, 1, 2, 3, 4, 5, 6}),
                      op::PadMode::CONSTANT},
        Params<float>{test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}),
                      test::NDArray<int64_t, 1>({0}),
                      test::NDArray<int64_t, 1>({3}),
                      test::NDArray<float, 1>({1, 2, 3, 4, 5, 6, 0, 0, 0}),
                      op::PadMode::CONSTANT}));

NGRAPH_INSTANTIATE_TEST_SUITE_P(
    ${BACKEND_NAME},
    pad_2d_constant_const_value_provided,
    PadBackendTest,
    testing::Values(Params<float>{test::NDArray<float, 2>({
                                      {1, 2},
                                      {3, 4},
                                  }),
                                  test::NDArray<int64_t, 1>({1, 2}),
                                  test::NDArray<int64_t, 1>({3, 4}),
                                  test::NDArray<float, 2>({
                                      {2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112},
                                      {2112, 2112, 1, 2, 2112, 2112, 2112, 2112},
                                      {2112, 2112, 3, 4, 2112, 2112, 2112, 2112},
                                      {2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112},
                                      {2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112},
                                      {2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112},
                                  }),
                                  op::PadMode::CONSTANT,
                                  2112},
                    Params<float>{test::NDArray<float, 2>({
                                      {1, 2},
                                      {3, 4},
                                  }),
                                  test::NDArray<int64_t, 1>({1, 2}),
                                  test::NDArray<int64_t, 1>({0, 0}),
                                  test::NDArray<float, 2>({
                                      {2112, 2112, 2112, 2112},
                                      {2112, 2112, 1, 2},
                                      {2112, 2112, 3, 4},
                                  }),
                                  op::PadMode::CONSTANT,
                                  2112},
                    Params<float>{test::NDArray<float, 2>({
                                      {1, 2},
                                      {3, 4},
                                  }),
                                  test::NDArray<int64_t, 1>({0, 0}),
                                  test::NDArray<int64_t, 1>({1, 2}),
                                  test::NDArray<float, 2>({
                                      {1, 2, 2112, 2112},
                                      {3, 4, 2112, 2112},
                                      {2112, 2112, 2112, 2112},
                                  }),
                                  op::PadMode::CONSTANT,
                                  2112}));

NGRAPH_INSTANTIATE_TEST_SUITE_P(${BACKEND_NAME},
                                pad_2d_constant_use_default_const,
                                PadBackendTest,
                                testing::Values(Params<float>{test::NDArray<float, 2>({
                                                                  {1, 2},
                                                                  {3, 4},
                                                              }),
                                                              test::NDArray<int64_t, 1>({1, 2}),
                                                              test::NDArray<int64_t, 1>({3, 4}),
                                                              test::NDArray<float, 2>({
                                                                  {0, 0, 0, 0, 0, 0, 0, 0},
                                                                  {0, 0, 1, 2, 0, 0, 0, 0},
                                                                  {0, 0, 3, 4, 0, 0, 0, 0},
                                                                  {0, 0, 0, 0, 0, 0, 0, 0},
                                                                  {0, 0, 0, 0, 0, 0, 0, 0},
                                                                  {0, 0, 0, 0, 0, 0, 0, 0},
                                                              }),
                                                              op::PadMode::CONSTANT},
                                                Params<float>{test::NDArray<float, 2>({
                                                                  {1, 2},
                                                                  {3, 4},
                                                              }),
                                                              test::NDArray<int64_t, 1>({1, 2}),
                                                              test::NDArray<int64_t, 1>({0, 0}),
                                                              test::NDArray<float, 2>({
                                                                  {0, 0, 0, 0},
                                                                  {0, 0, 1, 2},
                                                                  {0, 0, 3, 4},
                                                              }),
                                                              op::PadMode::CONSTANT},
                                                Params<float>{test::NDArray<float, 2>({
                                                                  {1, 2},
                                                                  {3, 4},
                                                              }),
                                                              test::NDArray<int64_t, 1>({0, 0}),
                                                              test::NDArray<int64_t, 1>({1, 2}),
                                                              test::NDArray<float, 2>({
                                                                  {1, 2, 0, 0},
                                                                  {3, 4, 0, 0},
                                                                  {0, 0, 0, 0},
                                                              }),
                                                              op::PadMode::CONSTANT}));

NGRAPH_INSTANTIATE_TEST_SUITE_P(
    ${BACKEND_NAME},
    pad_1d_edge,
    PadBackendTest,
    testing::Values(Params<float>{test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}),
                                  test::NDArray<int64_t, 1>({2}),
                                  test::NDArray<int64_t, 1>({3}),
                                  test::NDArray<float, 1>({1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6}),
                                  op::PadMode::EDGE},
                    Params<float>{test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}),
                                  test::NDArray<int64_t, 1>({1}),
                                  test::NDArray<int64_t, 1>({0}),
                                  test::NDArray<float, 1>({1, 1, 2, 3, 4, 5, 6}),
                                  op::PadMode::EDGE},
                    Params<float>{test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}),
                                  test::NDArray<int64_t, 1>({0}),
                                  test::NDArray<int64_t, 1>({2}),
                                  test::NDArray<float, 1>({1, 2, 3, 4, 5, 6, 6, 6}),
                                  op::PadMode::EDGE}));

NGRAPH_INSTANTIATE_TEST_SUITE_P(${BACKEND_NAME},
                                pad_2d_edge,
                                PadBackendTest,
                                testing::Values(Params<float>{test::NDArray<float, 2>({
                                                                  {1, 2},
                                                                  {3, 4},
                                                              }),
                                                              test::NDArray<int64_t, 1>({1, 2}),
                                                              test::NDArray<int64_t, 1>({2, 1}),
                                                              test::NDArray<float, 2>({
                                                                  {1, 1, 1, 2, 2},
                                                                  {1, 1, 1, 2, 2},
                                                                  {3, 3, 3, 4, 4},
                                                                  {3, 3, 3, 4, 4},
                                                                  {3, 3, 3, 4, 4},
                                                              }),
                                                              op::PadMode::EDGE},
                                                Params<float>{test::NDArray<float, 2>({
                                                                  {1, 2},
                                                                  {3, 4},
                                                              }),
                                                              test::NDArray<int64_t, 1>({1, 2}),
                                                              test::NDArray<int64_t, 1>({0, 0}),
                                                              test::NDArray<float, 2>({
                                                                  {1, 1, 1, 2},
                                                                  {1, 1, 1, 2},
                                                                  {3, 3, 3, 4},
                                                              }),
                                                              op::PadMode::EDGE},
                                                Params<float>{test::NDArray<float, 2>({
                                                                  {1, 2},
                                                                  {3, 4},
                                                              }),
                                                              test::NDArray<int64_t, 1>({0, 0}),
                                                              test::NDArray<int64_t, 1>({2, 1}),
                                                              test::NDArray<float, 2>({
                                                                  {1, 2, 2},
                                                                  {3, 4, 4},
                                                                  {3, 4, 4},
                                                                  {3, 4, 4},
                                                              }),
                                                              op::PadMode::EDGE}));

NGRAPH_INSTANTIATE_TEST_SUITE_P(
    ${BACKEND_NAME},
    pad_1d_reflect,
    PadBackendTest,
    testing::Values(Params<float>{test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}),
                                  test::NDArray<int64_t, 1>({2}),
                                  test::NDArray<int64_t, 1>({3}),
                                  test::NDArray<float, 1>({3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3}),
                                  op::PadMode::REFLECT},
                    Params<float>{test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}),
                                  test::NDArray<int64_t, 1>({1}),
                                  test::NDArray<int64_t, 1>({0}),
                                  test::NDArray<float, 1>({2, 1, 2, 3, 4, 5, 6}),
                                  op::PadMode::REFLECT},
                    Params<float>{test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}),
                                  test::NDArray<int64_t, 1>({0}),
                                  test::NDArray<int64_t, 1>({2}),
                                  test::NDArray<float, 1>({1, 2, 3, 4, 5, 6, 5, 4}),
                                  op::PadMode::REFLECT}));

NGRAPH_INSTANTIATE_TEST_SUITE_P(${BACKEND_NAME},
                                pad_2d_reflect,
                                PadBackendTest,
                                testing::Values(Params<float>{test::NDArray<float, 2>({
                                                                  {1, 2, 3},
                                                                  {4, 5, 6},
                                                                  {7, 8, 9},
                                                              }),
                                                              test::NDArray<int64_t, 1>({1, 2}),
                                                              test::NDArray<int64_t, 1>({2, 1}),
                                                              test::NDArray<float, 2>({
                                                                  {6, 5, 4, 5, 6, 5},
                                                                  {3, 2, 1, 2, 3, 2},
                                                                  {6, 5, 4, 5, 6, 5},
                                                                  {9, 8, 7, 8, 9, 8},
                                                                  {6, 5, 4, 5, 6, 5},
                                                                  {3, 2, 1, 2, 3, 2},
                                                              }),
                                                              op::PadMode::REFLECT},
                                                Params<float>{test::NDArray<float, 2>({
                                                                  {1, 2, 3},
                                                                  {4, 5, 6},
                                                                  {7, 8, 9},
                                                              }),
                                                              test::NDArray<int64_t, 1>({1, 2}),
                                                              test::NDArray<int64_t, 1>({0, 0}),
                                                              test::NDArray<float, 2>({
                                                                  {6, 5, 4, 5, 6},
                                                                  {3, 2, 1, 2, 3},
                                                                  {6, 5, 4, 5, 6},
                                                                  {9, 8, 7, 8, 9},
                                                              }),
                                                              op::PadMode::REFLECT},
                                                Params<float>{test::NDArray<float, 2>({
                                                                  {1, 2, 3},
                                                                  {4, 5, 6},
                                                                  {7, 8, 9},
                                                              }),
                                                              test::NDArray<int64_t, 1>({0, 0}),
                                                              test::NDArray<int64_t, 1>({2, 1}),
                                                              test::NDArray<float, 2>({
                                                                  {1, 2, 3, 2},
                                                                  {4, 5, 6, 5},
                                                                  {7, 8, 9, 8},
                                                                  {4, 5, 6, 5},
                                                                  {1, 2, 3, 2},
                                                              }),
                                                              op::PadMode::REFLECT}));

NGRAPH_INSTANTIATE_TEST_SUITE_P(
    ${BACKEND_NAME},
    pad_1d_symmetric,
    PadBackendTest,
    testing::Values(Params<float>{test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}),
                                  test::NDArray<int64_t, 1>({2}),
                                  test::NDArray<int64_t, 1>({3}),
                                  test::NDArray<float, 1>({2, 1, 1, 2, 3, 4, 5, 6, 6, 5, 4}),
                                  op::PadMode::SYMMETRIC},
                    Params<float>{test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}),
                                  test::NDArray<int64_t, 1>({1}),
                                  test::NDArray<int64_t, 1>({0}),
                                  test::NDArray<float, 1>({1, 1, 2, 3, 4, 5, 6}),
                                  op::PadMode::SYMMETRIC},
                    Params<float>{test::NDArray<float, 1>({1, 2, 3, 4, 5, 6}),
                                  test::NDArray<int64_t, 1>({0}),
                                  test::NDArray<int64_t, 1>({2}),
                                  test::NDArray<float, 1>({1, 2, 3, 4, 5, 6, 6, 5}),
                                  op::PadMode::SYMMETRIC}));

NGRAPH_INSTANTIATE_TEST_SUITE_P(${BACKEND_NAME},
                                pad_2d_symmetric,
                                PadBackendTest,
                                testing::Values(Params<float>{test::NDArray<float, 2>({
                                                                  {1, 2, 3},
                                                                  {4, 5, 6},
                                                                  {7, 8, 9},
                                                              }),
                                                              test::NDArray<int64_t, 1>({1, 2}),
                                                              test::NDArray<int64_t, 1>({2, 1}),
                                                              test::NDArray<float, 2>({
                                                                  {2, 1, 1, 2, 3, 3},
                                                                  {2, 1, 1, 2, 3, 3},
                                                                  {5, 4, 4, 5, 6, 6},
                                                                  {8, 7, 7, 8, 9, 9},
                                                                  {8, 7, 7, 8, 9, 9},
                                                                  {5, 4, 4, 5, 6, 6},
                                                              }),
                                                              op::PadMode::SYMMETRIC},
                                                Params<float>{test::NDArray<float, 2>({
                                                                  {1, 2, 3},
                                                                  {4, 5, 6},
                                                                  {7, 8, 9},
                                                              }),
                                                              test::NDArray<int64_t, 1>({1, 2}),
                                                              test::NDArray<int64_t, 1>({0, 0}),
                                                              test::NDArray<float, 2>({
                                                                  {2, 1, 1, 2, 3},
                                                                  {2, 1, 1, 2, 3},
                                                                  {5, 4, 4, 5, 6},
                                                                  {8, 7, 7, 8, 9},

                                                              }),
                                                              op::PadMode::SYMMETRIC},
                                                Params<float>{test::NDArray<float, 2>({
                                                                  {1, 2, 3},
                                                                  {4, 5, 6},
                                                                  {7, 8, 9},
                                                              }),
                                                              test::NDArray<int64_t, 1>({0, 0}),
                                                              test::NDArray<int64_t, 1>({2, 1}),
                                                              test::NDArray<float, 2>({
                                                                  {1, 2, 3, 3},
                                                                  {4, 5, 6, 6},
                                                                  {7, 8, 9, 9},
                                                                  {7, 8, 9, 9},
                                                                  {4, 5, 6, 6},
                                                              }),
                                                              op::PadMode::SYMMETRIC}));

NGRAPH_TEST(${BACKEND_NAME}, pad_to_large_symmetric_padding)
{
    const auto params_to_large = Params<float>{test::NDArray<float, 2>({
                                                   {1, 2},
                                                   {4, 5},
                                               }),
                                               test::NDArray<int64_t, 1>({0, 3}),
                                               test::NDArray<int64_t, 1>({0, 0}),
                                               test::NDArray<float, 2>({
                                                   {0, 0, 0, 0, 0},
                                                   {0, 0, 0, 0, 0},
                                               }),
                                               op::PadMode::SYMMETRIC};

    EXPECT_ANY_THROW(PadBackendTest::execute_test(params_to_large));

    const auto params_ok = Params<float>{test::NDArray<float, 2>({
                                             {1, 2},
                                             {4, 5},
                                         }),
                                         test::NDArray<int64_t, 1>({0, 2}),
                                         test::NDArray<int64_t, 1>({0, 0}),
                                         test::NDArray<float, 2>({
                                             {2, 1, 1, 2},
                                             {5, 4, 4, 5},
                                         }),
                                         op::PadMode::SYMMETRIC};

    EXPECT_NO_THROW(PadBackendTest::execute_test(params_ok));
}
NGRAPH_TEST(${BACKEND_NAME}, pad_to_large_reflect_padding)
{
    const auto params_to_large = Params<float>{test::NDArray<float, 2>({
                                                   {1, 2},
                                                   {4, 5},
                                               }),
                                               test::NDArray<int64_t, 1>({0, 2}),
                                               test::NDArray<int64_t, 1>({0, 0}),
                                               test::NDArray<float, 2>({
                                                   {0, 0, 0, 0},
                                                   {0, 0, 0, 0},
                                               }),
                                               op::PadMode::REFLECT};

    EXPECT_ANY_THROW(PadBackendTest::execute_test(params_to_large));

    const auto params_ok = Params<float>{test::NDArray<float, 2>({
                                             {1, 2},
                                             {4, 5},
                                         }),
                                         test::NDArray<int64_t, 1>({0, 1}),
                                         test::NDArray<int64_t, 1>({0, 0}),
                                         test::NDArray<float, 2>({
                                             {2, 1, 2},
                                             {5, 4, 5},
                                         }),
                                         op::PadMode::REFLECT};

    EXPECT_NO_THROW(PadBackendTest::execute_test(params_ok));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_1d)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{1}, {4});
    const auto pads_end = op::Constant::create(element::i64, Shape{1}, {5});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::f32, Shape{15});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(
        test::all_close_f({2112, 2112, 2112, 2112, 1, 2, 3, 4, 5, 6, 2112, 2112, 2112, 2112, 2112},
                          read_vector<float>(result),
                          MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_1d)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{1}, {4});
    const auto pads_end = op::Constant::create(element::i64, Shape{1}, {-2});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::f32, Shape{8});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f({2112, 2112, 2112, 2112, 1, 2, 3, 4},
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_1d_check_limits)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{1}, {4});
    const auto pads_end = op::Constant::create(element::i64, Shape{1}, {-7});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::f32, Shape{3});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        {2112, 2112, 2112}, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{1}, {2});
    const auto pads_end = op::Constant::create(element::i64, Shape{1}, {3});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::EDGE),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::f32, Shape{11});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        {1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6}, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_top_neg)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{1}, {2});
    const auto pads_end = op::Constant::create(element::i64, Shape{1}, {-3});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::EDGE),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::f32, Shape{5});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(
        test::all_close_f({1, 1, 1, 2, 3}, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_top_neg_bigger_than_tensor)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{1}, {2});
    const auto pads_end = op::Constant::create(element::i64, Shape{1}, {-7});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::EDGE),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::f32, Shape{1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f({1}, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_bottom_neg)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{1}, {-2});
    const auto pads_end = op::Constant::create(element::i64, Shape{1}, {3});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::EDGE),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::f32, Shape{7});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        {3, 4, 5, 6, 6, 6, 6}, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_1d_bottom_neg_bigger_than_tensor)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{1}, {-7});
    const auto pads_end = op::Constant::create(element::i64, Shape{1}, {3});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::EDGE),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::f32, Shape{2});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f({6, 6}, read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_2d)
{
    const Shape data_shape{3, 4};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{2}, {2, 3});
    const auto pads_end = op::Constant::create(element::i64, Shape{2}, {1, 2});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::EDGE),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
    auto result = backend->create_tensor(element::f32, Shape{6, 9});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{1, 1, 1, 1, 2, 3, 4, 4, 4},
                                                           {1, 1, 1, 1, 2, 3, 4, 4, 4},
                                                           {1, 1, 1, 1, 2, 3, 4, 4, 4},
                                                           {5, 5, 5, 5, 6, 7, 8, 8, 8},
                                                           {9, 9, 9, 9, 10, 11, 12, 12, 12},
                                                           {9, 9, 9, 9, 10, 11, 12, 12, 12}})
                                      .get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_edge_2d_with_neg)
{
    const Shape data_shape{3, 4};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{2}, {2, -1});
    const auto pads_end = op::Constant::create(element::i64, Shape{2}, {1, 2});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::EDGE),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
    auto result = backend->create_tensor(element::f32, Shape{6, 5});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{2, 3, 4, 4, 4},
                                                           {2, 3, 4, 4, 4},
                                                           {2, 3, 4, 4, 4},
                                                           {6, 7, 8, 8, 8},
                                                           {10, 11, 12, 12, 12},
                                                           {10, 11, 12, 12, 12}})
                                      .get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{1}, {2});
    const auto pads_end = op::Constant::create(element::i64, Shape{1}, {3});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::f32, Shape{11});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(std::vector<float>({3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_top_neg)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{1}, {2});
    const auto pads_end = op::Constant::create(element::i64, Shape{1}, {-3});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::f32, Shape{5});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        std::vector<float>({3, 2, 1, 2, 3}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_top_neg_bigger_than_tensor)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{1}, {2});
    const auto pads_end = op::Constant::create(element::i64, Shape{1}, {-7});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::f32, Shape{1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        std::vector<float>({3}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_bottom_neg)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{1}, {-2});
    const auto pads_end = op::Constant::create(element::i64, Shape{1}, {3});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::f32, Shape{7});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(std::vector<float>({3, 4, 5, 6, 5, 4, 3}),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_1d_bottom_neg_bigger_than_tensor)
{
    const Shape data_shape{6};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{1}, {-7});
    const auto pads_end = op::Constant::create(element::i64, Shape{1}, {3});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3, 4, 5, 6}));
    auto result = backend->create_tensor(element::f32, Shape{2});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        std::vector<float>({4, 3}), read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, DISABLED_pad_reflect_1d_multi_reflect)
{
    const Shape data_shape{3};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{1}, {10});
    const auto pads_end = op::Constant::create(element::i64, Shape{1}, {9});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, std::vector<float>({1, 2, 3}));
    auto result = backend->create_tensor(element::f32, Shape{22});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        std::vector<float>({3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2}),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_2d)
{
    const Shape data_shape{3, 4};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{2}, {2, 3});
    const auto pads_end = op::Constant::create(element::i64, Shape{2}, {1, 2});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a,
              test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());
    auto result = backend->create_tensor(element::f32, Shape{6, 9});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{12, 11, 10, 9, 10, 11, 12, 11, 10},
                                                           {8, 7, 6, 5, 6, 7, 8, 7, 6},
                                                           {4, 3, 2, 1, 2, 3, 4, 3, 2},
                                                           {8, 7, 6, 5, 6, 7, 8, 7, 6},
                                                           {12, 11, 10, 9, 10, 11, 12, 11, 10},
                                                           {8, 7, 6, 5, 6, 7, 8, 7, 6}})
                                      .get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_reflect_2d_with_neg)
{
    const Shape data_shape{3, 4};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{2}, {2, -1});
    const auto pads_end = op::Constant::create(element::i64, Shape{2}, {1, 2});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::REFLECT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a,
              test::NDArray<float, 2>({{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}).get_vector());
    auto result = backend->create_tensor(element::f32, Shape{6, 5});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{10, 11, 12, 11, 10},
                                                           {6, 7, 8, 7, 6},
                                                           {2, 3, 4, 3, 2},
                                                           {6, 7, 8, 7, 6},
                                                           {10, 11, 12, 11, 10},
                                                           {6, 7, 8, 7, 6}})
                                      .get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_2d)
{
    const Shape data_shape{2, 3};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{2}, {1, -1});
    const auto pads_end = op::Constant::create(element::i64, Shape{2}, {2, 0});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {9});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    auto result = backend->create_tensor(element::f32, Shape{5, 2});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(
        test::NDArray<float, 2>({{9, 9}, {2, 3}, {5, 6}, {9, 9}, {9, 9}}).get_vector(),
        read_vector<float>(result),
        MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_2d_all_negative)
{
    const Shape data_shape{3, 3};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{2}, {-1, -1});
    const auto pads_end = op::Constant::create(element::i64, Shape{2}, {-1, -1});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {9});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}).get_vector());
    auto result = backend->create_tensor(element::f32, Shape{1, 1});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{5}}).get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_0x0)
{
    const Shape data_shape{0, 0};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{2}, {2, 3});
    const auto pads_end = op::Constant::create(element::i64, Shape{2}, {3, 2});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    auto result = backend->create_tensor(element::f32, Shape{5, 5});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112}})
                                      .get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_0x3)
{
    const Shape data_shape{0, 3};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{2}, {2, 1});
    const auto pads_end = op::Constant::create(element::i64, Shape{2}, {3, 1});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    auto result = backend->create_tensor(element::f32, Shape{5, 5});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112}})
                                      .get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_2d_3x0)
{
    const Shape data_shape{3, 0};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{2}, {1, 3});
    const auto pads_end = op::Constant::create(element::i64, Shape{2}, {1, 2});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    auto result = backend->create_tensor(element::f32, Shape{5, 5});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f(test::NDArray<float, 2>({{2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112},
                                                           {2112, 2112, 2112, 2112, 2112}})
                                      .get_vector(),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_exterior_4d_1x2x2x2)
{
    const Shape data_shape{1, 2, 2, 2};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1});
    const auto pads_end = op::Constant::create(element::i64, Shape{4}, {0, 0, 1, 1});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {42});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    // clang-format off
    copy_data(a, test::NDArray<float, 4>(
        {
            {
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                },
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                }
            }
        }).get_vector());
    // clang-format on
    auto result = backend->create_tensor(element::f32, Shape{1, 2, 4, 4});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    // clang-format off
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 4>(
        {
            {
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                },
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 0.0f, 0.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                }
            }
        }).get_vector()),
        read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
    // clang-format on
}

NGRAPH_TEST(${BACKEND_NAME}, pad_negative_exterior_4d)
{
    const Shape data_shape{1, 3, 2, 2};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{4}, {0, -1, 1, 1});
    const auto pads_end = op::Constant::create(element::i64, Shape{4}, {0, -1, 1, 1});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {42});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    // clang-format off
    copy_data(a, test::NDArray<float, 4>(
        {
            {
                {
                    {0.0f, 0.0f},
                    {0.0f, 0.0f}
                },
                {
                    {1.0f, 1.0f},
                    {1.0f, 1.0f}
                },
                {
                    {2.0f, 2.0f},
                    {2.0f, 2.0f}
                }
            }
        }).get_vector());
    // clang-format on

    auto result = backend->create_tensor(element::f32, Shape{1, 1, 4, 4});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    // clang-format off
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 4>(
        {
            {
                {
                    {42.0f, 42.0f, 42.0f, 42.0f},
                    {42.0f, 1.0f, 1.0f, 42.0f},
                    {42.0f, 1.0f, 1.0f, 42.0f},
                    {42.0f, 42.0f, 42.0f, 42.0f}
                }
            }
        }).get_vector()),
        read_vector<float>(result), MIN_FLOAT_TOLERANCE_BITS));
    // clang-format on
}

// This test covers the case with multiple image and with asymetric pad
// bug has been found on nvGPU side now covered by this test
NGRAPH_TEST(${BACKEND_NAME}, pad_2channel_2image_asym)
{
    const Shape data_shape{2, 2, 4, 4};
    const auto window_movement_strides = Strides{2, 2};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
    const auto pads_end = op::Constant::create(element::i64, Shape{4}, {0, 0, 2, 2});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {42});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::CONSTANT),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a,
              test::NDArray<float, 4>({{{{0, 1, 0, 2}, // img 0 chan 0
                                         {0, 3, 2, 0},
                                         {2, 0, 0, 0},
                                         {0, 2, 1, 0}},

                                        {{0, 0, 0, 2}, // img 0 chan 1
                                         {0, 2, 3, 0},
                                         {2, 0, 1, 0},
                                         {2, 0, 0, 0}}},

                                       {{{0, 2, 1, 1}, // img 1 chan 0
                                         {0, 0, 2, 0},
                                         {0, 0, 1, 2},
                                         {0, 0, 0, 0}},

                                        {{2, 1, 0, 0}, // img 1 chan 1
                                         {0, 2, 0, 0},
                                         {1, 1, 2, 0},
                                         {1, 0, 0, 0}}}})
                  .get_vector());

    auto result = backend->create_tensor(element::f32, Shape{2, 2, 6, 6});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});

    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 4>({{{{0, 1, 0, 2, 42, 42}, // img 0 chan 0
                                                              {0, 3, 2, 0, 42, 42},
                                                              {2, 0, 0, 0, 42, 42},
                                                              {0, 2, 1, 0, 42, 42},
                                                              {42, 42, 42, 42, 42, 42},
                                                              {42, 42, 42, 42, 42, 42}},

                                                             {{0, 0, 0, 2, 42, 42}, // img 1 chan 0
                                                              {0, 2, 3, 0, 42, 42},
                                                              {2, 0, 1, 0, 42, 42},
                                                              {2, 0, 0, 0, 42, 42},
                                                              {42, 42, 42, 42, 42, 42},
                                                              {42, 42, 42, 42, 42, 42}}},

                                                            {{{0, 2, 1, 1, 42, 42}, // img 1 chan 0
                                                              {0, 0, 2, 0, 42, 42},
                                                              {0, 0, 1, 2, 42, 42},
                                                              {0, 0, 0, 0, 42, 42},
                                                              {42, 42, 42, 42, 42, 42},
                                                              {42, 42, 42, 42, 42, 42}},

                                                             {{2, 1, 0, 0, 42, 42}, // img 1 chan 1
                                                              {0, 2, 0, 0, 42, 42},
                                                              {1, 1, 2, 0, 42, 42},
                                                              {1, 0, 0, 0, 42, 42},
                                                              {42, 42, 42, 42, 42, 42},
                                                              {42, 42, 42, 42, 42, 42}}}})
                                       .get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}

NGRAPH_TEST(${BACKEND_NAME}, pad_symmetric)
{
    const Shape data_shape{2, 3};
    const auto data = make_shared<op::Parameter>(element::f32, data_shape);

    const auto pads_begin = op::Constant::create(element::i64, Shape{2}, {1, 2});
    const auto pads_end = op::Constant::create(element::i64, Shape{2}, {1, 2});
    const auto pad_val = op::Constant::create(element::f32, Shape{}, {2112});

    auto f = make_shared<Function>(
        make_shared<op::v1::Pad>(data, pads_begin, pads_end, pad_val, op::PadMode::SYMMETRIC),
        ParameterVector{data});

    auto backend = runtime::Backend::create("${BACKEND_NAME}");

    // Create some tensors for input/output
    auto a = backend->create_tensor(element::f32, data_shape);
    copy_data(a, test::NDArray<float, 2>({{1, 2, 3}, {4, 5, 6}}).get_vector());
    auto result = backend->create_tensor(element::f32, Shape{4, 7});

    auto handle = backend->compile(f);
    handle->call_with_validate({result}, {a});
    EXPECT_TRUE(test::all_close_f((test::NDArray<float, 2>({{2, 1, 1, 2, 3, 3, 2},
                                                            {2, 1, 1, 2, 3, 3, 2},
                                                            {5, 4, 4, 5, 6, 6, 5},
                                                            {5, 4, 4, 5, 6, 6, 5}})
                                       .get_vector()),
                                  read_vector<float>(result),
                                  MIN_FLOAT_TOLERANCE_BITS));
}
