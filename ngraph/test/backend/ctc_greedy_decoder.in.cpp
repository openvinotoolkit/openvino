//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, ctc_greedy_decoder)
{
    const int T = 3;
    const int N = 1;
    const int C = 3;
    const auto data_shape = Shape{T, N, C};
    const auto masks_shape = Shape{T, N};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto masks = make_shared<op::Parameter>(element::f32, masks_shape);
    auto decoder = make_shared<op::CTCGreedyDecoder>(data, masks, false);
    auto function = make_shared<Function>(decoder, ParameterVector{data, masks});
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({0.1f, 0.2f, 0.f, 0.4f, 0.3f, 0.f, 0.5f, 0.6f, 0.f});
    test_case.add_input<float>({1.0f, 1.0f, 1.0f});
    test_case.add_expected_output(Shape{N, T, 1, 1}, vector<float>{1.0f, 0.0f, 1.0f});

    test_case.run_with_tolerance_as_fp(1.0e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, ctc_greedy_decoder_f16)
{
    const int T = 3;
    const int N = 1;
    const int C = 3;
    const auto data_shape = Shape{T, N, C};
    const auto masks_shape = Shape{T, N};

    auto data = make_shared<op::Parameter>(element::f16, data_shape);
    auto masks = make_shared<op::Parameter>(element::f16, masks_shape);
    auto decoder = make_shared<op::CTCGreedyDecoder>(data, masks, false);
    auto function = make_shared<Function>(decoder, ParameterVector{data, masks});
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float16>({0.1f, 0.2f, 0.f, 0.4f, 0.3f, 0.f, 0.5f, 0.6f, 0.f});
    test_case.add_input<float16>({1.0f, 1.0f, 1.0f});
    test_case.add_expected_output(Shape{N, T, 1, 1}, vector<float16>{1.0f, 0.0f, 1.0f});

    test_case.run_with_tolerance_as_fp(1.0e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, ctc_greedy_decoder_multiple_batches)
{
    const int T = 3;
    const int N = 2;
    const int C = 3;
    const auto data_shape = Shape{T, N, C};
    const auto masks_shape = Shape{T, N};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto masks = make_shared<op::Parameter>(element::f32, masks_shape);
    auto decoder = make_shared<op::CTCGreedyDecoder>(data, masks, false);
    auto function = make_shared<Function>(decoder, ParameterVector{data, masks});
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({0.1f,
                                0.2f,
                                0.f,
                                0.15f,
                                0.25f,
                                0.f,
                                0.4f,
                                0.3f,
                                0.f,
                                0.45f,
                                0.35f,
                                0.f,
                                0.5f,
                                0.6f,
                                0.f,
                                0.55f,
                                0.65f,
                                0.f});

    test_case.add_input<float>({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

    test_case.add_expected_output(Shape{N, T, 1, 1},
                                  vector<float>{1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});

    test_case.run_with_tolerance_as_fp(1.0e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, ctc_greedy_decoder_single_batch_short_sequence)
{
    const int T = 3;
    const int N = 1;
    const int C = 3;
    const auto data_shape = Shape{T, N, C};
    const auto masks_shape = Shape{T, N};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto masks = make_shared<op::Parameter>(element::f32, masks_shape);
    auto decoder = make_shared<op::CTCGreedyDecoder>(data, masks, false);
    auto function = make_shared<Function>(decoder, ParameterVector{data, masks});
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({0.1f, 0.2f, 0.f, 0.4f, 0.3f, 0.f, 0.5f, 0.6f, 0.f});
    test_case.add_input<float>({1.0f, 1.0f, 0.0f});
    test_case.add_expected_output(Shape{N, T, 1, 1}, vector<float>{1.0f, 0.0f, -1.0f});

    test_case.run_with_tolerance_as_fp(1.0e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, ctc_greedy_decoder_merge)
{
    const int T = 3;
    const int N = 1;
    const int C = 3;
    const auto data_shape = Shape{T, N, C};
    const auto masks_shape = Shape{T, N};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto masks = make_shared<op::Parameter>(element::f32, masks_shape);
    auto decoder = make_shared<op::CTCGreedyDecoder>(data, masks, true);
    auto function = make_shared<Function>(decoder, ParameterVector{data, masks});
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({0.1f, 0.2f, 0.f, 0.3f, 0.4f, 0.f, 0.6f, 0.5f, 0.f});
    test_case.add_input<float>({1.0f, 1.0f, 1.0f});
    test_case.add_expected_output(Shape{N, T, 1, 1}, vector<float>{1.0f, 0.0f, -1.0f});

    test_case.run_with_tolerance_as_fp(1.0e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, ctc_greedy_decoder_single_no_merge)
{
    const int T = 3;
    const int N = 1;
    const int C = 3;
    const auto data_shape = Shape{T, N, C};
    const auto masks_shape = Shape{T, N};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto masks = make_shared<op::Parameter>(element::f32, masks_shape);
    auto decoder = make_shared<op::CTCGreedyDecoder>(data, masks, false);
    auto function = make_shared<Function>(decoder, ParameterVector{data, masks});
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>({0.1f, 0.2f, 0.f, 0.3f, 0.4f, 0.f, 0.6f, 0.5f, 0.f});
    test_case.add_input<float>({1.0f, 1.0f, 1.0f});
    test_case.add_expected_output(Shape{N, T, 1, 1}, vector<float>{1.0f, 1.0f, 0.0f});

    test_case.run_with_tolerance_as_fp(1.0e-4f);
}

NGRAPH_TEST(${BACKEND_NAME}, ctc_greedy_decoder_multiple_sequences)
{
    const int T = 2;
    const int N = 2;
    const int C = 3;
    const auto data_shape = Shape{T, N, C};
    const auto masks_shape = Shape{T, N};

    auto data = make_shared<op::Parameter>(element::f32, data_shape);
    auto masks = make_shared<op::Parameter>(element::f32, masks_shape);
    auto decoder = make_shared<op::CTCGreedyDecoder>(data, masks, false);
    auto function = make_shared<Function>(decoder, ParameterVector{data, masks});
    auto test_case = test::TestCase<TestEngine>(function);

    test_case.add_input<float>(
        {0.1f, 0.2f, 0.f, 0.4f, 0.3f, 0.f, 0.5f, 0.6f, 0.f, 0.7f, 0.8f, 0.f});
    test_case.add_input<float>({1.0f, 1.0f, 1.0f, 0.0f});
    test_case.add_expected_output(Shape{N, T, 1, 1}, vector<float>{1.0f, 1.0f, 0.0f, -1.0f});

    test_case.run_with_tolerance_as_fp(1.0e-4f);
}
