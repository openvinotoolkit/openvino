// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/op/depth_to_space.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_block_first_K1_BS2)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{1, 8, 2});
    auto depth_to_space =
        std::make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
    auto function = std::make_shared<Function>(NodeVector{depth_to_space}, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(
        {0.f, 2.f, 8.f, 10.f, 16.f, 18.f, 24.f, 26.f, 1.f, 3.f, 9.f, 11.f, 17.f, 19.f, 25.f, 27.f});
    test_case.add_expected_output<float>(
        Shape{1, 4, 4},
        {0.f, 1.f, 2.f, 3.f, 8.f, 9.f, 10.f, 11.f, 16.f, 17.f, 18.f, 19.f, 24.f, 25.f, 26.f, 27.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_block_first_K2_BS2)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{1, 8, 2, 2});
    auto depth_to_space =
        std::make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 2);
    auto function = std::make_shared<Function>(NodeVector{depth_to_space}, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({0.f,  2.f,  8.f,  10.f, 16.f, 18.f, 24.f, 26.f, 1.f,  3.f,  9.f,
                                11.f, 17.f, 19.f, 25.f, 27.f, 4.f,  6.f,  12.f, 14.f, 20.f, 22.f,
                                28.f, 30.f, 5.f,  7.f,  13.f, 15.f, 21.f, 23.f, 29.f, 31.f});
    test_case.add_expected_output<float>(
        Shape{1, 2, 4, 4}, {0.f,  1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f,
                            11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f,
                            22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_block_first_K2_BS4)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{1, 16, 2, 1});
    auto depth_to_space =
        std::make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST, 4);
    auto function = std::make_shared<Function>(NodeVector{depth_to_space}, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({0.f,  16.f, 1.f,  17.f, 2.f,  18.f, 3.f,  19.f, 4.f,  20.f, 5.f,
                                21.f, 6.f,  22.f, 7.f,  23.f, 8.f,  24.f, 9.f,  25.f, 10.f, 26.f,
                                11.f, 27.f, 12.f, 28.f, 13.f, 29.f, 14.f, 30.f, 15.f, 31.f});
    test_case.add_expected_output<float>(
        Shape{1, 1, 8, 4}, {0.f,  1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f,
                            11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f,
                            22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_depth_first_1K_BS2)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{1, 8, 2});
    auto depth_to_space =
        std::make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
    auto function = std::make_shared<Function>(NodeVector{depth_to_space}, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>(
        {0.f, 2.f, 1.f, 3.f, 4.f, 6.f, 5.f, 7.f, 8.f, 10.f, 9.f, 11.f, 12.f, 14.f, 13.f, 15.f});
    test_case.add_expected_output<float>(
        Shape{1, 4, 4},
        {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_depth_first_2K_BS2)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{1, 8, 2, 2});
    auto depth_to_space =
        std::make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 2);
    auto function = std::make_shared<Function>(NodeVector{depth_to_space}, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({0.f,  2.f,  8.f,  10.f, 16.f, 18.f, 24.f, 26.f, 1.f,  3.f,  9.f,
                                11.f, 17.f, 19.f, 25.f, 27.f, 4.f,  6.f,  12.f, 14.f, 20.f, 22.f,
                                28.f, 30.f, 5.f,  7.f,  13.f, 15.f, 21.f, 23.f, 29.f, 31.f});
    test_case.add_expected_output<float>(
        Shape{1, 2, 4, 4}, {0.f,  16.f, 2.f,  18.f, 1.f,  17.f, 3.f,  19.f, 8.f,  24.f, 10.f,
                            26.f, 9.f,  25.f, 11.f, 27.f, 4.f,  20.f, 6.f,  22.f, 5.f,  21.f,
                            7.f,  23.f, 12.f, 28.f, 14.f, 30.f, 13.f, 29.f, 15.f, 31.f});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, depth_to_space_depth_first_2K_BS4)
{
    auto A = std::make_shared<op::Parameter>(element::f32, Shape{1, 16, 2, 1});
    auto depth_to_space =
        std::make_shared<op::DepthToSpace>(A, op::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST, 4);
    auto function = std::make_shared<Function>(NodeVector{depth_to_space}, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(function);
    test_case.add_input<float>({0.f,  16.f, 1.f,  17.f, 2.f,  18.f, 3.f,  19.f, 4.f,  20.f, 5.f,
                                21.f, 6.f,  22.f, 7.f,  23.f, 8.f,  24.f, 9.f,  25.f, 10.f, 26.f,
                                11.f, 27.f, 12.f, 28.f, 13.f, 29.f, 14.f, 30.f, 15.f, 31.f});
    test_case.add_expected_output<float>(
        Shape{1, 1, 8, 4}, {0.f,  1.f,  2.f,  3.f,  4.f,  5.f,  6.f,  7.f,  8.f,  9.f,  10.f,
                            11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f,
                            22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.f});
    test_case.run();
}
