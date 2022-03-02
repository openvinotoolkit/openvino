
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/fake_quantize_decomposition_test.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<TestValues> testValues = {
    {
        {
            ov::element::f32,
            ngraph::Shape{1, 3, 1024 * 4, 1024 * 4},
            ov::element::f32,
            1.f,
            {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
            CommonTestUtils::DEVICE_CPU
        },
        {
            12,
            {
                {"fakeQuantize", "FakeQuantize"},
            },
            {},
        }
    },
    {
        {
            ov::element::f32,
            ngraph::Shape{1, 3, 1024 * 4, 1024 * 4},
            ov::element::f32,
            1.f,
            {{}, {}, {1, 3, 1, 1}, {1, 3, 1, 1}},
            CommonTestUtils::DEVICE_CPU
        },
        {
            12,
            {
                {"fakeQuantize", "FakeQuantize"},
            },
            {},
        }
    },
    {
        {
            ov::element::f32,
            ngraph::Shape{1, 3, 1024 * 4, 1024 * 4},
            ov::element::f32,
            1.f,
            {{}, {}, {}, {}},
            CommonTestUtils::DEVICE_CPU
        },
        {
            -1,
            {
                {"convert1", "Convert"},
                {"convert2", "Convert"},
                {"parameter", "Input"},
                {"relu1", "Eltwise"},
                {"relu2,fakeQuantize,relu3", "Subgraph"},
                {"result", "Output"}
            },
            {"FakeQuantize"},
        }
    },
    {
        {
            ov::element::f32,
            ngraph::Shape{1, 3, 1024 * 4, 1024 * 4},
            ov::element::f32,
            1.f,
            {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
            CommonTestUtils::DEVICE_CPU
        },
        {
            12,
            {
                {"fakeQuantize", "FakeQuantize"},
            },
            {},
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets,
    FakeQuantizeDecompositionTest,
    ::testing::ValuesIn(testValues),
    FakeQuantizeDecompositionTest::getTestCaseName);

}  // namespace
