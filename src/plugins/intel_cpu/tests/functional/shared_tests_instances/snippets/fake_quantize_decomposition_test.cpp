
// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "snippets/fake_quantize_decomposition_test.hpp"
#include "ie_system_conf.h"

using namespace LayerTestsDefinitions;
using namespace ngraph;

namespace {

namespace decompositionInSubgraph {
const std::vector<TestValues> testValuesDecompositionScalars = {
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{}, {}, {}, {}},
    },
};
const std::vector<TestValues> testValuesDecompositionPerChannel = {
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
    },
};

std::vector<std::pair<std::shared_ptr<Node>, std::pair<std::string, std::string> >> operations = {
    {std::make_shared<opset1::Abs>(), {"Subgraph", "Abs,fakeQuantize"}},
    {std::make_shared<ngraph::op::v4::Swish>(), {"Subgraph", "Swish,fakeQuantize"}},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_FQDecomposition_Scalars,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValuesDecompositionScalars),
        ::testing::ValuesIn(operations),
        // reorder (nChw[16|8]c) + MaxPool + Subgraph + reorder(nchw)
        ::testing::Values(std::pair<size_t, size_t>{4, 1}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_FQDecomposition_PerChannel,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
        ::testing::Values(testValuesDecompositionPerChannel[0]),
        ::testing::ValuesIn(operations),
        // reorder (nChw[16|8]c) + MaxPool + reorder(nChw[16|8]c) x6 + Subgraph + reorder(nchw)
        ::testing::Values(std::pair<size_t, size_t>{10, 1}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_FQDecomposition_PerChannel_Input,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
        ::testing::Values(testValuesDecompositionPerChannel[1]),
        ::testing::ValuesIn(operations),
        // reorder (nChw[16|8]c) + MaxPool + reorder(nChw[16|8]c) x4 + Subgraph + reorder(nchw)
        ::testing::Values(std::pair<size_t, size_t>{8, 1}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);
}  // namespace decompositionInSubgraph


namespace legacyFuse {
const std::vector<TestValues> testValuesLegacyFuse_binary_post = {
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}}
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{}, {}, {1, 3, 1, 1}, {1, 3, 1, 1}}
    },
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}}
    },
};

const std::vector<TestValues> testValuesLegacyFuse = {
    {
        ov::element::f32,
        ngraph::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{}, {}, {}, {}}
    },
};

std::vector<std::pair<std::shared_ptr<Node>, std::pair<std::string, std::string>>> operations = {
    {std::make_shared<opset1::Convolution>(), {"Convolution", "Convolution,fakeQuantize"}},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValuesLegacyFuse),
        ::testing::ValuesIn(operations),
        // if ISA has avx512, conv node will use brgconv, there will be a extra reorder(nhwc)
        // for brg, reorder (nChw[16|8]c) + MaxPool + reorder(nhwc) + reorder(Acdb16a) + Convolution(nhwc) + reorder(nchw)
        // for no brg, reorder (nChw[16|8]c) + MaxPool + reorder(ABcd8b8a) + Convolution(nchw8c) + reorder(nchw)
        ::testing::Values(InferenceEngine::with_cpu_x86_avx512_core() ? std::pair<size_t, size_t>{6, 0} : std::pair<size_t, size_t>{5, 0}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_binary_post,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValuesLegacyFuse_binary_post),
        ::testing::ValuesIn(operations),
        // if ISA has avx512_amx, conv node will use brgconv, there will be a extra reorder(nhwc).
        // if it's avx512 + binary_post ops, conv node will not use brgconv.
        // for brg, reorder (nChw[16|8]c) + MaxPool + reorder(nhwc) + reorder(Acdb16a) + Convolution(nhwc) + reorder(nchw)
        // for no brg, reorder (nChw[16|8]c) + MaxPool + reorder(ABcd8b8a) + Convolution(nChw8c) + reorder(nchw)
        ::testing::Values(InferenceEngine::with_cpu_x86_avx512_core_amx() ? std::pair<size_t, size_t>{6, 0} : std::pair<size_t, size_t>{5, 0}),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);

}  // namespace legacyFuse

}  // namespace
