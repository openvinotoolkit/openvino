
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "snippets/fake_quantize_decomposition_test.hpp"

using namespace ov::test::snippets;

namespace {

namespace decompositionInSubgraph {
const std::vector<TestValues> testValuesDecompositionScalars = {
    {
        ov::element::f32,
        ov::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{}, {}, {}, {}},
    },
    {
        ov::element::f16,
        ov::Shape{1, 3, 16, 16},
        ov::element::f16,
        1.f,
        {{}, {}, {}, {}},
    },
};
const std::vector<TestValues> testValuesDecompositionPerChannel = {
    {
        ov::element::f32,
        ov::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
    },
    {
        ov::element::f16,
        ov::Shape{1, 3, 16, 16},
        ov::element::f16,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
    },
};

const std::vector<TestValues> testValuesDecompositionPerChannelInput = {
    {
        ov::element::f32,
        ov::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
    },
        {
        ov::element::f16,
        ov::Shape{1, 3, 16, 16},
        ov::element::f16,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
    },
};

std::vector<std::pair<std::shared_ptr<ov::Node>, std::pair<std::string, std::string> >> operations = {
    {std::make_shared<ov::op::v0::Abs>(), {"Subgraph", "Abs,fakeQuantize"}},
    {std::make_shared<ov::op::v4::Swish>(), {"Subgraph", "Swish,fakeQuantize"}},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_FQDecomposition_Scalars,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValuesDecompositionScalars),
        ::testing::ValuesIn(operations),
        ::testing::Values(std::pair<size_t, size_t>{1, 1}),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_FQDecomposition_PerChannel,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValuesDecompositionPerChannel),
        ::testing::ValuesIn(operations),
        ::testing::Values(std::pair<size_t, size_t>{1, 1}),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_FQDecomposition_PerChannel_Input,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValuesDecompositionPerChannelInput),
        ::testing::ValuesIn(operations),
        ::testing::Values(std::pair<size_t, size_t>{1, 1}),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);
}  // namespace decompositionInSubgraph

#ifdef OPENVINO_ARCH_X86_64
namespace legacyFuse {
const std::vector<TestValues> testValuesLegacyFuse = {
    {
        ov::element::f32,
        ov::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}}
    },
    {
        ov::element::f32,
        ov::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{}, {}, {1, 3, 1, 1}, {1, 3, 1, 1}}
    },
    {
        ov::element::f32,
        ov::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{}, {}, {}, {}}
    },
    {
        ov::element::f32,
        ov::Shape{1, 3, 16, 16},
        ov::element::f32,
        1.f,
        {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}}
    },
};

std::vector<std::pair<std::shared_ptr<ov::Node>, std::pair<std::string, std::string>>> operations = {
    {std::make_shared<ov::op::v1::Convolution>(), {"Convolution", "Convolution,fakeQuantize"}},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets,
    FakeQuantizeDecompositionTest,
    ::testing::Combine(
        ::testing::ValuesIn(testValuesLegacyFuse),
        ::testing::ValuesIn(operations),
        // reorder (nChw[16|8]c) + Convolution(with internal weight reordering) + reorder(nchw)
        ::testing::Values(std::pair<size_t, size_t>{3, 0}),
        ::testing::Values(ov::test::utils::DEVICE_CPU)),
    FakeQuantizeDecompositionTest::getTestCaseName);
}  // namespace legacyFuse
#endif

}  // namespace
