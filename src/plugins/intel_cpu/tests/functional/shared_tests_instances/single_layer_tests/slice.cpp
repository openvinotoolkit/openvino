// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/slice.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ov::test;

namespace {

const std::vector<ElementType> inputPrecisions = {
    ElementType::f32,
    ElementType::bf16,
    ElementType::i8
};

const std::vector<ElementType> inputPrecisionsOther = {
    ElementType::i64,
    ElementType::i32,
    ElementType::i16,
    ElementType::u8
};

std::vector<Slice8SpecificParams> staticParams = {
        Slice8SpecificParams{ {{{}, {{ 16 }}}}, { 4 }, { 12 }, { 1 }, { 0 } },
        Slice8SpecificParams{ {{{}, {{ 16 }}}}, { 0 }, { 8 }, { 2 }, { 0 } },
        Slice8SpecificParams{ {{{}, {{ 20, 10, 5 }}}}, { 0, 0}, { 10, 20}, { 1, 1 }, { 1, 0 } },
        Slice8SpecificParams{ {{{}, {{ 1, 2, 12, 100 }}}}, { 0, 1, 0, 1 }, { 1, 2, 5, 100 }, { 1, 1, 1, 10 }, {} },
        Slice8SpecificParams{ {{{}, {{ 1, 12, 100 }}}}, { 0, 9, 0 }, { 1, 11, 1 }, { 1, 1, 1 }, { 0, 1, -1 } },
        Slice8SpecificParams{ {{{}, {{ 1, 12, 100 }}}}, { 0, 1, 0 }, { 10, -1, 10 }, { 1, 1, 1 }, { -3, -2, -1} },
        Slice8SpecificParams{ {{{}, {{ 2, 12, 100 }}}}, { 1, 12, 100 }, { 0, 7, 0 }, { -1, -1, -1 }, {} },
        Slice8SpecificParams{ {{{}, {{ 2, 12, 100 }}}}, { 1, 4, 99 }, { 0, 9, 0 }, { -1, 2, -1 }, {} },
        Slice8SpecificParams{ {{{}, {{ 2, 12, 100 }}}}, { -1, -1, -1 }, { 0, 4, 0 }, { -1, -2, -1 }, {} },
        Slice8SpecificParams{ {{{}, {{ 2, 12, 100 }}}}, { -1, -1, -1 }, { 0, 0, 4 }, { -1, -1, -1 }, {2, 0, 1} },
        Slice8SpecificParams{ {{{}, {{ 2, 12, 100 }}}}, { 0, 0, 4 }, { -5, -1, -1 }, { 1, 2, 1 }, {2, 0, 1} },
        Slice8SpecificParams{ {{{}, {{ 2, 2, 2, 2 }}}}, { 0, 0, 0, 0 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 }, {} },
        Slice8SpecificParams{ {{{}, {{ 2, 2, 2, 2 }}}}, { 1, 1, 1, 1 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 }, {} },
        Slice8SpecificParams{ {{{}, {{ 2, 2, 4, 3 }}}}, { 0, 0, 0, 0 }, { 2, 2, 4, 3 }, { 1, 1, 2, 1 }, { -4, 1, -2, 3 } },
        Slice8SpecificParams{ {{{}, {{ 2, 2, 4, 2 }}}}, { 1, 0, 0, 1 }, { 2, 2, 4, 2 }, { 1, 1, 2, 1 }, {} },
        Slice8SpecificParams{ {{{}, {{ 1, 2, 4, 2 }}}}, { 0, 1, 0, 1 }, { 10, 2, 4, 2 }, { 1, 1, 2, 1 }, {} },
        Slice8SpecificParams{ {{{}, {{ 1, 2, 4, 2 }}}}, { 1, 0, 1, 0 }, { 2, 4, 2, 10 }, { 1, 2, 1, 1 }, { -1, -2, -3, -4 } },
        Slice8SpecificParams{ {{{}, {{ 10, 2, 4, 2 }}}}, { 9, 1, 3, 0 }, { 0, 0, 0, 1 }, { -1, -1, -1, 1 }, {} },
        Slice8SpecificParams{ {{{}, {{ 10, 2, 4, 2 }}}}, { 19, 1, -1, 0 }, { -10, 0, 0, -1 }, { -1, -1, -1, 1 }, {} },
        Slice8SpecificParams{ {{{}, {{ 3, 2, 4, 200 }}}}, { 0, 1, -1, -1 }, { 3, 2, 0, 0 }, { 1, 1, -2, -1 }, {} },
        Slice8SpecificParams{ {{{}, {{ 2, 4, 5, 5, 68 }}}}, { 0, 1, 0, 0, 0 }, {
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max() }, { 1, 1, 1, 1, 16 }, {} },
        Slice8SpecificParams{ {{{}, {{ 10, 12 }}}}, { -1, 1 }, { -9999, 10 }, { -1, 1 }, {} },
        Slice8SpecificParams{ {{{}, {{ 5, 5, 5, 5 }}}}, { -1, 0, -1, 0 }, { -50, -1, -60, -1 }, { -1, 1, -1, 1 }, {} },
        Slice8SpecificParams{ {{{}, {{ 1, 5, 32, 32 }}}}, { 0, 2, 5, 4 }, { 1, 4, 28, 27 }, { 1, 1, 1, 1 }, { 0, 1, 2, 3 } },
        Slice8SpecificParams{ {{{}, {{ 1, 5, 32, 20 }}}}, { 0, 1, 0, 0 }, { 1, 3, 32, 20 }, { 1, 1, 1, 1 }, { 0, 1, 2, 3 } },
        Slice8SpecificParams{ {{{}, {{ 2, 5, 32, 20 }}}}, { 0, 0, 10, 0 }, { 1, 3, 20, 20 }, { 1, 1, 1, 1 }, { 0, 1, 2, 3 } },
        Slice8SpecificParams{ {{{}, {{ 1, 5, 32, 32 }}}}, { 0, 0, 20, 20 }, { 1, 5, 25, 26 }, { 1, 1, 1, 2 }, { 0, 1, 2, 3 } },
        Slice8SpecificParams{ {{{}, {{ 2, 5, 32, 32 }}}}, { 0, 0, 0, 20 }, { 1, 2, 30, 30 }, { 1, 1, 2, 1 }, { 0, 1, 2, 3 } },
        Slice8SpecificParams{ {{{}, {{ 1, 5, 32, 20 }}}}, { 0, 0, 2, 10 }, { 1, 3, 32, 20 }, { 1, 1, 1, 1 }, { 0, 1, 2, 3 } },
        Slice8SpecificParams{ {{{}, {{ 2, 5, 32, 32 }}}}, { 0, 1, 0, 10 }, { 1, 5, 32, 30 }, { 1, 1, 1, 1 }, { 0, 1, 2, 3 } },
        Slice8SpecificParams{ {{{}, {{ 1, 5, 32, 20 }}}}, { 0, 1, 2, 10 }, { 1, 5, 32, 18 }, { 1, 1, 1, 2 }, { 0, 1, 2, 3 } },
        Slice8SpecificParams{ {{{}, {{ 2, 8, 32, 20 }}}}, { 0, 0, 2, 10 }, { 1, 8, 32, 18 }, { 1, 2, 1, 2 }, { 0, 1, 2, 3 } },
        Slice8SpecificParams{ {{{}, {{ 2, 8, 32, 20 }}}}, { 0, -20, -15 }, { 2, -5, 3 }, { 1, 1, 1 }, { 0, 2, 1 } }
};

INSTANTIATE_TEST_SUITE_P(smoke_Static, Slice8LayerTest,
        ::testing::Combine(
            ::testing::ValuesIn(staticParams),
            ::testing::ValuesIn(inputPrecisions),
            ::testing::Values(ElementType::undefined),
            ::testing::Values(ElementType::undefined),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(std::map<std::string, std::string>())),
        Slice8LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PrecisionTransformation, Slice8LayerTest,
        ::testing::Combine(
            ::testing::Values(staticParams[0]),
            ::testing::ValuesIn(inputPrecisionsOther),
            ::testing::Values(ElementType::undefined),
            ::testing::Values(ElementType::undefined),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(std::map<std::string, std::string>())),
        Slice8LayerTest::getTestCaseName);


std::vector<Slice8SpecificParams> dynamicParams = {
        Slice8SpecificParams{ {{{ -1 }, {{ 8 }, { 16 }}}}, { 4 }, { 12 }, { 1 }, { 0 } },
        Slice8SpecificParams{ {{{ ov::Dimension(2, 20) }, {{ 5 }, { 15 }}}}, { 0 }, { 8 }, { 2 }, { 0 } },
        Slice8SpecificParams{ {{{ -1, -1, -1 }, {{ 20, 10, 5 }, {5, 10, 20}}}}, { 0, 0}, { 10, 20}, { 1, 1 }, { 1, 0 } },
        Slice8SpecificParams{ {{{ -1, -1, -1, -1 }, {{ 1, 2, 12, 100 }}}}, { 0, 1, 0, 1 }, { 1, 2, 5, 100 }, { 1, 1, 1, 10 }, {} },
        Slice8SpecificParams{ {{{ -1, ov::Dimension(2, 20), -1 }, {{ 1, 12, 100 }, { 2, 12, 100 }}}}, { 0, 9, 0 }, { 1, 11, 1 }, { 1, 1, 1 }, {} },
        Slice8SpecificParams{ {{{ ov::Dimension(1, 5), ov::Dimension(1, 5), ov::Dimension(1, 5), ov::Dimension(1, 5) },
            {{ 2, 2, 2, 2 }, { 2, 2, 4, 3 }, { 2, 2, 4, 2 }, { 1, 2, 4, 2 }}}},
            { 0, 0, 0, 0 }, { 2, 2, 2, 2 }, { 1, 1, 1, 1 }, {} },
        Slice8SpecificParams{ {{{ -1, ov::Dimension(1, 5), ov::Dimension(1, 5), -1 }, {{ 10, 2, 4, 2 }, { 10, 4, 2, 2 }}}},
            { 9, 1, 3, 0 }, { 0, 0, 0, 1 }, { -1, -1, -1, 1 }, {} },
        Slice8SpecificParams{ {{{ -1, ov::Dimension(1, 5), -1, -1, ov::Dimension(30, 70) }, {{ 2, 4, 5, 5, 68 }, { 2, 3, 7, 7, 33 }}}},
            { 0, 1, 0, 0, 0 }, {
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max() }, { 1, 1, 1, 1, 16 }, {} },
        Slice8SpecificParams{ {{{ov::Dimension(1, 5), ov::Dimension(1, 7), ov::Dimension(1, 35), ov::Dimension(1, 35)},
            {{ 1, 5, 32, 32 }, { 2, 5, 32, 20 }, { 2, 5, 32, 32 }}}}, { 0, 2, 5, 4 }, { 1, 4, 28, 27 }, { 1, 1, 1, 1 }, { 0, 1, 2, 3 } }
};

INSTANTIATE_TEST_SUITE_P(smoke_Dynamic, Slice8LayerTest,
        ::testing::Combine(
            ::testing::ValuesIn(dynamicParams),
            ::testing::ValuesIn(inputPrecisions),
            ::testing::Values(ElementType::undefined),
            ::testing::Values(ElementType::undefined),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(CommonTestUtils::DEVICE_CPU),
            ::testing::Values(std::map<std::string, std::string>())),
        Slice8LayerTest::getTestCaseName);
}  // namespace
