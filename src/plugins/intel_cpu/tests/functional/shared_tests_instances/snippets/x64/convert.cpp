// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/convert.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>> types_Convert = {
        { { ov::element::f32 }, { ov::element::bf16 } },
        { { ov::element::f32 }, { ov::element::u8 } },
        { { ov::element::f32 }, { ov::element::i8 } },
        { { ov::element::f32 }, { ov::element::f16 } },

        { { ov::element::f16 }, { ov::element::f32 } },
        { { ov::element::f16 }, { ov::element::bf16 } },
        { { ov::element::f16 }, { ov::element::i8 } },
        { { ov::element::f16 }, { ov::element::u8 } },

        { { ov::element::bf16 }, { ov::element::f32 } },
        { { ov::element::bf16 }, { ov::element::i8 } },
        { { ov::element::bf16 }, { ov::element::u8 } },
        { { ov::element::bf16 }, { ov::element::f16 } },

        { { ov::element::i8 }, { ov::element::f32 } },
        { { ov::element::i8 }, { ov::element::bf16 } },
        { { ov::element::i8 }, { ov::element::u8 }  },
        { { ov::element::i8 }, { ov::element::f16 }  },

        { { ov::element::u8 }, { ov::element::f32 } },
        { { ov::element::u8 }, { ov::element::bf16 } },
        { { ov::element::u8 }, { ov::element::i8 } },
        { { ov::element::u8 }, { ov::element::f16 } },
};

const std::vector<std::vector<ov::test::InputShape>> inputShapes_Convert = {
        { {{}, {{2, 16}}} },
        { {{}, {{5, 5}}} },
        { {{}, {{2, 12, 1}}} },
        // DS(dynamic shape)
        { {{-1, -1}, {{2, 16}, {2, 8}, {2, 16}}} },
        { {{{1, 5}, 5}, {{5, 5}, {1, 5}, {5, 5}}} },
        { {{{1, 10}, {4, 12}, {1, 2}}, {{2, 12, 1}, {4, 4, 2}, {2, 12, 1}}} }
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Convert, Convert,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_Convert),
                                 ::testing::ValuesIn(types_Convert),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Convert::getTestCaseName);

const std::vector<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>> types_ConvertInput = {
        { { ov::element::f32 }, { ov::element::bf16 } },

        { { ov::element::bf16 }, { ov::element::f32 } },

        { { ov::element::i8 }, { ov::element::f32 } },
        { { ov::element::i8 }, { ov::element::bf16 } },
        { { ov::element::i8 }, { ov::element::f16 } },

        { { ov::element::u8 }, { ov::element::f32 } },
        { { ov::element::u8 }, { ov::element::bf16 } },
        { { ov::element::u8 }, { ov::element::f16 } },
};

const std::vector<std::vector<ov::test::InputShape>> inputShapes_ConvertInput = {
        { {{}, {{2, 16}}}, {{}, {{1, 16}}} },
        { {{}, {{5, 18}}}, {{}, {{5, 1}}} },
        { {{}, {{3, 1}}}, {{}, {{3, 21}}} },
        // DS
        { {{-1, -1}, {{2, 16}, {1, 16}, {2, 16}}}, {{-1, -1}, {{1, 16}, {2, 16}, {1, 16}}} },
        { {{5, -1}, {{5, 18}, {5, 1}, {5, 18}}}, {{-1, 1}, {{5, 1}, {1, 1}, {5, 1}}} },
        { {{{1, 4}, {1, 8}}, {{3, 1}, {4, 8}, {3, 1}}}, {{{1, 4}, {8, 21}}, {{3, 21}, {1, 8}, {3, 21}}} }
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertInput, ConvertInput,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_ConvertInput),
                                 ::testing::ValuesIn(types_ConvertInput),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Convert::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertOutput, ConvertOutput,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_ConvertInput),
                                 ::testing::ValuesIn(types_ConvertInput),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Convert::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertStub, ConvertStub,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_ConvertInput),
                                 ::testing::ValuesIn(types_ConvertInput),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Convert::getTestCaseName);

const std::vector<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>> types_ConvertPartialInputsAndResults = {
        { { ov::element::i8, ov::element::i8, ov::element::f32 }, { ov::element::f32, ov::element::i8 } },
};

const std::vector<std::vector<ov::test::InputShape>> inputShapes_ConvertPartialInputsAndResults = {
        { {{}, {{2, 16}}}, {{}, {{1, 16}}}, {{}, {{1, 1}}} },
        { {{}, {{5, 18}}}, {{}, {{5, 1}}}, {{}, {{1, 18}}} },
        { {{}, {{3, 1}}}, {{}, {{3, 21}}}, {{}, {{3, 1}}} },
        // DS
        { {{-1, -1}, {{3, 1}, {2, 4}, {3, 1}}}, {{{1, 3}, -1}, {{3, 21}, {2, 1}, {3, 21}}}, {{{1, 3}, {1, 2}}, {{3, 1}, {1, 1}, {3, 1}}} },
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertPartialInputsAndResults, ConvertPartialInputsAndResults,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_ConvertPartialInputsAndResults),
                                 ::testing::ValuesIn(types_ConvertPartialInputsAndResults),
                                 ::testing::Values(2), // subgraph & roll after subgraph
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Convert::getTestCaseName);

const std::vector<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>> types_ConvertMany = {
        { { ov::element::f32, ov::element::u8}, {} },
        { { ov::element::f32, ov::element::u8, ov::element::i8 }, {} },
        { { ov::element::f32, ov::element::f32, ov::element::i8, ov::element::i8 }, {} },
};

const std::vector<std::vector<ov::test::InputShape>> inputShapes_ConvertManyOnInputs = {
        { {{}, {{5, 5, 5, 5}}} },
        // DS
        { {{-1, -1, -1, -1}, {{5, 5, 5, 5}, {3, 3, 3, 3}, {5, 5, 5, 5}}} }
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertManyOnInputs, ConvertManyOnInputs,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_ConvertManyOnInputs),
                                 ::testing::ValuesIn(types_ConvertMany),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Convert::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertManyOnOutputs, ConvertManyOnOutputs,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_ConvertManyOnInputs),
                                 ::testing::ValuesIn(types_ConvertMany),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Convert::getTestCaseName);

const std::vector<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>> types_ConvertManyIO = {
        { { ov::element::f32, ov::element::u8}, {ov::element::i8} },
        { { ov::element::f32, ov::element::u8, ov::element::i8 }, { ov::element::u8, ov::element::i8, ov::element::f32, ov::element::f32 } },
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertManyOnInputOutput, ConvertManyOnInputOutput,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes_ConvertManyOnInputs),
                                 ::testing::ValuesIn(types_ConvertManyIO),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Convert::getTestCaseName);

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov