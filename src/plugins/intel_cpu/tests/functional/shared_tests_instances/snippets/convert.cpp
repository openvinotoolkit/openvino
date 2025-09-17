// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/convert.hpp"
#include "common_test_utils/test_constants.hpp"
#include "utils.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {

static std::vector<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>> getTypesConvert() {
    std::vector<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>> types = {
        { { ov::element::f32 }, { ov::element::f16 } },
        { { ov::element::f32 }, { ov::element::i8 } },
        { { ov::element::f32 }, { ov::element::u8 } },

        { { ov::element::f16 }, { ov::element::f32 } },
        { { ov::element::f16 }, { ov::element::i8 } },
        { { ov::element::f16 }, { ov::element::u8 } },

        { { ov::element::i8 }, { ov::element::f32 } },
        { { ov::element::i8 }, { ov::element::f16 } },
        { { ov::element::i8 }, { ov::element::u8 } },

        { { ov::element::u8 }, { ov::element::f32 } },
        { { ov::element::u8 }, { ov::element::f16 } },
        { { ov::element::u8 }, { ov::element::i8 } },
    };

    if (is_bf16_supported_by_brgemm()) {
        types.push_back({{ ov::element::f32 }, { ov::element::bf16 }});
        types.push_back({{ ov::element::f16 }, { ov::element::bf16 }});
        types.push_back({{ ov::element::bf16 }, { ov::element::f32 }});
        types.push_back({{ ov::element::bf16 }, { ov::element::i8 }});
        types.push_back({{ ov::element::bf16 }, { ov::element::u8 }});
        types.push_back({{ ov::element::bf16 }, { ov::element::f16 }});
        types.push_back({{ ov::element::i8 }, { ov::element::bf16 }});
        types.push_back({{ ov::element::u8 }, { ov::element::bf16 }});
    }

    return types;
}

static std::vector<std::vector<ov::test::InputShape>> getInputShapesConvert() {
    std::vector<std::vector<ov::test::InputShape>> shapes = {
        { {{}, {{2, 16}}} },
        { {{}, {{5, 7}}} },
        { {{}, {{2, 12, 1}}} },
        { {{{1, 6}, 6}, {{6, 6}, {1, 6}, {6, 6}}} },
    };

    // Add dynamic shapes if platform supports them better (x86_64)
    if (is_bf16_supported_by_brgemm()) {
        shapes[1] = { {{}, {{5, 5}}} };
        shapes.push_back({ {{-1, -1}, {{2, 16}, {2, 8}, {2, 16}}} });
        shapes.push_back({ {{{1, 5}, 5}, {{5, 5}, {1, 5}, {5, 5}}} });
        shapes.push_back({ {{{1, 10}, {4, 12}, {1, 2}}, {{2, 12, 1}, {4, 4, 2}, {2, 12, 1}}} });
    }

    return shapes;
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Convert, Convert,
                         ::testing::Combine(
                                 ::testing::ValuesIn(getInputShapesConvert()),
                                 ::testing::ValuesIn(getTypesConvert()),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Convert::getTestCaseName);

static std::vector<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>> getTypesConvertInput() {
    std::vector<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>> types = {
        { { ov::element::f32 }, { ov::element::f16 } },

        { { ov::element::f16 }, { ov::element::f32 } },

        { { ov::element::i8 }, { ov::element::f32 } },
        { { ov::element::i8 }, { ov::element::f16 } },

        { { ov::element::u8 }, { ov::element::f32 } },
        { { ov::element::u8 }, { ov::element::f16 } },
    };

    if (is_bf16_supported_by_brgemm()) {
        types.push_back({{ ov::element::f32 }, { ov::element::bf16 }});
        types.push_back({{ ov::element::bf16 }, { ov::element::f32 }});
        types.push_back({{ ov::element::i8 }, { ov::element::bf16 }});
        types.push_back({{ ov::element::u8 }, { ov::element::bf16 }});
    }

    return types;
}

static std::vector<std::vector<ov::test::InputShape>> getInputShapesConvertInput() {
    std::vector<std::vector<ov::test::InputShape>> shapes = {
        { {{}, {{2, 16}}}, {{}, {{1, 16}}} },
        { {{}, {{5, 18}}}, {{}, {{5, 1}}} },
        { {{}, {{3, 1}}}, {{}, {{3, 21}}} },
        { {{{1, 6}, 6}, {{6, 6}, {1, 6}, {6, 6}}}, {{{1, 6}, 6}, {{1, 6}, {6, 6}, {1, 6}}} },
    };

    // Add dynamic shapes if platform supports them better (x86_64)
    if (is_bf16_supported_by_brgemm()) {
        shapes.push_back({ {{-1, -1}, {{2, 16}, {1, 16}, {2, 16}}}, {{-1, -1}, {{1, 16}, {2, 16}, {1, 16}}} });
        shapes.push_back({ {{5, -1}, {{5, 18}, {5, 1}, {5, 18}}}, {{-1, 1}, {{5, 1}, {1, 1}, {5, 1}}} });
        shapes.push_back({ {{{1, 4}, {1, 8}}, {{3, 1}, {4, 8}, {3, 1}}}, {{{1, 4}, {8, 21}}, {{3, 21}, {1, 8}, {3, 21}}} });
    }

    return shapes;
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertInput, ConvertInput,
                         ::testing::Combine(
                                 ::testing::ValuesIn(getInputShapesConvertInput()),
                                 ::testing::ValuesIn(getTypesConvertInput()),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Convert::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertOutput, ConvertOutput,
                         ::testing::Combine(
                                 ::testing::ValuesIn(getInputShapesConvertInput()),
                                 ::testing::ValuesIn(getTypesConvertInput()),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Convert::getTestCaseName);

static std::vector<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>> getTypesConvertStub() {
    std::vector<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>> types = {
        { { ov::element::i8 }, { ov::element::f32 } },
        { { ov::element::i8 }, { ov::element::f16 } },

        { { ov::element::u8 }, { ov::element::f32 } },
        { { ov::element::u8 }, { ov::element::f16 } },
    };

    if (is_bf16_supported_by_brgemm()) {
        types.push_back({{ ov::element::i8 }, { ov::element::bf16 }});
        types.push_back({{ ov::element::u8 }, { ov::element::bf16 }});
    }

    return types;
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertStub, ConvertStub,
                         ::testing::Combine(
                                 ::testing::ValuesIn(getInputShapesConvertInput()),
                                 ::testing::ValuesIn(getTypesConvertStub()),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Convert::getTestCaseName);

const std::vector<std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>> types_ConvertPartialInputsAndResults = {
        { { ov::element::i8, ov::element::i8, ov::element::f32 }, { ov::element::f32, ov::element::i8 } },
};

static std::vector<std::vector<ov::test::InputShape>> getInputShapesConvertPartialInputsAndResults() {
    std::vector<std::vector<ov::test::InputShape>> shapes = {
        { {{}, {{2, 16}}}, {{}, {{1, 16}}}, {{}, {{1, 1}}} },
        { {{}, {{5, 18}}}, {{}, {{5, 1}}}, {{}, {{1, 18}}} },
        { {{}, {{3, 1}}}, {{}, {{3, 21}}}, {{}, {{3, 1}}} },
        { {{{1, 3}, 4}, {{1, 4}, {2, 4}, {3, 4}}}, {{{1, 3}, 4}, {{3, 4}, {2, 4}, {3, 4}}}, {{{1, 3}, 4}, {{1, 4}, {1, 4}, {3, 4}}} },
    };

    // Add dynamic shapes if platform supports them better (x86_64)
    if (is_bf16_supported_by_brgemm()) {
        shapes[3] = { {{-1, -1}, {{3, 1}, {2, 4}, {3, 1}}}, {{{1, 3}, -1}, {{3, 21}, {2, 1}, {3, 21}}}, {{{1, 3}, {1, 2}}, {{3, 1}, {1, 1}, {3, 1}}} };
    }

    return shapes;
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertPartialInputsAndResults, ConvertPartialInputsAndResults,
                         ::testing::Combine(
                                 ::testing::ValuesIn(getInputShapesConvertPartialInputsAndResults()),
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

static std::vector<std::vector<ov::test::InputShape>> getInputShapesConvertManyOnInputs() {
    std::vector<std::vector<ov::test::InputShape>> shapes = {
        { {{}, {{5, 5, 5, 5}}} },
        { {{{3, 5}, {3, 5}, {3, 5}, 5}, {{5, 5, 5, 5}, {3, 3, 3, 5}, {5, 5, 5, 5}}} }
    };

    // Add dynamic shapes if platform supports them better (x86_64)
    if (is_bf16_supported_by_brgemm()) {
        shapes[1] = { {{-1, -1, -1, -1}, {{5, 5, 5, 5}, {3, 3, 3, 3}, {5, 5, 5, 5}}} };
    }

    return shapes;
}

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertManyOnInputs, ConvertManyOnInputs,
                         ::testing::Combine(
                                 ::testing::ValuesIn(getInputShapesConvertManyOnInputs()),
                                 ::testing::ValuesIn(types_ConvertMany),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Convert::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_ConvertManyOnOutputs, ConvertManyOnOutputs,
                         ::testing::Combine(
                                 ::testing::ValuesIn(getInputShapesConvertManyOnInputs()),
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
                                 ::testing::ValuesIn(getInputShapesConvertManyOnInputs()),
                                 ::testing::ValuesIn(types_ConvertManyIO),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         Convert::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov
