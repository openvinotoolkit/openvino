// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef HAVE_AVX2
#include "unpack.hpp"

namespace {

const auto TestCases = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::i4}),
        ::testing::ValuesIn({ov::element::Type_t::i8, ov::element::Type_t::f16}),
        ::testing::ValuesIn({ov::element::Type_t::dynamic}), // no used in this test
        ::testing::ValuesIn({ov::element::Type_t::dynamic}), // no used in this test
        ::testing::ValuesIn({3lu, 0lu}),
        ::details::ShapesIn({Tensors{input={1, 1, 1, 32};
}
, Tensors {
    input = {1, 1, 1, 128};
}
, Tensors {
    input = {1, 1, 1, 390};
}
, Tensors {
    input = {1, 1, 1, 82};
}
}),
        ::testing::ValuesIn({true, false}),
        ::testing::ValuesIn({true, false})
);

INSTANTIATE_TEST_SUITE_P(UnpackTests, UnpackTests,
                         TestCases,
                         UnpackTests::getTestCaseName);

const auto TestCasesScale = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::i4}), // TODO: add i8 as input for test
        ::testing::ValuesIn({ov::element::Type_t::f16, ov::element::Type_t::f32}),
        ::testing::ValuesIn({ov::element::Type_t::f16, ov::element::Type_t::f32}),
        ::testing::ValuesIn({ov::element::Type_t::dynamic}), // no used in this test
        ::testing::ValuesIn({3lu, 0lu}),
        ::details::ShapesIn({Tensors{input={1,32, 128};     scale = {1, 32, 1};
}
, Tensors {
    input = {32, 128};
    scale = {32, 1};
}
, Tensors {
    input = {64, 160};
    scale = {64, 1};
}
, Tensors {
    input = {1024, 4};
    scale = {64, 1};
}
, Tensors {
    input = {1, 1, 1024, 4};
    scale = {1, 1, 64, 1};
}
}),
        ::testing::ValuesIn({true, false}),
        ::testing::ValuesIn({true, false})
);

INSTANTIATE_TEST_SUITE_P(UnpackWithScaleTests, UnpackWithScaleTests,
                         TestCasesScale,
                         UnpackWithScaleTests::getTestCaseName);


const auto TestCasesScaleAndZeroPoints = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::u4}),
        ::testing::ValuesIn({ov::element::Type_t::f16}),
        ::testing::ValuesIn({ov::element::Type_t::f16}),
        ::testing::ValuesIn({ov::element::Type_t::u4}),
        ::testing::ValuesIn({3lu, 0lu}),
        ::details::ShapesIn({Tensors{input={1,32, 128};     scale = {1, 32, 1};},
                             Tensors{input={1,64, 160};     scale = {1, 64, 1};},
                             Tensors{input={1,1024, 4};     scale = {1, 64, 1};},
                             Tensors{input={1,1, 1024, 4};  scale = {1, 1, 64, 1};},
                             Tensors{input={64, 1};         scale = {64, 1};}}),
        ::testing::ValuesIn({true, false}),
        ::testing::ValuesIn({true, false})
);

INSTANTIATE_TEST_SUITE_P(UnpackTestsWithScaleAndZeroPoint, UnpackTestsWithScaleAndZeroPoint,
                         TestCasesScaleAndZeroPoints,
                         UnpackTestsWithScaleAndZeroPoint::getTestCaseName);

const auto TestCasesScaleAndZeroPoints2 = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::u4}),
        ::testing::ValuesIn({ov::element::Type_t::f16}),
        ::testing::ValuesIn({ov::element::Type_t::f32}),
        ::testing::ValuesIn({ov::element::Type_t::f32}),
        ::testing::ValuesIn({3lu, 0lu}),
        ::details::ShapesIn({Tensors{input={32, 32, 64};    scale = {32, 1, 64};},
                             Tensors{input={64, 64, 128};   scale = {64, 1, 128};},
                             Tensors{input={64, 32, 32};    scale = {64, 1, 32};}}),
        ::testing::ValuesIn({true, false}),
        ::testing::ValuesIn({true, false})
);

INSTANTIATE_TEST_SUITE_P(UnpackTestsWithScaleAndZeroPointTest2, UnpackTestsWithScaleAndZeroPointTest2,
                         TestCasesScaleAndZeroPoints2,
                         UnpackTestsWithScaleAndZeroPointTest2::getTestCaseName);

const auto TestCasesScaleAndZeroPoints3 = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::u4}),
        ::testing::ValuesIn({ov::element::Type_t::f16}),
        ::testing::ValuesIn({ov::element::Type_t::f16}),
        ::testing::ValuesIn({ov::element::Type_t::u4}),
        ::testing::ValuesIn({3lu, 0lu}),
        ::details::ShapesIn({Tensors{input={1, 32, 128};     scale = {1, 32, 1};   zerop = {1, 32, 1};},
                             Tensors{input={16, 64, 64};     scale = {16, 64, 1};  zerop = {16, 64, 1};},
                             Tensors{input={1, 1024, 4};     scale = {1, 64, 1};   zerop = {1, 32, 1};}}),
        ::testing::ValuesIn({true, false}),
        ::testing::ValuesIn({true, false})
);

INSTANTIATE_TEST_SUITE_P(UnpackTestsWithScaleAndZeroPointTest3, UnpackTestsWithScaleAndZeroPointTest3,
                         TestCasesScaleAndZeroPoints3,
                         UnpackTestsWithScaleAndZeroPointTest3::getTestCaseName);

} // anonymous namespace

#endif // __AVX2__
