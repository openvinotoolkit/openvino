// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef HAVE_AVX2
#    include "unpack.hpp"

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

const auto TestCasesU4 = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::u4}),
        ::testing::ValuesIn({ov::element::Type_t::f16}),
        ::testing::ValuesIn({ov::element::Type_t::dynamic}), // no used in this test
        ::testing::ValuesIn({ov::element::Type_t::dynamic}), // no used in this test
        ::testing::ValuesIn({3lu, 0lu}),
        ::details::ShapesIn({Tensors{input={1, 1, 1, 64};
}
, Tensors {
    input = {1, 1, 1, 128};
}
}),
        ::testing::ValuesIn({true, false}),
        ::testing::ValuesIn({true, false})
);

INSTANTIATE_TEST_SUITE_P(UnpackTestsU4, UnpackTestsU4,
                         TestCasesU4,
                         UnpackTestsU4::getTestCaseName);

// The widening: every sub-byte storage the host presents in an i4 container decodes to the same values there.
const auto TestCasesWidenSubbyte = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::i2,
                             ov::element::Type_t::u1}),
        ::testing::ValuesIn({ov::element::Type_t::i4}),
        ::testing::ValuesIn({ov::element::Type_t::dynamic}),  // no used in this test
        ::testing::ValuesIn({ov::element::Type_t::dynamic}),  // no used in this test
        ::testing::Values(3lu),
        ::details::ShapesIn({Tensors{input={1, 1, 1, 32};}}),
        ::testing::Values(true),
        ::testing::Values(true));

INSTANTIATE_TEST_SUITE_P(WidenSubbyteTests,
                         WidenSubbyteTests,
                         TestCasesWidenSubbyte,
                         WidenSubbyteTests::getTestCaseName);

TEST_P(WidenSubbyteTests, widen) {
    ov::Tensor subbyte(fromType, input_shape, input.data());
    const auto widened = ov::npuw::util::widen_subbyte_to_i4(subbyte);
    ASSERT_EQ(widened.get_element_type(), ov::element::i4);
    ASSERT_EQ(widened.get_shape(), input_shape);
    // A packing bug then reports as a size mismatch here rather than as a read past the payload in std::equal.
    ASSERT_EQ(widened.get_byte_size(), ref_output.size());
    const auto* actual = static_cast<const int8_t*>(widened.data());
    ASSERT_TRUE(std::equal(ref_output.begin(), ref_output.end(), actual)) << "the widened payload differs";

    // The invariant the change rests on: the readers decode the same values from both containers. The codes are
    // small enough for f16 to hold exactly, so the comparison needs no tolerance. The suite has no unsigned
    // readers, so u1 rests on the byte-level check above.
    if (fromType == ov::element::i2) {
        const auto elements = static_cast<int>(shape_size(input_shape));
        std::vector<int8_t> source_values(elements * sizeof(uint16_t));
        std::vector<int8_t> widened_values(source_values.size());
        details::unpack_i2f16(static_cast<const int8_t*>(subbyte.data()), source_values.data(), elements);
        details::unpack_i4f16(actual, widened_values.data(), elements);
        ASSERT_EQ(source_values, widened_values) << "the two containers decode to different values";
    }
}

TEST(WidenSubbyteOddTail, decodes_theLastCodes) {
    const ov::Shape shape{1, 5};
    const std::size_t elements = shape_size(shape);
    // 0, 1, -2, -1, 1; the unused codes are nonzero so reading a sixth code changes the high nibble.
    const std::vector<int8_t> data{static_cast<int8_t>(0xE4), static_cast<int8_t>(0xFD)};
    ov::Tensor subbyte(ov::element::i2, shape, const_cast<int8_t*>(data.data()));

    // The i4 reference reads pairs; include the zero padding code in both decoded buffers.
    std::vector<int8_t> expected((elements + 1) * sizeof(uint16_t));
    details::unpack_i2f16(data.data(), expected.data(), static_cast<int>(elements));

    const auto widened = ov::npuw::util::widen_subbyte_to_i4(subbyte);
    ASSERT_EQ(widened.get_element_type(), ov::element::i4);
    ASSERT_EQ(widened.get_byte_size(), (elements + 1) / 2);
    const auto* actual = static_cast<const uint8_t*>(widened.data());
    EXPECT_EQ(actual[widened.get_byte_size() - 1] & 0xF0, 0) << "the unused high nibble must be zero";
    std::vector<int8_t> widened_values(expected.size());
    details::unpack_i4f16(static_cast<const int8_t*>(widened.data()),
                          widened_values.data(),
                          static_cast<int>(elements + 1));
    ASSERT_EQ(widened_values, expected) << "the widened tail differs";
}

// The sign coding, pinned by hand: 0b1000_0001 is element 0 = 1 and element 7 = 1 (MSB-first), and the widened
// container holds two codes per byte with the low nibble first, so the first byte is 0xF1 (+1 low, -1 high) and
// the last is 0x1F.
TEST(WidenSubbyteSignGrid, widensAKnownByte) {
    const ov::Shape shape{1, 8};
    const std::vector<uint8_t> data{0x81};
    ov::Tensor subbyte(ov::element::u1, shape, const_cast<uint8_t*>(data.data()));

    const auto widened = ov::npuw::util::widen_subbyte_to_i4(subbyte);
    ASSERT_EQ(widened.get_element_type(), ov::element::i4);
    ASSERT_EQ(widened.get_byte_size(), 4u);

    const auto* actual = static_cast<const uint8_t*>(widened.data());
    EXPECT_EQ(actual[0], 0xF1) << "the first pair's codes";
    EXPECT_EQ(actual[1], 0xFF);
    EXPECT_EQ(actual[2], 0xFF);
    EXPECT_EQ(actual[3], 0x1F) << "the last pair's codes";
}
const auto TestCasesScale = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::i2,
                             ov::element::Type_t::i4,
                             ov::element::Type_t::u1}), // TODO: add i8 as input for test
        ::testing::ValuesIn({ov::element::Type_t::f16, ov::element::Type_t::f32}),
        ::testing::ValuesIn({ov::element::Type_t::f16, ov::element::Type_t::f32}),
        ::testing::ValuesIn({ov::element::Type_t::dynamic}),  // no used in this test
        ::testing::ValuesIn({3lu, 0lu}),
        ::details::ShapesIn({Tensors{input={1,32, 128};     scale = {1, 32, 1};},
                             Tensors{input={32, 128};     scale = {32, 1};},
                             Tensors{input={64, 160};     scale = {64, 1};},
                             Tensors{input={1024, 4};     scale = {64, 1};},
                             Tensors{input={1, 1, 1024, 4};  scale = {1, 1, 64, 1};},
                             // Twenty codes exercise partial-byte tails for u1.
                             Tensors{input={1, 2, 10}; scale = {1, 2, 1};}}),
        ::testing::ValuesIn({true, false}),
        ::testing::ValuesIn({true, false}));

INSTANTIATE_TEST_SUITE_P(UnpackWithScaleTests,
                         UnpackWithScaleTests,
                         TestCasesScale,
                         UnpackWithScaleTests::getTestCaseName);

const auto TestCasesScaleI4F16 = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::i4}),
        ::testing::ValuesIn({ov::element::Type_t::f16}),
        ::testing::ValuesIn({ov::element::Type_t::f32}),
        ::testing::ValuesIn({ov::element::Type_t::dynamic}), // no used in this test
        ::testing::ValuesIn({3lu, 0lu}),
        ::details::ShapesIn({Tensors{input={16, 32, 256};     scale = {16, 1, 256};
}
, Tensors {
    input = {32, 64, 128};
    scale = {32, 1, 128};
}
}),
        ::testing::ValuesIn({true, false}),
        ::testing::ValuesIn({true, false})
);

INSTANTIATE_TEST_SUITE_P(UnpackWithScaleTestsI4F16, UnpackWithScaleTestsI4F16,
                         TestCasesScaleI4F16,
                         UnpackWithScaleTestsI4F16::getTestCaseName);

const auto TestCasesScaleF8 = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::f8e4m3, ov::element::Type_t::f8e5m2, ov::element::Type_t::f8e8m0}),
        ::testing::ValuesIn({ov::element::Type_t::f16}),
        ::testing::ValuesIn({ov::element::Type_t::f32}),
        ::testing::ValuesIn({ov::element::Type_t::dynamic}), // no used in this test
        ::testing::ValuesIn({3lu, 0lu}),
        ::details::ShapesIn({Tensors{input={1, 64, 128};     scale = {1, 1, 128};
}
, Tensors {
    input = {32, 128};
    scale = {32, 1};
}
, Tensors {
    input = {64, 256};
    scale = {64, 1};
}
}),
        ::testing::ValuesIn({true, false}),
        ::testing::ValuesIn({true, false})
);

INSTANTIATE_TEST_SUITE_P(UnpackF8WithScaleTests, UnpackF8WithScaleTests,
                         TestCasesScaleF8,
                         UnpackF8WithScaleTests::getTestCaseName);

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

const auto TestCasesI8Asymmetric = ::testing::Combine(
        ::testing::ValuesIn({ov::Shape{2, 4}, ov::Shape{2, 8}, ov::Shape{3, 9}, ov::Shape{4, 16}}),
        ::testing::ValuesIn({ov::element::Type_t::f16, ov::element::Type_t::f32}),
        ::testing::ValuesIn({0lu, 2lu, 3lu}),
        ::testing::ValuesIn({true, false}),
        ::testing::ValuesIn({true, false}));

INSTANTIATE_TEST_SUITE_P(I8AsymmetricUnpackTests,
                         I8AsymmetricUnpackTests,
                         TestCasesI8Asymmetric,
                         I8AsymmetricUnpackTests::getTestCaseName);
TEST(UnpackU1OddTail, decodes_theLastCodes) {
    const ov::Shape shape{1, 13};
    const std::size_t elements = shape_size(shape);
    const std::vector<uint8_t> data{0x60, 0xA0};
    ov::Tensor subbyte(ov::element::u1, shape, const_cast<uint8_t*>(data.data()));
    ASSERT_EQ(subbyte.get_byte_size(), data.size());

    std::vector<int8_t> expected(elements * sizeof(uint16_t));
    details::unpack_u1f16(reinterpret_cast<const int8_t*>(data.data()),
                           expected.data(),
                           static_cast<int>(elements));

    std::vector<ov::float16> scale_values{ov::float16(1.0f)};
    auto scale = ov::make_tensor(ov::element::f16, ov::Shape{1, 1}, scale_values.data());
    std::vector<int8_t> to_storage(expected.size());
    auto to = ov::make_tensor(ov::element::f16, shape, to_storage.data());
    ASSERT_NO_THROW_WITH_MESSAGE(
        ov::npuw::util::unpack(ov::get_tensor_impl(subbyte), scale, to, ov::npuw::util::UnpackOptions{true, 0, false}));
    ASSERT_EQ(to_storage, expected) << "the tail differs";

    // 0xA0 is 1010_0000, so element 12 is zero and decodes to -0.5.
    const auto* last = reinterpret_cast<const uint16_t*>(to_storage.data()) + (elements - 1);
    EXPECT_EQ(*last, details::float_to_half(-0.5f));
}

} // anonymous namespace

#endif // __AVX2__
