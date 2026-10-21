// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef HAVE_AVX2
#include "unpack.hpp"

#include "lazy_tensor.hpp"
#include "openvino/op/constant.hpp"

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

TEST_P(WidenSubbyteTests, widen) {
    ov::Tensor subbyte(fromType, input_shape, input.data());
    const auto widened = ov::npuw::util::widen_subbyte_to_i4(subbyte);
    ASSERT_EQ(widened.get_element_type(), ov::element::i4);
    ASSERT_EQ(widened.get_shape(), input_shape);
    // A packing bug then reports as a size mismatch here rather than as a read past the payload in std::equal.
    ASSERT_EQ(widened.get_byte_size(), ref_output.size());
    const auto* actual = static_cast<const int8_t*>(widened.data());
    ASSERT_TRUE(std::equal(ref_output.begin(), ref_output.end(), actual)) << "the widened payload differs";

    // The invariant the change rests on: the readers decode the same values from both containers. The codes
    // are small enough for f16 to hold exactly, so the comparison needs no tolerance. The suite has no unsigned
    // readers, so u2/u1 rest on the byte-level check above.
    if (fromType == ov::element::i2) {
        const auto elements = static_cast<int>(shape_size(input_shape));
        std::vector<int8_t> source_values(elements * sizeof(uint16_t));
        std::vector<int8_t> widened_values(source_values.size());
        details::unpack_i2f16(static_cast<const int8_t*>(subbyte.data()), source_values.data(), elements);
        details::unpack_i4f16(actual, widened_values.data(), elements);
        ASSERT_EQ(source_values, widened_values) << "the two containers decode to different values";
    }
}

TEST(WidenSubbyteLazyTensor, evals_the_permute_convert_chain) {
    // The chain DQMatMulGQ2iP registers: the closure is permuted, then converted to i4.
    const ov::Shape shape{2, 2, 16};
    {
        const ov::element::Type type(ov::element::Type_t::i2);
        const std::size_t elements = shape_size(shape);
        // Dims 0 and 1 are equal, so permute({1, 0, 2}) leaves the payload's order unchanged.
        const std::vector<int8_t> data(elements * type.bitwidth() / 8, static_cast<int8_t>(0xE4));
        auto constant = std::make_shared<ov::op::v0::Constant>(type, shape, data.data());
        const auto lazy = ov::npuw::weights::LazyTensor(constant).permute({1, 0, 2}).convert(ov::element::i4);

        ASSERT_EQ(lazy.eval_meta().type, ov::element::i4);
        ASSERT_EQ(lazy.eval_meta().shape, shape);

        std::vector<int8_t> expected(elements / 2);
        details::widen_i2_to_i4(data.data(), expected.data(), static_cast<int>(elements));
        const auto widened = lazy.eval();
        ASSERT_EQ(widened.get_element_type(), ov::element::i4);
        ASSERT_EQ(widened.get_byte_size(), expected.size());
        const auto* actual = static_cast<const int8_t*>(widened.data());
        ASSERT_TRUE(std::equal(expected.begin(), expected.end(), actual)) << "the widened payload differs";
    }
}

const auto TestCasesWidenSubbyte = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::i2,
                             ov::element::Type_t::u2,
                             ov::element::Type_t::u1}),
        ::testing::ValuesIn({ov::element::Type_t::i4}),
        ::testing::ValuesIn({ov::element::Type_t::dynamic}),  // no used in this test
        ::testing::ValuesIn({ov::element::Type_t::dynamic}),  // no used in this test
        ::testing::Values(3lu),
        ::details::ShapesIn({Tensors{input={1, 1, 1, 32};
}
, Tensors {
    input = {1, 10, 128};
}
, Tensors {
    input = {1, 16, 256};
}
, Tensors {
    input = {1, 16, 320};
}
}),
        ::testing::Values(true),
        ::testing::Values(true)
);

INSTANTIATE_TEST_SUITE_P(WidenSubbyteTests, WidenSubbyteTests, TestCasesWidenSubbyte, WidenSubbyteTests::getTestCaseName);

const auto TestCasesScale = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::i2,
                             ov::element::Type_t::i4}), // TODO: add i8 as input for test
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
    // One shape for every width in the list: 24 codes is a whole byte at 1, 2 and 4 bits.
    input = {1, 2, 12};
    scale = {1, 2, 1};
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

} // anonymous namespace

#endif // __AVX2__
