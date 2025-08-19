// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef HAVE_AVX2
#    include "transpose.hpp"

namespace {

const auto TestCases = ::testing::Combine(
        ::testing::ValuesIn({ov::element::Type_t::i4, ov::element::Type_t::f32}),
        ::details::ShapesIn({Tensors{input={1, 10, 16};
}
, Tensors {
    input = {1, 10, 128};
}
, Tensors {
    input = {1, 16, 256};
}
, Tensors {
    input = {1, 16, 300};
}
})
);

TEST_P(TransposeTests, transpose) {
    ASSERT_NO_THROW_WITH_MESSAGE(outTensor = ov::npuw::util::transpose(inTensor));
    int8_t* dst = static_cast<int8_t*>(outTensor.data());
    output = std::vector<int8_t>(dst, dst + output.size());
    ASSERT_TRUE(details::ArraysMatch(output, ref_output));
}

INSTANTIATE_TEST_SUITE_P(TransposeTests, TransposeTests, TestCases, TransposeTests::getTestCaseName);

}  // anonymous namespace

#endif  // __AVX2__
