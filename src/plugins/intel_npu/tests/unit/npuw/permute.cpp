// // Copyright (C) 2024 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

// //#ifdef HAVE_AVX2
// #include "permute.hpp"

// namespace {

// const auto TestCases = ::testing::Combine(
//         ::testing::ValuesIn({ov::element::Type_t::i4, ov::element::Type_t::f16, ov::element::Type_t::f32}),
//         ::details::ShapesIn({Tensors{input={1, 10, 16};
// }
// , Tensors {
//     input = {1, 10, 128};
// }
// , Tensors {
//     input = {1, 16, 256};
// }
// , Tensors {
//     input = {1, 16, 300};
// }
// }),
// ::testing::ValuesIn(
//     {
//     std::vector<std::size_t>({2, 0, 1}),
//     std::vector<std::size_t>({0, 2, 1}),
//     std::vector<std::size_t>({1, 0, 2}),
//     std::vector<std::size_t>({1, 2, 0})
// }
// )
// );

// INSTANTIATE_TEST_SUITE_P(PermuteTests, PermuteTests, TestCases, PermuteTests::getTestCaseName);

// }  // anonymous namespace

// //#endif  // __AVX2__