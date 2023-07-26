// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/transpose_softmax.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<ov::Shape> inputShape = {
    ov::Shape{1, 128, 3, 16},
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeSoftmax, TransposeSoftmax,
                     ::testing::Combine(
                             ::testing::Values(inputShape),
                             ::testing::Values(std::vector<int64_t>{0, 2, 3, 1}),
                             ::testing::Values(-1),
                             ::testing::Values(1),
                             ::testing::Values(1),
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                     TransposeSoftmax::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_TransposeSoftmaxEltwise, TransposeSoftmaxEltwise,
                         ::testing::Combine(
                                 ::testing::Values(inputShape),
                                 ::testing::Values(std::vector<int64_t>{0, 2, 3, 1}),
                                 ::testing::Values(-1),
                                 ::testing::Values(1),
                                 ::testing::Values(1),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         TransposeSoftmax::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov