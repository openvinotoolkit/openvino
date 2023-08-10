// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/softmax.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

const std::vector<ov::Shape> inputShape = {
    ov::Shape{1, 16},
    ov::Shape{1, 32},
    ov::Shape{1, 1},
    ov::Shape{1, 9},
    ov::Shape{1, 17},
    ov::Shape{1, 19},
    ov::Shape{1, 49},
    ov::Shape{1, 50},
    ov::Shape{5, 16},
    ov::Shape{5, 32},
    ov::Shape{5, 1},
    ov::Shape{5, 9},
    ov::Shape{5, 17},
    ov::Shape{5, 19},
    ov::Shape{5, 49},
    ov::Shape{5, 50},
    ov::Shape{1, 3, 128, 128},
    ov::Shape{1, 3, 128, 129},
    ov::Shape{1, 3, 128, 130},
    ov::Shape{1, 3, 128, 1},
    ov::Shape{1, 3, 128, 9},
    ov::Shape{1, 3, 128, 16},
    ov::Shape{1, 3, 128, 17},
    ov::Shape{1, 3, 128, 20},
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Softmax, Softmax,
                     ::testing::Combine(
                             ::testing::ValuesIn(inputShape),
                             ::testing::Values(-1),
                             ::testing::Values(1),
                             ::testing::Values(1),
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                     Softmax::getTestCaseName);

const std::vector<std::pair<ov::Shape, ov::Shape>> inputShapesPair = {
    std::pair<ov::Shape, ov::Shape>{ov::Shape{1, 5, 16, 35}, ov::Shape{1, 5, 16, 35}},
    std::pair<ov::Shape, ov::Shape>{ov::Shape{1, 5, 16, 1}, ov::Shape{1, 5, 16, 35}},
    std::pair<ov::Shape, ov::Shape>{ov::Shape{1, 5, 16, 35}, ov::Shape{1, 5, 1, 1}},
    std::pair<ov::Shape, ov::Shape>{ov::Shape{1, 5, 16, 1}, ov::Shape{1, 5, 16, 1}},
    std::pair<ov::Shape, ov::Shape>{ov::Shape{1, 5, 16, 35}, ov::Shape{1, 5, 1, 35}},
    std::pair<ov::Shape, ov::Shape>{ov::Shape{1, 5, 1, 35}, ov::Shape{1, 5, 1, 35}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_AddSoftmax, AddSoftmax,
                     ::testing::Combine(
                             ::testing::ValuesIn(inputShapesPair),
                             ::testing::Values(-1),
                             ::testing::Values(1),
                             ::testing::Values(1),
                             ::testing::Values(ov::test::utils::DEVICE_CPU)),
                     AddSoftmax::getTestCaseName);

} // namespace
} // namespace snippets
} // namespace test
} // namespace ov