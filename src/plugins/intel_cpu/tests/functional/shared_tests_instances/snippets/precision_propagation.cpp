// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "snippets/precision_propagation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::test::snippets;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
    //ngraph::element::f16
};

const std::vector<PrecisionPropagationTestValues> params = {
    //{
    //    ngraph::element::i32,
    //    ngraph::PartialShape({1, 3, 16, 16}),
    //    ngraph::element::i32,
    //    ngraph::PartialShape({1, 3, 16, 16}),
    //},
    {
        ngraph::element::i8,
        ngraph::PartialShape({1, 3, 16, 16}),
        ngraph::element::i8,
        ngraph::PartialShape({1, 3, 16, 16}),
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_Snippets, PrecisionPropagationTest,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(params)),
    PrecisionPropagationTest::getTestCaseName);
}  // namespace
