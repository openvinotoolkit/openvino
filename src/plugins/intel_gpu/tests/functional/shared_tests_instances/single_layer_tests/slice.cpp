// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/slice.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::Slice8LayerTest;
using ov::test::Slice8SpecificParams;
using ov::test::Slice8Params;

static const std::vector<Slice8SpecificParams> static_params = {
        Slice8SpecificParams{ {{{}, {{ 16 }}}}, { 4 }, { 12 }, { 1 }, { 0 } },
        Slice8SpecificParams{ {{{}, {{ 16 }}}}, { 0 }, { 8 }, { 2 }, { 0 } },
        Slice8SpecificParams{ {{{}, {{ 20, 10, 5 }}}}, { 0, 0}, { 10, 20}, { 1, 1 }, { 1, 0 } },
        Slice8SpecificParams{ {{{}, {{ 1, 2, 12, 100 }}}}, { 0, 1, 0, 1 }, { 1, 2, 5, 100 }, { 1, 1, 1, 10 }, {} },
        Slice8SpecificParams{ {{{}, {{ 1, 12, 100 }}}}, { 0, 9, 0 }, { 1, 11, 1 }, { 1, 1, 1 }, { 0, 1, -1 } },
        Slice8SpecificParams{ {{{}, {{ 1, 12, 100 }}}}, { 0, 1, 0 }, { 10, -1, 10 }, { 1, 1, 1 }, { -3, -2, -1} },
        Slice8SpecificParams{ {{{}, {{ 2, 12, 100 }}}}, { 1, 12, 100 }, { 0, 7, 0 }, { -1, -1, -1 }, {} },
        Slice8SpecificParams{ {{{}, {{ 2, 12, 100 }}}}, { 1, 4, 99 }, { 0, 9, 0 }, { -1, 2, -1 }, {} },
        Slice8SpecificParams{ {{{}, {{ 2, 12, 100 }}}}, { -1, -1, -1 }, { 0, 4, 0 }, { -1, -2, -1 }, {} },
        Slice8SpecificParams{ {{{}, {{ 2, 12, 100 }}}}, { -1, -1, -1 }, { 0, 0, 4 }, { -1, -1, -1 }, {2, 0, 1} },
        Slice8SpecificParams{ {{{}, {{ 2, 12, 100 }}}}, { 0, 0, 4 }, { -5, -1, -1 }, { 1, 2, 1 }, {2, 0, 1} },
        Slice8SpecificParams{ {{{}, {{ 2, 4, 5, 5, 68 }}}}, { 0, 1, 0, 0, 0 }, {
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max(),
                std::numeric_limits<std::int64_t>::max() }, { 1, 1, 1, 1, 16 }, {} },
        Slice8SpecificParams{ {{{}, {{ 10, 12 }}}}, { -1, 1 }, { -9999, 10 }, { -1, 1 }, {} },
};

static const std::vector<ov::element::Type> types = {
    ov::element::i64,
    ov::element::i32,
    ov::element::f32,
    ov::element::f16
};

INSTANTIATE_TEST_SUITE_P(
        smoke_GPU, Slice8LayerTest,
        ::testing::Combine(
            ::testing::ValuesIn(static_params),
            ::testing::ValuesIn(types),
            ::testing::Values(ov::test::utils::DEVICE_GPU)),
        Slice8LayerTest::getTestCaseName);


}  // namespace