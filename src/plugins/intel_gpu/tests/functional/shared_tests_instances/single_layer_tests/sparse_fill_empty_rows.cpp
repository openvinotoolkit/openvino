// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/sparse_fill_empty_rows.hpp"

#include "common_test_utils/test_constants.hpp"

namespace ov::test {
INSTANTIATE_TEST_SUITE_P(smoke_SparseFillEmptyRows_static,
                         SparseFillEmptyRowsLayerTest,
                         SparseFillEmptyRowsLayerTest::GetStaticTestDataForDevice(ov::test::utils::DEVICE_GPU),
                         SparseFillEmptyRowsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_SparseFillEmptyRows_dynamic,
                         SparseFillEmptyRowsLayerTest,
                         SparseFillEmptyRowsLayerTest::GetDynamicTestDataForDevice(ov::test::utils::DEVICE_GPU),
                         SparseFillEmptyRowsLayerTest::getTestCaseName);
}  // namespace ov::test
