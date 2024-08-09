// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "execution_graph_tests/duplicate_inputs_outputs_names.hpp"

#include "common_test_utils/test_constants.hpp"

using namespace ExecutionGraphTests;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_duplicateInputsOutputsNames,
                         ExecGraphDuplicateInputsOutputsNames,
                         ::testing::Values(ov::test::utils::DEVICE_CPU),
                         ExecGraphDuplicateInputsOutputsNames::getTestCaseName);

}  // namespace
