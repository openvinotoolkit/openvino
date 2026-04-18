// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/convert_color_nv12.hpp"
#include "functional_test_utils/crash_handler.hpp"

namespace ov {
namespace test {
TEST_P(ConvertColorNV12LayerTest, Inference) {
    ov::test::utils::CrashHandler::SetUpPipelineAfterCrash(true);
    run();
}
} // namespace test
} // namespace ov
