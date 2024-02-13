// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/crash_handler.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/summary/environment.hpp"
#include "functional_test_utils/summary/op_summary.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "ov_models/pass/convert_prc.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace LayerTestsUtils {

using TargetDevice = std::string;

class LayerTestsCommon {
protected:
    LayerTestsCommon();
};

}  // namespace LayerTestsUtils
