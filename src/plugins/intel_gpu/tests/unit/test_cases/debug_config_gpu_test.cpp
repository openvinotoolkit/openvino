// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "intel_gpu/runtime/debug_configuration.hpp"

using namespace cldnn;
using namespace ::tests;

TEST(debug_config_test, check_debug_config_off_on_release) {
#ifdef NDEBUG
    auto config = get_test_default_config(get_test_engine());
    GPU_DEBUG_IF(1) {
        GTEST_FAIL();   /* This should be disabled in case of release build */
    }
#endif
}
