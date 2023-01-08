// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "test_utils/test_utils.h"
#include "intel_gpu/runtime/debug_configuration.hpp"

using namespace cldnn;
using namespace ::tests;

TEST(debug_config_test, check_debug_config_off_on_release) {
#ifdef NDEBUG
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(1) {
        GTEST_FAIL();   /* This should be disabled in case of release build */
    }
#endif
}
