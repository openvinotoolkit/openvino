// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/runtime/core.hpp"

TEST(ImportModel, ImportModelWithNullContextThrows) {
    ov::Core core;
    ov::RemoteContext context;
    std::istringstream stream("None");
    ASSERT_THROW(core.import_model(stream, context, {}), ov::Exception);
}
