// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"

TEST(ov_layout, ov_layout_create_static_layout) {
    const char* str = "[N,C,H,W]";
    const char* desc = "NCHW";
    ov_layout_t* layout = nullptr;

    OV_ASSERT_OK(ov_layout_create(desc, &layout));
    const char* res = ov_layout_to_string(layout);

    EXPECT_STREQ(res, str);
    ov_layout_free(layout);
    ov_free(res);
}

TEST(ov_layout, ov_layout_create_dynamic_layout) {
    const char* str = "[N,...,C]";
    const char* desc = "N...C";
    ov_layout_t* layout = nullptr;

    OV_ASSERT_OK(ov_layout_create(desc, &layout));
    const char* res = ov_layout_to_string(layout);

    EXPECT_STREQ(res, str);
    ov_layout_free(layout);
    ov_free(res);
}
