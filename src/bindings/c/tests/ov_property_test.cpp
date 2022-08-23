// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"

TEST(ov_property, ov_properties_init_test) {
    ov_properties_t properties;
    OV_ASSERT_OK(ov_properties_init(&properties, 6));

    ov_properties_deinit(&properties);
}
