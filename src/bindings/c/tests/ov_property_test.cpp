// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "ov_test.hpp"

TEST(ov_property, ov_properties_create_test) {
    ov_properties_t properties;
    OV_ASSERT_OK(ov_properties_create(&properties, 6));

    ov_properties_free(&properties);
}
