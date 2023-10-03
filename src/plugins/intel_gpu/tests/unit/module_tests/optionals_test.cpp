// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "intel_gpu/runtime/optionals.hpp"
#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(optional_data_types, empty) {
    cldnn::optional_data_type o;
    ASSERT_FALSE(o.has_value());
    ASSERT_ANY_THROW(o.value());
    ASSERT_EQ(o.value_or(data_types::u8), data_types::u8);
}

TEST(optional_data_types, basic) {
    cldnn::data_types dt = cldnn::data_types::f32;
    cldnn::optional_data_type o = dt;
    ASSERT_EQ(o.value(), cldnn::data_types::f32);

    cldnn::optional_data_type o1 = o;
    ASSERT_EQ(o1.value(), cldnn::data_types::f32);

    cldnn::optional_data_type o2 = cldnn::data_types::f32;
    ASSERT_EQ(o2.value(), cldnn::data_types::f32);

    {
        cldnn::optional_data_type o2 = { cldnn::data_types::f32 };
        ASSERT_EQ(o2.value(), cldnn::data_types::f32);
    }

    optional_data_type o3(cldnn::data_types::f16);
    ASSERT_EQ(o3.value(), cldnn::data_types::f16);

    optional_data_type o4(cldnn::data_types::f32);
    ASSERT_EQ(o4.value(), cldnn::data_types::f32);
}
