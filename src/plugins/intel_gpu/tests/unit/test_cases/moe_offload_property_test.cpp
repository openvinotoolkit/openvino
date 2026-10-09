// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "intel_gpu/runtime/internal_properties.hpp"

namespace ov::test {

using ::tests::get_test_default_config;
using ::tests::get_test_engine;

TEST(moe_offload_property_test, execution_config_roundtrip) {
    auto config = get_test_default_config(get_test_engine());

    ASSERT_EQ(config.get_offload_ratio(), 0U);

    config.set_property(ov::intel_gpu::offload_ratio(37));

    ASSERT_EQ(config.get_offload_ratio(), 37U);
}

TEST(moe_offload_property_test, default_value_is_zero) {
    auto config = get_test_default_config(get_test_engine());
    ASSERT_EQ(config.get_offload_ratio(), 0U);
}

TEST(moe_offload_property_test, set_and_get_various_values) {
    auto config = get_test_default_config(get_test_engine());

    config.set_property(ov::intel_gpu::offload_ratio(1));
    ASSERT_EQ(config.get_offload_ratio(), 1U);

    config.set_property(ov::intel_gpu::offload_ratio(50));
    ASSERT_EQ(config.get_offload_ratio(), 50U);

    config.set_property(ov::intel_gpu::offload_ratio(100));
    ASSERT_EQ(config.get_offload_ratio(), 100U);
}

TEST(moe_offload_property_test, set_back_to_zero_disables) {
    auto config = get_test_default_config(get_test_engine());

    config.set_property(ov::intel_gpu::offload_ratio(37));
    ASSERT_EQ(config.get_offload_ratio(), 37U);

    config.set_property(ov::intel_gpu::offload_ratio(0));
    ASSERT_EQ(config.get_offload_ratio(), 0U);
}

}  // namespace ov::test