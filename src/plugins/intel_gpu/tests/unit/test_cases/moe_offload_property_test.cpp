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

    ASSERT_EQ(config.get_moe_offload_ratio(), 0U);

    config.set_property(ov::intel_gpu::moe_offload_ratio(37));

    ASSERT_EQ(config.get_moe_offload_ratio(), 37U);
}

}  // namespace ov::test