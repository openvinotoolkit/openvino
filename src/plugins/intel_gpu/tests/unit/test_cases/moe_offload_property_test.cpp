// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/op/parameter.hpp"

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

TEST(moe_offload_property_test, auto_string_normalizes_to_auto_sentinel) {
    auto config = get_test_default_config(get_test_engine());

    ASSERT_NO_THROW(config.set_user_property({{ov::intel_gpu::offload_ratio.name(), std::string("AUTO")}}));
    ASSERT_EQ(config.get_offload_ratio(), ov::intel_gpu::OFFLOAD_RATIO_AUTO);
}

TEST(moe_offload_property_test, auto_string_is_case_insensitive) {
    auto config = get_test_default_config(get_test_engine());

    ASSERT_NO_THROW(config.set_user_property({{ov::intel_gpu::offload_ratio.name(), std::string("auto")}}));
    ASSERT_EQ(config.get_offload_ratio(), ov::intel_gpu::OFFLOAD_RATIO_AUTO);
}

TEST(moe_offload_property_test, set_back_to_zero_disables) {
    auto config = get_test_default_config(get_test_engine());

    config.set_property(ov::intel_gpu::offload_ratio(37));
    ASSERT_EQ(config.get_offload_ratio(), 37U);

    config.set_property(ov::intel_gpu::offload_ratio(0));
    ASSERT_EQ(config.get_offload_ratio(), 0U);
}

TEST(moe_offload_property_test, auto_ratio_enables_weights_path_in_apply_rt_info) {
    auto config = get_test_default_config(get_test_engine());
    config.set_property(ov::intel_gpu::offload_ratio(ov::intel_gpu::OFFLOAD_RATIO_AUTO));
    ASSERT_EQ(config.get_offload_ratio(), ov::intel_gpu::OFFLOAD_RATIO_AUTO);

    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 16});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{input}, ov::ParameterVector{input});
    const std::string fake_weights_path = "/path/to/model.bin";
    model->get_rt_info()["__weights_path"] = fake_weights_path;

    auto context = std::make_shared<ov::intel_gpu::RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{get_test_engine().get_device()});
    config.finalize(context.get(), model.get());

    ASSERT_EQ(config.get_weights_path(), fake_weights_path);
}

TEST(moe_offload_property_test, typed_cxx_sentinel_sets_auto) {
    auto config = get_test_default_config(get_test_engine());
    config.set_property(ov::intel_gpu::offload_ratio(ov::intel_gpu::OFFLOAD_RATIO_AUTO));
    ASSERT_EQ(config.get_offload_ratio(), ov::intel_gpu::OFFLOAD_RATIO_AUTO);
}

TEST(moe_offload_property_test, user_property_integer_auto_sentinel) {
    auto config = get_test_default_config(get_test_engine());
    config.set_user_property({{ov::intel_gpu::offload_ratio.name(), int64_t{-1}}});
    ASSERT_EQ(config.get_offload_ratio(), ov::intel_gpu::OFFLOAD_RATIO_AUTO);
}

TEST(moe_offload_property_test, invalid_values_rejected) {
    auto config = get_test_default_config(get_test_engine());
    // Value > 100 should throw
    EXPECT_THROW(config.set_user_property({{ov::intel_gpu::offload_ratio.name(), int64_t{101}}}), ov::Exception);
    // Value < -1 should throw
    EXPECT_THROW(config.set_user_property({{ov::intel_gpu::offload_ratio.name(), int64_t{-2}}}), ov::Exception);
    // Invalid string should throw
    EXPECT_THROW(config.set_user_property({{ov::intel_gpu::offload_ratio.name(), std::string("INVALID")}}), ov::Exception);
}

TEST(moe_offload_property_test, config_clone_preserves_offload_ratio) {
    auto config = get_test_default_config(get_test_engine());
    config.set_property(ov::intel_gpu::offload_ratio(ov::intel_gpu::OFFLOAD_RATIO_AUTO));
    auto cloned = config.clone();
    ASSERT_EQ(cloned.get_offload_ratio(), ov::intel_gpu::OFFLOAD_RATIO_AUTO);

    config.set_property(ov::intel_gpu::offload_ratio(42));
    cloned = config.clone();
    ASSERT_EQ(cloned.get_offload_ratio(), 42);
}

}  // namespace ov::test