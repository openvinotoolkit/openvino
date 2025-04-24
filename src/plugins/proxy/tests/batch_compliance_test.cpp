// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/proxy/properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "proxy_tests.hpp"

using namespace ov::proxy::tests;

TEST_F(ProxyTests, can_parse_and_inherit_batch_property) {
    register_plugin_support_reshape(core, "MOCK_DEVICE", {{ov::proxy::configuration::alias.name(), "MOCK_DEVICE"}});
    auto available_devices = core.get_available_devices();
    auto model = create_model_with_add();
    auto compiled_model_default = core.compile_model(model, "MOCK_DEVICE", ov::hint::performance_mode("THROUGHPUT"));
#ifdef ENABLE_AUTO_BATCH
    EXPECT_NO_THROW(compiled_model_default.get_property(ov::auto_batch_timeout));  // batch enabled by default
    EXPECT_EQ(compiled_model_default.get_property(ov::auto_batch_timeout), 1000);  // default value
#endif
    auto compiled_model_with_batch = core.compile_model(model,
                                                        "MOCK_DEVICE",
                                                        ov::hint::performance_mode("THROUGHPUT"),
                                                        ov::hint::allow_auto_batching(true),
                                                        ov::auto_batch_timeout(8));
#ifdef ENABLE_AUTO_BATCH
    EXPECT_NO_THROW(compiled_model_with_batch.get_property(ov::auto_batch_timeout));
    EXPECT_EQ(compiled_model_with_batch.get_property(ov::auto_batch_timeout), 8);
#endif
    auto compiled_model_no_batch = core.compile_model(model,
                                                      "MOCK_DEVICE",
                                                      ov::hint::performance_mode("THROUGHPUT"),
                                                      ov::hint::allow_auto_batching(false));
    EXPECT_ANY_THROW(compiled_model_no_batch.get_property(ov::auto_batch_timeout));
}

TEST_F(ProxyTests, can_parse_and_inherit_batch_property_for_device_name_with_id) {
    register_plugin_support_reshape(core, "MOCK_DEVICE", {{ov::proxy::configuration::alias.name(), "MOCK_DEVICE"}});
    auto available_devices = core.get_available_devices();
    auto model = create_model_with_add();
    auto compiled_model_default = core.compile_model(model, "MOCK_DEVICE.1", ov::hint::performance_mode("THROUGHPUT"));
#ifdef ENABLE_AUTO_BATCH
    EXPECT_NO_THROW(compiled_model_default.get_property(ov::auto_batch_timeout));  // batch enabled by default
    EXPECT_EQ(compiled_model_default.get_property(ov::auto_batch_timeout), 1000);  // default value
#endif
    auto compiled_model_with_batch = core.compile_model(model,
                                                        "MOCK_DEVICE.1",
                                                        ov::hint::performance_mode("THROUGHPUT"),
                                                        ov::hint::allow_auto_batching(true),
                                                        ov::auto_batch_timeout(8));
#ifdef ENABLE_AUTO_BATCH
    EXPECT_NO_THROW(compiled_model_with_batch.get_property(ov::auto_batch_timeout));
    EXPECT_EQ(compiled_model_with_batch.get_property(ov::auto_batch_timeout), 8);
#endif
    auto compiled_model_no_batch = core.compile_model(model,
                                                      "MOCK_DEVICE.2",
                                                      ov::hint::performance_mode("THROUGHPUT"),
                                                      ov::hint::allow_auto_batching(false));
    EXPECT_ANY_THROW(compiled_model_no_batch.get_property(ov::auto_batch_timeout));
}