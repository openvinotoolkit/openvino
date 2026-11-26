// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// P2-a S2 queue guards of GPU_ENABLE_INORDER_LIFETIME_RESTRICTIONS (design 19, 3.5):
//   P-G2  (L24)  : a blob compiled for an in-order queue with the switch on must not be imported with an out-of-order queue.
//   P-G2b (L24b) : an external out-of-order queue (queue type is detected from the handle, not from the config) is rejected
//                  when the config asks for the in-order restrictions.
// The switch only changes the compile-time memory restrictions from S4 on; the guards are already active in S2.

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/subgraph_builders/conv_pool_relu.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

#ifdef OV_GPU_WITH_OCL_RT
#    include "openvino/runtime/intel_gpu/ocl/ocl.hpp"
#    include "remote_tensor_tests/helpers.hpp"
#endif

namespace {

// The options are release-internal: a user cannot pass them through Core properties (PluginConfig::set_user_property rejects them), the supported
// route is the environment (applied after finalize_impl, therefore also over the oneDNN "force in-order" rule and over an imported blob).
// oneDNN forces an in-order queue unless use_onednn is decided explicitly, which would make the out-of-order variants vacuous.
struct env_guard {
    std::vector<std::string> names;
    void set(const std::string& name, const std::string& value) {
        ::setenv(name.c_str(), value.c_str(), 1);
        names.push_back(name);
    }
    ~env_guard() {
        for (const auto& n : names) {
            ::unsetenv(n.c_str());
        }
    }
};

void set_compile_env(env_guard& g, const char* queue, bool lifetime_switch) {
    g.set("OV_GPU_QUEUE_TYPE", queue);
    g.set("OV_GPU_USE_ONEDNN", "0");
    g.set("OV_GPU_ENABLE_INORDER_LIFETIME_RESTRICTIONS", lifetime_switch ? "1" : "0");
}

class InorderLifetimeQueueGuard : public ::testing::Test {
public:
    std::shared_ptr<ov::Model> model;
    void SetUp() override {
        model = ov::test::utils::make_conv_pool_relu();
    }
};

// L24. Export with in_order + switch on; importing with an out-of-order queue override throws, importing unchanged works.
TEST_F(InorderLifetimeQueueGuard, L24_import_overriding_queue_to_out_of_order_throws) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core core;
    ov::CompiledModel compiled;
    std::stringstream blob;
    {
        env_guard env;
        set_compile_env(env, "in-order", true);
        OV_ASSERT_NO_THROW(compiled = core.compile_model(model, ov::test::utils::DEVICE_GPU));
        OV_ASSERT_NO_THROW(compiled.export_model(blob));
    }

    {   // control: imported unchanged, it infers
        std::stringstream in(blob.str());
        ov::CompiledModel imported;
        OV_ASSERT_NO_THROW(imported = core.import_model(in, ov::test::utils::DEVICE_GPU));
        auto req = imported.create_infer_request();
        req.set_input_tensor(ov::test::utils::create_and_fill_tensor(imported.input().get_element_type(), imported.input().get_shape()));
        OV_ASSERT_NO_THROW(req.infer());
    }
    {   // the override that invalidates the restrictions (the blob keeps the compile-time switch; only the queue type is overridden)
        env_guard env;
        env.set("OV_GPU_QUEUE_TYPE", "out-of-order");
        env.set("OV_GPU_USE_ONEDNN", "0");
        std::stringstream in(blob.str());
        EXPECT_THROW(core.import_model(in, ov::test::utils::DEVICE_GPU), ov::Exception);
    }
}

// L24 control: a blob compiled with the switch OFF can still be imported as out-of-order (the guard must not fire without the switch).
TEST_F(InorderLifetimeQueueGuard, L24_control_switch_off_import_as_out_of_order_works) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core core;
    ov::CompiledModel compiled;
    std::stringstream blob;
    {
        env_guard env;
        set_compile_env(env, "in-order", false);
        OV_ASSERT_NO_THROW(compiled = core.compile_model(model, ov::test::utils::DEVICE_GPU));
        OV_ASSERT_NO_THROW(compiled.export_model(blob));
    }
    ov::CompiledModel imported;
    env_guard env;
    env.set("OV_GPU_QUEUE_TYPE", "out-of-order");
    env.set("OV_GPU_USE_ONEDNN", "0");
    OV_ASSERT_NO_THROW(imported = core.import_model(blob, ov::test::utils::DEVICE_GPU));
}

#ifdef OV_GPU_WITH_OCL_RT
// L24b. The shared queue created by the OpenCL helper is out-of-order (CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE).
TEST_F(InorderLifetimeQueueGuard, L24b_external_out_of_order_queue_with_inorder_restrictions_throws) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core core;
    auto ocl_instance = std::make_shared<OpenCL>();
    auto remote_context = ov::intel_gpu::ocl::ClContext(core, ocl_instance->_queue.get());
    // control: the external out-of-order queue itself is fine without the switch
    {
        env_guard env;
        set_compile_env(env, "in-order", false);
        OV_ASSERT_NO_THROW(core.compile_model(model, remote_context));
    }
    // config says in-order + switch on, the stream behind the handle is out-of-order
    {
        env_guard env;
        set_compile_env(env, "in-order", true);
        EXPECT_THROW(core.compile_model(model, remote_context), ov::Exception);
    }
}

// L24c (M2). The blob was compiled with the switch on; the import call overrides the switch to OFF and uses an external out-of-order queue. The
// restrictions inside the blob are still the in-order ones, so the guard must fire although the runtime switch is off.
TEST_F(InorderLifetimeQueueGuard, L24c_import_with_the_switch_overridden_off_and_an_external_ooo_queue_throws) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    ov::Core core;
    std::stringstream blob;
    {
        env_guard env;
        set_compile_env(env, "in-order", true);
        ov::CompiledModel compiled;
        OV_ASSERT_NO_THROW(compiled = core.compile_model(model, ov::test::utils::DEVICE_GPU));
        OV_ASSERT_NO_THROW(compiled.export_model(blob));
    }
    auto ocl_instance = std::make_shared<OpenCL>();
    auto remote_context = ov::intel_gpu::ocl::ClContext(core, ocl_instance->_queue.get());
    {   // control: a blob compiled WITHOUT the switch imports fine on the same external queue
        std::stringstream plain;
        {
            env_guard env;
            set_compile_env(env, "in-order", false);
            ov::CompiledModel compiled;
            OV_ASSERT_NO_THROW(compiled = core.compile_model(model, ov::test::utils::DEVICE_GPU));
            OV_ASSERT_NO_THROW(compiled.export_model(plain));
        }
        env_guard env;
        set_compile_env(env, "in-order", false);
        OV_ASSERT_NO_THROW(core.import_model(plain, remote_context));
    }
    env_guard env;
    set_compile_env(env, "in-order", false);   // runtime switch overridden to off (and queue kept in-order)
    EXPECT_THROW(core.import_model(blob, remote_context), ov::Exception);
}
#endif  // OV_GPU_WITH_OCL_RT

}  // namespace
