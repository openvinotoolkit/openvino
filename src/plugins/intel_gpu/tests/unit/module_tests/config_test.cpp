// Copyright (C) 2022-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/runtime/plugin_config.hpp"
#include "openvino/runtime/properties.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

TEST(config_test, basic) {
    ov::intel_gpu::NewExecutionConfig cfg;
    std::cerr << cfg.to_string();

    cfg.set_user_property(ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    cfg.set_property(ov::hint::inference_precision(ov::element::f32));

    std::cerr << "PROF: " << cfg.enable_profiling.value << std::endl;

    std::cerr << cfg.to_string();

    std::cerr << cfg.get_property(ov::hint::inference_precision) << std::endl;
    std::cerr << cfg.get_property(ov::hint::execution_mode) << std::endl;

    auto ctx = std::make_shared<ov::intel_gpu::RemoteContextImpl>("GPU", std::vector<device::ptr>{ get_test_engine().get_device() });
    cfg.finalize(ctx, {});
    std::cerr << cfg.to_string();
//     std::cerr << get_prop<ov::hint::inference_precision>() << std::endl;
//     std::cerr << get_prop<test1>() << std::endl;
}
