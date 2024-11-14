// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/data.hpp>
#include "intel_gpu/runtime/compilation_context.hpp"

#include "fully_connected_inst.h"

#include "program_wrapper.h"

using namespace cldnn;
using namespace ::tests;

TEST(kernel_impl_params_relevance, weights_layout) {
    auto& engine = get_test_engine();

    const int32_t in_b = 1;
    const int32_t in_f = 4;
    const int32_t wei_o = 3;

    auto input_dyn_layout = layout{ ov::PartialShape{ ov::Dimension(1, 10), in_f }, data_types::f32, format::bfyx };
    auto actual_input_data = engine.allocate_memory(layout{ ov::PartialShape{ in_b, in_f }, data_types::f32, format::bfyx });
    auto weights_data = engine.allocate_memory({ ov::PartialShape{ wei_o, in_f }, data_types::f32, format::bfyx });

    cldnn::topology topology{
        input_layout("input", input_dyn_layout),
        data("weights", weights_data),
        fully_connected("fc", input_info("input"), "weights")
    };

    auto fc_opt_impl = ov::intel_gpu::ImplementationDesc(format::bfyx, "fully_connected_gpu_bf_tiled", impl_types::ocl);
    ExecutionConfig cfg{ ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc", fc_opt_impl} }),
                         ov::intel_gpu::optimize_data(true),
                         ov::intel_gpu::allow_new_shape_infer(true) };

    // 1. Compile network with forced `fully_connected_gpu_bf_tiled` kernel => optimized shape-agnostic
    //    kernel will be used
    network network(engine, topology, cfg);
    network.set_input_data("input", actual_input_data);

    // 2. Force reference `fully_connected_gpu_bfyx_ref` kernel impl before execution,
    //    so during _node->type()->choose_impl(*_node); call for static kernel version reference
    //    impl will be used. Call execute() to trigger desired kernel compilation
    auto fc_ref_impl = ov::intel_gpu::ImplementationDesc(format::bfyx, "fully_connected_gpu_bfyx_ref", impl_types::ocl);
    auto force_impl_prop = ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc", fc_ref_impl} });
    program_wrapper::update_configs_properties(*network.get_program(), {force_impl_prop});

    network.execute();

    // 3. WA: Call wait_all() to wait for all queued kernels compilation finish (including above `fully_connected_gpu_bfyx_ref`)
    network.get_program()->get_compilation_context().wait_all();

    // 4. Call execute() second time with same input shape to use pre-compiled `fully_connected_gpu_bfyx_ref` kernel
    network.execute();

    // 5. Get FC instance
    auto inst = network.get_primitive("fc");
    auto fc_inst = std::dynamic_pointer_cast<fully_connected_inst>(inst);
    ASSERT_TRUE(fc_inst != nullptr);

    // 6. The weight memory of fc node is reordered at build time for fully_connected_gpu_bf_tiled kernel
    ASSERT_EQ(fc_inst->get_node().get_dependency(1).get_output_layout().format, format::os_iyx_osv16);

    // 7. Requset instance's weights memory, compare it with original weights buffer and check
    //    if original layout is used (required for `fully_connected_gpu_bfyx_ref` kernel)
    auto used_weights_memory = fc_inst->weights_memory()->get_layout();
    ASSERT_EQ(weights_data->get_layout().compatible(used_weights_memory), true);
}
