// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/data.hpp"

#include "reorder_inst.h"
#include "fully_connected_inst.h"
#include "impls/registry/registry.hpp"
#include "graph/impls/ocl/register.hpp"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(weights_factory, reorder_test) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);
    const int input_f = 32, output_f = 32;


    auto weights_layout = layout(ov::PartialShape{ output_f, input_f }, data_types::f32, format::bfyx);
    auto weights_data_input = engine.allocate_memory(weights_layout);
    auto weights_data_vec = rg.generate_random_1d<float>(output_f * input_f, -1, 1);
    set_values(weights_data_input, weights_data_vec);

    cldnn::topology topology {
        input_layout("input", layout{ ov::PartialShape{ -1, input_f }, data_types::f32, format::bfyx }),
        data("weights", weights_data_input),
        fully_connected("fc", input_info("input"), "weights")
    };

    ov::intel_gpu::ImplementationDesc fc_impl_desc = { format::bfyx, "fully_connected_gpu_bf_tiled", impl_types::ocl };
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"fc", fc_impl_desc} })),
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    cldnn::network network(engine, topology, config);

    auto inst = network.get_primitive("fc");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);

    // Get required WeightsReorderParams
    auto weights_reorder_params = impl->get_weights_reorder_params();
    ASSERT_TRUE(weights_reorder_params != nullptr);

    // Constuct kernel_impl_params for weights reorder based requested WeightsReorderParams
    auto reorder_kernel_params = std::make_shared<kernel_impl_params>();
    reorder_kernel_params->desc = std::make_shared<reorder>("weights_reorder", input_info(), weights_reorder_params);
    reorder_kernel_params->unique_id = weights_reorder_params->hash();
    reorder_kernel_params->input_layouts.push_back(weights_reorder_params->get_input_layout());
    reorder_kernel_params->output_layouts.push_back(weights_reorder_params->get_output_layout());
    reorder_kernel_params->prog = network.get_program().get();

    // Create new generic_layer_impl
    auto factory = reorder::type_id()->get_best_impl(impl_types::ocl, shape_types::static_shape);
    auto reorder_impl = factory->create(*reorder_kernel_params);
    ASSERT_TRUE(reorder_impl != nullptr);

    // Compile kernel
    auto& kernel_cache = network.get_program()->get_kernels_cache();
    auto kernels = kernel_cache.compile(*reorder_kernel_params, reorder_impl->get_kernels_source());
    ASSERT_TRUE(kernels.size() == 1);
    reorder_impl->set_kernels(kernels);

    // Allocate memmory and execute generic_layer
    auto output_weights_layout = weights_reorder_params->get_output_layout();
    auto weights_data_output = engine.allocate_memory({ output_weights_layout });

    kernel_arguments_data args;
    args.inputs.push_back(weights_data_input);
    args.outputs.push_back(weights_data_output);

    auto reorder_inst = std::make_shared<cldnn::reorder_inst>(network);

    reorder_inst->set_impl(reorder_impl->clone());

    reorder_inst->get_impl()->set_arguments(*reorder_inst, args);
    reorder_inst->get_impl()->execute({}, *reorder_inst);

    network.get_stream().finish();

    // Compare with expected resutls
    cldnn::mem_lock<float> output_ptr(weights_data_output, get_test_stream());
    for (int o = 0; o < output_f; o++) {
        for (int i = 0; i < input_f; i++) {
            auto tensor_coord = tensor(std::vector<tensor::value_type>{o, i}, 0);
            size_t input_idx = output_weights_layout.get_linear_offset(tensor_coord);
            ASSERT_EQ(weights_data_vec[o * input_f + i], output_ptr[input_idx]);
        }
    }
}
