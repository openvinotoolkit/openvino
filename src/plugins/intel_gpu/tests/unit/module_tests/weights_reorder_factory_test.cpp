// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/reorder.hpp"

#include "reorder_inst.h"
#include "fully_connected_inst.h"
#include "registry/registry.hpp"
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

// Regression test for grouped convolution weight reorder: goiyx -> g_os_is_yx_isv16_osv16

TEST(weights_factory, grouped_conv_reorder_goiyx_to_g_os_is_yx_isv16_osv16) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    const int groups = 32;
    const int oc_per_group = 4;
    const int ic_per_group = 4;
    const int filter_x = 3;
    const int filter_y = 3;

    // Allocate weights in goiyx format [groups, OC/group, IC/group, filter_y, filter_x]
    auto weights_size = tensor(group(groups), batch(oc_per_group), feature(ic_per_group), spatial(filter_x, filter_y));
    auto weights_input_layout = layout(data_types::f32, format::goiyx, weights_size);
    auto weights_data_input = engine.allocate_memory(weights_input_layout);
    auto weights_data_vec = rg.generate_random_1d<float>(
        static_cast<int>(weights_input_layout.get_linear_size()), -1, 1);
    set_values(weights_data_input, weights_data_vec);

    // Directly construct the output layout in g_os_is_yx_isv16_osv16 format
    auto weights_output_layout = layout(data_types::f32, format::g_os_is_yx_isv16_osv16, weights_size);

    // Create WeightsReorderParams directly (no convolution needed)
    auto weights_reorder_params = std::make_shared<WeightsReorderParams>(
        weights_input_layout, weights_output_layout, false, true);

    // Build a minimal network just to get program/stream context for kernel compilation
    cldnn::topology topology{
        input_layout("input", layout{ ov::PartialShape{ 1, 1 }, data_types::f32, format::bfyx }),
        reorder("out", input_info("input"), layout{ ov::PartialShape{ 1, 1 }, data_types::f32, format::bfyx })
    };

    ExecutionConfig config = get_test_default_config(engine);
    cldnn::network network(engine, topology, config);

    // Construct kernel_impl_params for the weights reorder
    auto reorder_kernel_params = std::make_shared<kernel_impl_params>();
    reorder_kernel_params->desc = std::make_shared<reorder>("weights_reorder", input_info(), weights_reorder_params);
    reorder_kernel_params->unique_id = weights_reorder_params->hash();
    reorder_kernel_params->input_layouts.push_back(weights_reorder_params->get_input_layout());
    reorder_kernel_params->output_layouts.push_back(weights_reorder_params->get_output_layout());
    reorder_kernel_params->prog = network.get_program().get();

    // Create reorder implementation
    auto factory = reorder::type_id()->get_best_impl(impl_types::ocl, shape_types::static_shape);
    auto reorder_impl = factory->create(*reorder_kernel_params);
    ASSERT_TRUE(reorder_impl != nullptr);

    // Compile and execute the reorder kernel
    auto& kernel_cache = network.get_program()->get_kernels_cache();
    auto kernels = kernel_cache.compile(*reorder_kernel_params, reorder_impl->get_kernels_source());
    ASSERT_TRUE(kernels.size() == 1);
    reorder_impl->set_kernels(kernels);

    auto weights_data_output = engine.allocate_memory(weights_output_layout);

    kernel_arguments_data args;
    args.inputs.push_back(weights_data_input);
    args.outputs.push_back(weights_data_output);

    auto reorder_inst = std::make_shared<cldnn::reorder_inst>(network);
    reorder_inst->set_impl(reorder_impl->clone());
    reorder_inst->get_impl()->set_arguments(*reorder_inst, args);
    reorder_inst->get_impl()->execute({}, *reorder_inst);

    OV_ASSERT_NO_THROW(network.get_stream().finish());

    // Verify: each element at logical position [g, o, i, y, x] in the source goiyx layout
    // must appear at the correct position in the g_os_is_yx_isv16_osv16 output layout.
    cldnn::mem_lock<float> output_ptr(weights_data_output, get_test_stream());

    for (int g = 0; g < groups; ++g) {
        for (int o = 0; o < oc_per_group; ++o) {
            for (int i = 0; i < ic_per_group; ++i) {
                for (int y = 0; y < filter_y; ++y) {
                    for (int x = 0; x < filter_x; ++x) {
                        auto src_coord = tensor(group(g), batch(o), feature(i), spatial(x, y, 0, 0));
                        size_t src_offset = weights_input_layout.get_linear_offset(src_coord);
                        size_t dst_offset = weights_output_layout.get_linear_offset(src_coord);
                        ASSERT_EQ(weights_data_vec[src_offset], output_ptr[dst_offset])
                            << "at g=" << g << " o=" << o << " i=" << i
                            << " y=" << y << " x=" << x;
                    }
                }
            }
        }
    }
}
