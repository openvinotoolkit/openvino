// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/permute.hpp>

#include "fully_connected_inst.h"
#include "eltwise_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace fake_alignment_tests {

struct fc_fake_align_params {
    layout input_layout;
    layout weight_layout;
    data_types data_type;
    layout expected_input_layout_igpu;
    layout expected_output_layout_igpu;
    layout expected_input_layout_dgpu;
    layout expected_output_layout_dgpu;

};

class fully_connected_fake_align_test : public testing::TestWithParam<fc_fake_align_params> {};

TEST_P(fully_connected_fake_align_test, fake_alignment) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_size = p.input_layout.get_partial_shape().size();
    auto input_layout_prim = std::make_shared<input_layout>("input", p.input_layout);
    auto weight_layout_prim = std::make_shared<input_layout>("weight", p.weight_layout);
    auto fully_connected_prim = std::make_shared<fully_connected>("output", input_info("input"), "weight", "", p.data_type, input_size);

    cldnn::program prog(engine);

    auto& input_node = prog.get_or_create(input_layout_prim);
    auto& weight_node = prog.get_or_create(weight_layout_prim);
    auto& fully_connected_node = prog.get_or_create(fully_connected_prim);
    program_wrapper::add_connection(prog, input_node, fully_connected_node);
    program_wrapper::add_connection(prog, weight_node, fully_connected_node);

    auto impl_param = fully_connected_node.get_kernel_impl_params();
    impl_param->output_layouts[0] = fully_connected_inst::calc_output_layouts<ov::PartialShape>(fully_connected_node, *fully_connected_node.get_kernel_impl_params())[0];

    if (impl_param->get_input_layout().is_dynamic() || impl_param->get_output_layout().is_dynamic()) {
        EXPECT_THROW(fully_connected_inst::get_fake_aligned_params(*impl_param), std::exception);
    } else {
        auto updated_param = fully_connected_inst::get_fake_aligned_params(*impl_param);
        if (!engine.get_device_info().supports_immad) {
            ASSERT_EQ(updated_param.get_input_layout(), p.expected_input_layout_igpu);
            ASSERT_EQ(updated_param.get_output_layout(), p.expected_output_layout_igpu);
        } else {
            ASSERT_EQ(updated_param.get_input_layout(), p.expected_input_layout_dgpu);
            ASSERT_EQ(updated_param.get_output_layout(), p.expected_output_layout_dgpu);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, fully_connected_fake_align_test,
    testing::ValuesIn(std::vector<fc_fake_align_params>{
        {
            layout{ov::PartialShape{0, 1024}, data_types::i8, format::bfyx, padding{{1,1,1,1}, 0}},    // input_layout
            layout{ov::PartialShape{1000, 1024}, data_types::i8, format::bfyx},                        // weight layout
            data_types::f16,
            layout{ov::PartialShape{0, 1024}, data_types::i8, format::bfyx, padding{{1,1,1,1}, 0}},    // fake_aligned input layout_igpu
            layout{ov::PartialShape{0, 1000}, data_types::f16, format::bfyx},                          // fake_aligned output layout_igpu
            layout{ov::PartialShape{0, 1024}, data_types::i8, format::bfyx, padding{{1,1,1,1}, 0}},    // fake_aligned input layout_dgpu
            layout{ov::PartialShape{0, 1000}, data_types::f16, format::bfyx}                           // fake_aligned output layout_dgpu
        },
        {
            layout{ov::PartialShape{11, 1024}, data_types::i8, format::bfyx, padding{{1,1,1,1}, 0}},   // input_layout
            layout{ov::PartialShape{1000, 1024}, data_types::i8, format::bfyx},                        // weight layout
            data_types::f16,
            layout{ov::PartialShape{16, 1024}, data_types::i8, format::bfyx, padding{{1,1,1,1}, 0}},   // fake_aligned input layout_igpu
            layout{ov::PartialShape{16, 1000}, data_types::f16, format::bfyx},                         // fake_aligned output layout_igpu
            layout{ov::PartialShape{16, 1024}, data_types::i8, format::bfyx, padding{{1,1,1,1}, 0}},   // fake_aligned input layout_dgpu
            layout{ov::PartialShape{16, 1000}, data_types::f16, format::bfyx}                          // fake_aligned output layout_dgpu

        },
        {
            layout{ov::PartialShape{133, 511}, data_types::i8, format::bfyx, padding{{1,1,1,1}, 0}},   // input_layout
            layout{ov::PartialShape{800, 511}, data_types::i8, format::bfyx},                          // weight layout
            data_types::f16,
            layout{ov::PartialShape{144, 511}, data_types::i8, format::bfyx, padding{{1,1,1,1}, 0}},   // fake_aligned input layout_igpu
            layout{ov::PartialShape{144, 800}, data_types::f16, format::bfyx},                         // fake_aligned output layout_igpu
            layout{ov::PartialShape{136, 511}, data_types::i8, format::bfyx, padding{{1,1,1,1}, 0}},   // fake_aligned input layout_dgpu
            layout{ov::PartialShape{136, 800}, data_types::f16, format::bfyx}                          // fake_aligned output layout_dgpu
        },
        {
            layout{ov::PartialShape::dynamic(2), data_types::i8, format::bfyx, padding{{1,1,1,1}, 0}}, // input_layout
            layout{ov::PartialShape{1000, 1024}, data_types::i8, format::bfyx},                        // weight layout
            data_types::f16,
            layout{ov::PartialShape{-1, -1}, data_types::i8, format::bfyx},                            // fake_aligned input layout_igpu // dummy
            layout{ov::PartialShape{-1, -1}, data_types::f16, format::bfyx},                           // fake_aligned output layout_igpu // dummy
            layout{ov::PartialShape{-1, -1}, data_types::i8, format::bfyx},                            // fake_aligned input layout_dgpu // dummy
            layout{ov::PartialShape{-1, -1}, data_types::f16, format::bfyx}                            // fake_aligned output layout_dgpu // dummy
        },
        {
            layout{ov::PartialShape{1, 55, 511}, data_types::f16, format::bfyx},                       // input_layout
            layout{ov::PartialShape{800, 511}, data_types::f16, format::bfyx},                         // weight layout
            data_types::f16,
            layout{ov::PartialShape{64, 1, 511}, data_types::f16, format::bfyx},                       // fake_aligned input layout_igpu
            layout{ov::PartialShape{64, 1, 800}, data_types::f16, format::bfyx},                       // fake_aligned output layout_igpu
            layout{ov::PartialShape{56, 1, 511}, data_types::f16, format::bfyx},                       // fake_aligned input layout_dgpu
            layout{ov::PartialShape{56, 1, 800}, data_types::f16, format::bfyx}                        // fake_aligned output layout_dgpu
        },
        {
            layout{ov::PartialShape{2, 55, 511}, data_types::f16, format::bfyx},                       // input_layout
            layout{ov::PartialShape{800, 511}, data_types::f16, format::bfyx},                         // weight layout
            data_types::f16,
            layout{ov::PartialShape{112, 1, 511}, data_types::f16, format::bfyx},                      // fake_aligned input layout_igpu
            layout{ov::PartialShape{112, 1, 800}, data_types::f16, format::bfyx},                      // fake_aligned output layout_igpu
            layout{ov::PartialShape{112, 1, 511}, data_types::f16, format::bfyx},                      // fake_aligned input layout_dgpu
            layout{ov::PartialShape{112, 1, 800}, data_types::f16, format::bfyx}                       // fake_aligned output layout_dgpu
        },
        {
            layout{ov::PartialShape{55, 1, 511}, data_types::f16, format::bfyx},                       // input_layout
            layout{ov::PartialShape{800, 511}, data_types::f16, format::bfyx},                         // weight layout
            data_types::f16,
            layout{ov::PartialShape{64, 1, 511}, data_types::f16, format::bfyx},                       // fake_aligned input layout_igpu
            layout{ov::PartialShape{64, 1, 800}, data_types::f16, format::bfyx},                       // fake_aligned output layout_igpu
            layout{ov::PartialShape{56, 1, 511}, data_types::f16, format::bfyx},                       // fake_aligned input layout_dgpu
            layout{ov::PartialShape{56, 1, 800}, data_types::f16, format::bfyx}                        // fake_aligned output layout_dgpu
        },
        {
            layout{ov::PartialShape{240, 1,  511}, data_types::f16, format::bfyx},                     // input_layout
            layout{ov::PartialShape{800, 511}, data_types::f16, format::bfyx},                         // weight layout
            data_types::f16,
            layout{ov::PartialShape{240, 1, 511}, data_types::f16, format::bfyx},                      // fake_aligned input layout_igpu
            layout{ov::PartialShape{240, 1, 800}, data_types::f16, format::bfyx},                      // fake_aligned output layout_igpu
            layout{ov::PartialShape{240, 1, 511}, data_types::f16, format::bfyx},                      // fake_aligned input layout_dgpu
            layout{ov::PartialShape{240, 1, 800}, data_types::f16, format::bfyx}                       // fake_aligned output layout_dgpu
        },
        {
            layout{ov::PartialShape{241, 1, 511}, data_types::f16, format::bfyx},                      // input_layout
            layout{ov::PartialShape{800, 511}, data_types::f16, format::bfyx},                         // weight layout
            data_types::f16,
            layout{ov::PartialShape{256, 1, 511}, data_types::f16, format::bfyx},                      // fake_aligned input layout_igpu
            layout{ov::PartialShape{256, 1, 800}, data_types::f16, format::bfyx},                      // fake_aligned output layout_igpu
            layout{ov::PartialShape{248, 1, 511}, data_types::f16, format::bfyx},                      // fake_aligned input layout_dgpu
            layout{ov::PartialShape{248, 1, 800}, data_types::f16, format::bfyx}                       // fake_aligned output layout_dgpu
        },
        {
            layout{ov::PartialShape{257, 1, 511}, data_types::f16, format::bfyx},                      // input_layout
            layout{ov::PartialShape{800, 511}, data_types::f16, format::bfyx},                         // weight layout
            data_types::f16,
            layout{ov::PartialShape{272, 1, 511}, data_types::f16, format::bfyx},                      // fake_aligned input layout_igpu
            layout{ov::PartialShape{272, 1, 800}, data_types::f16, format::bfyx},                      // fake_aligned output layout_igpu
            layout{ov::PartialShape{264, 1, 511}, data_types::f16, format::bfyx},                      // fake_aligned input layout_dgpu
            layout{ov::PartialShape{264, 1, 800}, data_types::f16, format::bfyx}                       // fake_aligned output layout_dgpu
        },
        {
            layout{ov::PartialShape{55, 1, 511}, data_types::f16, format::bfyx, padding{{2,0,1,0}, 0}}, // input_layout
            layout{ov::PartialShape{800, 511}, data_types::f16, format::bfyx},                          // weight layout
            data_types::f16,
            layout{ov::PartialShape{64, 1, 511}, data_types::f16, format::bfyx, padding{{2,0,1,0}, 0}}, // fake_aligned input layout_igpu
            layout{ov::PartialShape{64, 1, 800}, data_types::f16, format::bfyx},                        // fake_aligned output layout_igpu
            layout{ov::PartialShape{56, 1, 511}, data_types::f16, format::bfyx, padding{{2,0,1,0}, 0}}, // fake_aligned input layout_dgpu
            layout{ov::PartialShape{56, 1, 800}, data_types::f16, format::bfyx}                         // fake_aligned output layout_dgpu
        },
        {
            layout{ov::PartialShape{55, 1, 511}, data_types::f16, format::bfyx, padding{{0,1,1,0}, 0}}, // input_layout
            layout{ov::PartialShape{800, 511}, data_types::f16, format::bfyx},                          // weight layout
            data_types::f16,
            layout{ov::PartialShape{55, 1, 511}, data_types::f16, format::bfyx, padding{{0,1,1,0}, 0}}, // fake_aligned input layout_igpu
            layout{ov::PartialShape{55, 1, 800}, data_types::f16, format::bfyx},                        // fake_aligned output layout_igpu
            layout{ov::PartialShape{55, 1, 511}, data_types::f16, format::bfyx, padding{{0,1,1,0}, 0}}, // fake_aligned input layout_dgpu
            layout{ov::PartialShape{55, 1, 800}, data_types::f16, format::bfyx}                         // fake_aligned output layout_dgpu
        },

        /* int4 compressed weights */
        {
            layout{ov::PartialShape{240, 1, 511}, data_types::f16, format::bfyx},                      // input_layout
            layout{ov::PartialShape{800, 511}, data_types::u4, format::bfyx},                          // weight layout
            data_types::f16,
            layout{ov::PartialShape{240, 1, 511}, data_types::f16, format::bfyx},                      // fake_aligned input layout_igpu
            layout{ov::PartialShape{240, 1, 800}, data_types::f16, format::bfyx},                      // fake_aligned output layout_igpu
            layout{ov::PartialShape{240, 1, 511}, data_types::f16, format::bfyx},                      // fake_aligned input layout_dgpu
            layout{ov::PartialShape{240, 1, 800}, data_types::f16, format::bfyx}                       // fake_aligned output layout_dgpu
        },
        {
            layout{ov::PartialShape{241, 1, 511}, data_types::f16, format::bfyx},                      // input_layout
            layout{ov::PartialShape{800, 511}, data_types::u4, format::bfyx},                          // weight layout
            data_types::f16,
            layout{ov::PartialShape{256, 1, 511}, data_types::f16, format::bfyx},                      // fake_aligned input layout_igpu
            layout{ov::PartialShape{256, 1, 800}, data_types::f16, format::bfyx},                      // fake_aligned output layout_igpu
            layout{ov::PartialShape{248, 1, 511}, data_types::f16, format::bfyx},                      // fake_aligned input layout_dgpu
            layout{ov::PartialShape{248, 1, 800}, data_types::f16, format::bfyx}                       // fake_aligned output layout_dgpu
        },
        {
            layout{ov::PartialShape{257, 1, 511}, data_types::f16, format::bfyx},                      // input_layout
            layout{ov::PartialShape{800, 511}, data_types::u4, format::bfyx},                          // weight layout
            data_types::f16,
            layout{ov::PartialShape{320, 1, 511}, data_types::f16, format::bfyx},                      // fake_aligned input layout_igpu
            layout{ov::PartialShape{320, 1, 800}, data_types::f16, format::bfyx},                      // fake_aligned output layout_igpu
            layout{ov::PartialShape{264, 1, 511}, data_types::f16, format::bfyx},                      // fake_aligned input layout_dgpu
            layout{ov::PartialShape{264, 1, 800}, data_types::f16, format::bfyx}                       // fake_aligned output layout_dgpu
        },
    }));

class fully_connected_skip_fake_align_test : public testing::TestWithParam<fc_fake_align_params> {};

// Skip fake alignment when fused desc has full tensor dependency
TEST_P(fully_connected_skip_fake_align_test, skip_fake_alignment_case) {
    auto p = GetParam();

    auto& engine = get_test_engine();
    topology topology;
    cldnn::program prog(engine);

    topology.add(input_layout("input", p.input_layout));
    topology.add(input_layout("eltwise_data1",p.input_layout));
    topology.add(eltwise("eltwise_add1", { input_info("input"), input_info("eltwise_data1") }, eltwise_mode::sum));

    topology.add(input_layout("weights", p.weight_layout));
    topology.add(fully_connected("fc_prim1", input_info("eltwise_add1"), "weights", "",
                 cldnn::data_types::f32, p.input_layout.get_rank(), p.weight_layout.get_rank()));

    topology.add(input_layout("bias",
                 layout{ov::PartialShape{1, 1, p.expected_output_layout_igpu.get_dims()[2]}, cldnn::data_types::f32, cldnn::format::bfyx}));
    topology.add(eltwise("bias_add", { input_info("fc_prim1"), input_info("bias") }, eltwise_mode::sum));

    topology.add(input_layout("dequantize_scale",
                 layout{ov::PartialShape{1, 1, p.expected_output_layout_igpu.get_dims()[2]}, cldnn::data_types::f32, cldnn::format::bfyx}));
    topology.add(eltwise("eltwise_multiply", { input_info("bias_add"), input_info("dequantize_scale") }, eltwise_mode::prod));

    topology.add(input_layout("eltwise_data2", p.expected_output_layout_igpu));
    topology.add(eltwise("eltwise_add2", { input_info("eltwise_multiply"), input_info("eltwise_data2") }, eltwise_mode::sum));
    topology.add(permute("permute", input_info("eltwise_add2"), {2, 1, 0}));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);

    auto impl_param = network.get_primitive("fc_prim1")->get_impl_params();

    if (impl_param->get_input_layout().is_dynamic() || impl_param->get_output_layout().is_dynamic()) {
        EXPECT_THROW(fully_connected_inst::get_fake_aligned_params(*impl_param), std::exception);
    } else {
        auto updated_param = fully_connected_inst::get_fake_aligned_params(*impl_param);
        if (!engine.get_device_info().supports_immad) {
            ASSERT_EQ(updated_param.get_input_layout(), p.expected_input_layout_igpu);
            ASSERT_EQ(updated_param.get_output_layout(), p.expected_output_layout_igpu);
        } else {
            ASSERT_EQ(updated_param.get_input_layout(), p.expected_input_layout_dgpu);
            ASSERT_EQ(updated_param.get_output_layout(), p.expected_output_layout_dgpu);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, fully_connected_skip_fake_align_test,
    testing::ValuesIn(std::vector<fc_fake_align_params>{
        {
            layout{ov::PartialShape{1, 1000, 2048}, data_types::u8, format::bfyx},    // input_layout
            layout{ov::PartialShape{512, 2048}, data_types::i8, format::bfyx},        // weight layout
            data_types::f32,
            layout{ov::PartialShape{1, 1000, 2048}, data_types::u8, format::bfyx},    // skiped fake_aligned input layout_igpu
            layout{ov::PartialShape{1, 1000, 512}, data_types::f32, format::bfyx},    // skipped fake_aligned output layout_igpu
            layout{ov::PartialShape{1, 1000, 2048}, data_types::u8, format::bfyx},    // skipped fake_aligned input layout_dgpu
            layout{ov::PartialShape{1, 1000, 512}, data_types::f32, format::bfyx}     // skipped fake_aligned output layout_dgpu
        },
    }));
}  // fake_alignment_tests
