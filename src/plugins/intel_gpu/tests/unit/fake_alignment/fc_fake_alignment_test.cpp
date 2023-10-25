// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "fully_connected_inst.h"

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

    auto input_layout_prim = std::make_shared<input_layout>("input", p.input_layout);
    auto weight_layout_prim = std::make_shared<input_layout>("weight", p.weight_layout);
    auto fully_connected_prim = std::make_shared<fully_connected>("output", input_info("input"), "weight", "", p.data_type);

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

    }));

}  // fake_alignment_tests
