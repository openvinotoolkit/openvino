// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/program.hpp"
#include "concatenation_inst.h"
#include "crop_inst.h"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "fully_connected_inst.h"
#include "gather_inst.h"
#include "pass_manager.h"
#include "permute_inst.h"
#include "reshape_inst.h"
#include "shape_of_inst.h"
#include "convolution_inst.h"
#include "dft_inst.h"
#include "to_string_utils.h"

#include "program_wrapper.h"

#include <memory>

using namespace cldnn;
using namespace ::tests;

TEST(add_required_reorders, prevent_input_dt_changing_for_convs) {
    auto& engine = get_test_engine();

    int input_b = 1, input_f = 16, input_y = 3, input_x = 3;
    int output_b = input_b, output_f = 16, output_y = 3, output_x = 3;

    auto input_mem = engine.allocate_memory({ {input_b, input_f, input_y, input_x}, data_types::u8, format::bs_fs_yx_bsv16_fsv32 });
    auto input2_mem = engine.allocate_memory({ {input_b, input_f, input_y, input_x}, data_types::u8, format::bs_fs_yx_bsv16_fsv32 });
    auto weights_mem = engine.allocate_memory({ {16, 16, 1, 1}, data_types::i8, format::bfyx });

    auto input = input_layout("input", input_mem->get_layout());
    auto input_const = data("input_const", input2_mem);
    auto weights = data("weights", weights_mem);
    auto eltwise1 = eltwise("eltwise1", input_info("input"), input_info("input_const"), eltwise_mode::sum);
    auto conv1 = convolution("conv1", input_info("eltwise1"), {"weights"}, {}, 1);
    auto output_reorder = reorder("reorder", input_info("conv1"), { data_types::f32, format::bfyx, { output_b, output_f, output_y, output_x } });

    topology topology_test(input, input_const, eltwise1, weights, conv1, output_reorder);

    ExecutionConfig config_test = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv1_impl_test = { format::bfyx, "", impl_types::ocl };
    config_test.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv1", conv1_impl_test } }));

    auto prog = program::build_program(engine, topology_test, config_test, false, true);
    program_wrapper::apply_opt_pass<add_required_reorders>(*prog);

    ASSERT_NE(prog, nullptr);
    ASSERT_TRUE(prog->has_node("conv1"));
    ASSERT_EQ(prog->get_node("conv1").get_dependency(0).get_output_layout(false).data_type, data_types::u8);
}
