// Copyright (C) 2025-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "primitive_inst_test_helper.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/eltwise.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <thread>
#include <type_traits>
#include <fstream>
#include <tuple>

#include "convolution_inst.h"

using namespace cldnn;
using namespace ::tests;

namespace {
const std::string no_bias = "";
}

struct mem_reset_params {
    ov::Dimension::value_type in_channel;
    bool is_dynamic;
    bool need_reset;
};

class mem_reset_test : public testing::TestWithParam<mem_reset_params> {};

TEST_P(mem_reset_test, need_reset_output_memory_test) {
    auto p = GetParam();
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    tests::random_generator rg(GET_SUITE_NAME);

    ov::PartialShape target_pshape = {1, p.in_channel, 64, 64};
    ov::PartialShape input_pshape;

    if (p.is_dynamic) {
        for (size_t i = 0; i < target_pshape.size(); ++i) {
            input_pshape.emplace_back(ov::Dimension());
        }
        input_pshape[1] = target_pshape[1];
    } else {
        input_pshape = target_pshape;
    }

    ov::PartialShape weights_pshape = {16, p.in_channel, 3, 3};
    layout in_layout{ input_pshape, data_types::f16, format::bfyx };
    layout weights_layout{ weights_pshape, data_types::f16, format::bfyx };
    auto input_data = rg.generate_random_1d<ov::float16>(ov::shape_size(target_pshape.get_shape()), -1, 1);
    auto input_mem = engine.allocate_memory({ target_pshape, data_types::f16, format::bfyx });
    set_values(input_mem, input_data);

    auto weights_data = rg.generate_random_1d<ov::float16>(weights_layout.count(), -1, 1);
    auto weights_mem = engine.allocate_memory(weights_layout);
    set_values(weights_mem, weights_data);

    auto input1 = input_layout("input1", in_layout);
    auto input2 = input_layout("input2", in_layout);
    auto weights = data("weights", weights_mem);
    auto eltw = eltwise("eltwise", {input_info("input1"), input_info("input2")}, eltwise_mode::sum);
    auto eltw_reorder = reorder("reorder1", input_info("eltwise"), format::b_fs_yx_fsv16, data_types::f16 );
    auto conv = convolution("conv",
                            input_info("reorder1"),
                            "weights",
                            no_bias,
                            1,
                            ov::Strides{1, 1},
                            ov::Strides{1, 1},
                            ov::CoordinateDiff{0, 0},
                            ov::CoordinateDiff{0, 0},
                            false);
    auto output_reorder = reorder("reorder", input_info("conv"), format::bfyx, data_types::f32 );

    topology t(input1, input2, weights, eltw, eltw_reorder, conv, output_reorder);

    ExecutionConfig config_test_blocked = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv_impl_test_blocked = { format::b_fs_yx_fsv16, "", impl_types::onednn };
    config_test_blocked.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ { "conv", conv_impl_test_blocked } }));
    config_test_blocked.set_property(ov::intel_gpu::optimize_data(true));
    config_test_blocked.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network_test_blocked(engine, t, config_test_blocked);

    network_test_blocked.set_input_data("input1", input_mem);
    network_test_blocked.set_input_data("input2", input_mem);

    auto outputs_test_blocked = network_test_blocked.execute();

    // Additional reorder is added and fused when force_implemenetations enable in dynamic
    auto target_primitive_id = p.is_dynamic ? "reorder1_0_reorder_2" : "reorder1";
    auto reorder_inst = network_test_blocked.get_primitive(target_primitive_id);
    ASSERT_TRUE(PrimitiveInstTestHelper::need_reset_output_memory(reorder_inst) == p.need_reset);
}

INSTANTIATE_TEST_SUITE_P(smoke, mem_reset_test,
    testing::Values(
        // static
        mem_reset_params{ 9, false, true },        // If tensor is not packed(not aligned to 16), need_reset_output_memory == true
        mem_reset_params{ 16, false, false },
        // dynamic
        mem_reset_params{ 9, true, true },
        mem_reset_params{ 16, true, false }
    )
);
