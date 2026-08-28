// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "primitive_inst_test_helper.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/eltwise.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
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

TEST(mem_reset_test, static_crop_shared_by_onednn_convs_resets_dirty_padding) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    const ov::PartialShape input_shape = {1, 170, 4, 4};
    const ov::PartialShape weights_shape = {16, 85, 1, 1};
    const layout input_layout{input_shape, data_types::f16, format::b_fs_yx_fsv16};
    const layout weights_layout{weights_shape, data_types::f16, format::bfyx};

    auto input_mem = engine.allocate_memory(input_layout);
    auto weights_mem = engine.allocate_memory(weights_layout);
    set_values(input_mem, std::vector<ov::float16>(input_layout.get_linear_size(), ov::float16(1.0f)));
    set_values(weights_mem, std::vector<ov::float16>(weights_layout.count(), ov::float16(1.0f)));

    topology topology(cldnn::input_layout("input", input_layout),
                      data("weights", weights_mem),
                      crop("split_out1", input_info("input"), tensor(1, 85, 4, 4), tensor(0, 85, 0, 0)),
                      convolution("conv0",
                                  input_info("split_out1"),
                                  "weights",
                                  no_bias,
                                  1,
                                  ov::Strides{1, 1},
                                  ov::Strides{1, 1},
                                  ov::CoordinateDiff{0, 0},
                                  ov::CoordinateDiff{0, 0},
                                  false),
                      convolution("conv1",
                                  input_info("split_out1"),
                                  "weights",
                                  no_bias,
                                  1,
                                  ov::Strides{1, 1},
                                  ov::Strides{1, 1},
                                  ov::CoordinateDiff{0, 0},
                                  ov::CoordinateDiff{0, 0},
                                  false),
                      reorder("output0", input_info("conv0"), format::bfyx, data_types::f32),
                      reorder("output1", input_info("conv1"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{
        {"split_out1", {format::b_fs_yx_fsv16, "generic_eltwise_ref", impl_types::ocl}},
        {"conv0", {format::b_fs_yx_fsv16, "", impl_types::onednn}},
        {"conv1", {format::b_fs_yx_fsv16, "", impl_types::onednn}},
    }));
    config.set_property(ov::intel_gpu::enable_memory_pool(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto crop_output = network.get_primitive("split_out1")->output_memory_ptr();
    ASSERT_NE(crop_output, nullptr);
    const auto nan = ov::float16(std::numeric_limits<float>::quiet_NaN());
    set_values(crop_output, std::vector<ov::float16>(crop_output->get_layout().get_linear_size(), nan));

    auto outputs = network.execute();
    for (const auto& output_id : {"output0", "output1"}) {
        auto output_mem = outputs.at(output_id).get_memory();
        cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output_mem, get_test_stream());
        for (const auto value : output_ptr)
            ASSERT_EQ(value, 85.0f);
    }
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
