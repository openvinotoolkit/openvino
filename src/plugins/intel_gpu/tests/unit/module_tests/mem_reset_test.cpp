// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "primitive_inst_test_helper.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/resample.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/shape_of.hpp>

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

// The target shape of an optimized out reshape is read on host by its shape inference, so it must not be reset even if
// the users of the reshape need a zero-filled padding. The reset is an asynchronous fill which may run after the CPU
// impl of shape_of has written the shape, and the reshape would then read [0,0,0,0].
TEST(mem_reset_shape_infer_dep_test, optimized_out_reshape_target_shape) {
    auto& engine = get_test_engine();
    tests::random_generator rg(GET_SUITE_NAME);

    const ov::Dimension::value_type channels = 16;
    const ov::Shape target_shape = {1, channels, 8, 8};
    const ov::Shape data_shape = {1, 1, channels, 8, 8};
    layout target_layout{ov::PartialShape{-1, channels, -1, -1}, data_types::f16, format::bfyx};
    layout data_layout{ov::PartialShape{-1, 1, channels, -1, -1}, data_types::f16, format::bfzyx};
    layout weights_layout{ov::PartialShape{16, channels, 3, 3}, data_types::f16, format::bfyx};

    auto target_mem = engine.allocate_memory({target_shape, data_types::f16, format::bfyx});
    auto data_mem = engine.allocate_memory({data_shape, data_types::f16, format::bfzyx});
    auto weights_mem = engine.allocate_memory(weights_layout);
    set_values(target_mem, rg.generate_random_1d<ov::float16>(ov::shape_size(target_shape), -1, 1));
    set_values(data_mem, rg.generate_random_1d<ov::float16>(ov::shape_size(data_shape), -1, 1));
    set_values(weights_mem, rg.generate_random_1d<ov::float16>(weights_layout.count(), -1, 1));

    // reshape(relu(data), shape_of(target)) is optimized out, the dynamic resample is runtime skippable (marked as
    // optimized out until it is executed), and the bfyx ocl convolution needs a zero-padded input.
    topology t(input_layout("target", target_layout),
               input_layout("data", data_layout),
               data("weights", weights_mem),
               shape_of("shape_of", input_info("target"), data_types::i32),
               activation("relu", input_info("data"), activation_func::relu),
               reshape("reshape", input_info("relu"), input_info("shape_of"), false, ov::PartialShape::dynamic(4)),
               resample("resample",
                        input_info("reshape"),
                        std::vector<int64_t>{},
                        std::vector<float>{2.f, 2.f},
                        std::vector<int64_t>{2, 3},
                        {0, 0, 0, 0},
                        {0, 0, 0, 0},
                        0,
                        -0.75f,
                        resample::InterpolateOp::InterpolateMode::NEAREST,
                        resample::InterpolateOp::ShapeCalcMode::SCALES),
               convolution("conv",
                           input_info("resample"),
                           "weights",
                           no_bias,
                           1,
                           ov::Strides{1, 1},
                           ov::Strides{1, 1},
                           ov::CoordinateDiff{1, 1},
                           ov::CoordinateDiff{1, 1},
                           false),
               reorder("output", input_info("conv"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    ov::intel_gpu::ImplementationDesc conv_impl = {format::bfyx, "", impl_types::ocl};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"conv", conv_impl}}));
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network net(engine, t, config);
    net.set_input_data("target", target_mem);
    net.set_input_data("data", data_mem);

    // Before the first execution, as when shape_of is prepared in the first inference: the data input of the reshape
    // needs the reset, its target shape must not get it.
    ASSERT_TRUE(net.get_primitive("reshape")->can_be_optimized());
    ASSERT_TRUE(PrimitiveInstTestHelper::need_reset_output_memory(net.get_primitive("relu")));
    ASSERT_FALSE(PrimitiveInstTestHelper::need_reset_output_memory(net.get_primitive("shape_of")));

    auto outputs = net.execute();
    ASSERT_EQ(outputs.at("output").get_memory()->get_layout().get_shape(), ov::Shape({1, 16, 16, 16}));
}
