// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/graph/program.hpp"
#include "data_inst.h"
#include "gemm_inst.h"
#include "softmax_inst.h"
#include "pass_manager.h"
#include "program_wrapper.h"

#include <memory>
#include <vector>

using namespace cldnn;
using namespace ::tests;

TEST(clamp_fp16_output_test, test_gemm_softmax_simple) {
    auto& engine = get_test_engine();
    ov::Shape in1_shape = { 1, 1, 3, 4 };
    ov::Shape in2_shape = { 1, 4 };
    auto in1_layout = layout{ov::PartialShape::dynamic(in1_shape.size()), data_types::f32, format::bfyx};
    auto in2_layout = layout{ov::PartialShape::dynamic(in2_shape.size()), data_types::f32, format::bfyx};
    auto input1 = engine.allocate_memory(layout{ov::PartialShape(in1_shape), data_types::f32, format::bfyx});
    auto input2 = engine.allocate_memory(layout{ov::PartialShape(in2_shape), data_types::f32, format::bfyx});

    std::vector<float> input1_data = {
        1.f, -2.f, 3.f, -4.f,
        5.f, 6.f, 1.f, 2.f,
        3.f, 3.f, 2.f, -1.f,
    };

    std::vector<float> input2_data = {
        2.f, 5.f, -4.f, -7.f
    };
    set_values(input1, input1_data);
    set_values(input2, input2_data);

    std::vector<float> out_data = {
        0.f, 0.8803f, 0.1192f
    };

    topology topology;
    topology.add(input_layout("input1", in1_layout),
                 input_layout("input2", in2_layout),
                 reorder("input1_fp16", input_info("input1"), format::any, data_types::f16),
                 reorder("input2_fp16", input_info("input2"), format::any, data_types::f16),
                 gemm("gemm", { input_info("input1_fp16"), input_info("input2_fp16") }, data_types::f16, false, true, 1.0f, 0.0f, 4, 2),
                 softmax("softmax", input_info("gemm"), 2),
                 reorder("reorder", input_info("softmax"), format::any, data_types::f32)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("reorder");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("reorder").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    ASSERT_EQ(output_ptr.size(), (uint32_t)3);
    for (uint32_t i = 0; i < out_data.size(); ++i) {
        ASSERT_NEAR(output_ptr[i], out_data[i], 1e-4);
    }
}

TEST(clamp_fp16_output_test, test_gemm_softmax_mult_fused) {
    auto& engine = get_test_engine();
    ov::Shape in1_shape = { 1, 1, 3, 4 };
    ov::Shape in2_shape = { 1, 4 };
    auto in1_layout = layout{ov::PartialShape::dynamic(in1_shape.size()), data_types::f32, format::bfyx};
    auto in2_layout = layout{ov::PartialShape::dynamic(in2_shape.size()), data_types::f32, format::bfyx};
    auto input1 = engine.allocate_memory(layout{ov::PartialShape(in1_shape), data_types::f32, format::bfyx});
    auto input2 = engine.allocate_memory(layout{ov::PartialShape(in2_shape), data_types::f32, format::bfyx});
    auto input_elt = engine.allocate_memory({ov::PartialShape{1, 1, 3, 1}, data_types::f32, format::bfyx});

    std::vector<float> input1_data = {
        1.f, -2.f, 3.f, -4.f,
        5.f, 6.f, 1.f, 2.f,
        3.f, 3.f, 2.f, -1.f,
    };

    std::vector<float> input2_data = {
        2.f, 5.f, -4.f, -7.f
    };

    std::vector<float> elt_data = {
        10.f, -5.5f, -0.05f
    };

    set_values(input1, input1_data);
    set_values(input2, input2_data);
    set_values(input_elt, elt_data);

    std::vector<float> out_data = {
        0.1209f,  0.0269f, 0.8520f
    };

    topology topology;
    topology.add(input_layout("input1", in1_layout),
                 input_layout("input2", in2_layout),
                 data("elt_input", input_elt),
                 reorder("input1_fp16", input_info("input1"), format::any, data_types::f16),
                 reorder("input2_fp16", input_info("input2"), format::any, data_types::f16),
                 reorder("elt_input_fp16", input_info("elt_input"), format::any, data_types::f16),
                 gemm("gemm", { input_info("input1_fp16"), input_info("input2_fp16") }, data_types::f16, false, true, 1.0f, 0.0f, 4, 2),
                 eltwise("eltwise", input_info("gemm"), input_info("elt_input_fp16"), eltwise_mode::sum),
                 softmax("softmax", input_info("eltwise"), 2),
                 reorder("reorder", input_info("softmax"), format::any, data_types::f32)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("reorder");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    auto output = outputs.at("reorder").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    auto prog = network.get_program();
    ASSERT_TRUE(!has_node(*prog, "eltwise"));

    ASSERT_EQ(output_ptr.size(), (uint32_t)3);
    for (uint32_t i = 0; i < out_data.size(); ++i) {
        ASSERT_NEAR(output_ptr[i], out_data[i], 1e-4);
    }
}
