// Copyright (C) 2023-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>

#include "test_utils.h"

#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/mvn.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/reshape.hpp>


#include "eltwise_inst.h"
// #include "fully_connected_inst.h"

using namespace cldnn;
using namespace tests;


TEST(check_hash_value, eltwise_basic) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 2, 2, 2, 2 } });

    auto key_prim_id = "eltwise";
    topology topology;
    topology.add(input_layout("input", input1->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(eltwise(key_prim_id, { input_info("input"), input_info("input2") }, eltwise_mode::sum));

    auto prog = program::build_program(engine, topology, ExecutionConfig{});
    network net(prog, 0);
    const auto  prim_inst = net.get_primitive(key_prim_id);
    const auto  primitve  = prim_inst->desc();
    const auto& prog_node = net.get_program()->get_node(key_prim_id);

    const auto primitive_hash = primitve->hash();
    const auto params_hash = prog_node.get_kernel_impl_params()->hash();

    ASSERT_EQ(primitive_hash, 11385140218618178073UL);
    ASSERT_EQ(params_hash, 10460622021476296271UL);
}

TEST(check_hash_value, fc_basic) {
    auto& engine = get_test_engine();

    const int32_t b = 1, in_f = 128, in_x = 1, in_y = 1, out_f = 65;

    auto input_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { b, in_f, in_y, in_x } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { out_f, in_f, in_y, in_x } });
    auto bias_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, out_f, 1 } });

    const auto key_prim_id = "fc";
    topology topology(
        input_layout("input", input_prim->get_layout()),
        data("weights", weights_prim),
        data("bias", bias_prim),
        fully_connected(key_prim_id, input_info("input"), "weights", "bias")
    );

    auto prog = program::build_program(engine, topology, ExecutionConfig{});
    network net(prog, 0);
    const auto  prim_inst = net.get_primitive(key_prim_id);
    const auto  primitve  = prim_inst->desc();
    const auto& prog_node = net.get_program()->get_node(key_prim_id);

    const auto primitive_hash = primitve->hash();
    const auto params_hash = prog_node.type()->get_fake_aligned_params(*prog_node.get_kernel_impl_params()).hash();

    ASSERT_EQ(primitive_hash, 7881065839556591629UL);
    ASSERT_EQ(params_hash, 12327057149074647711UL);
}

TEST(check_hash_value, gather_basic) {
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfzyx, tensor{ 3, 2, 2, 4, 3} }); // Dictionary
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, tensor{ 3, 2, 1, 3 } }); // Indexes

    int64_t axis = 3;
    int64_t batch_dim = -1;
    bool negative_indexes = true;

    auto key_prim_id = "gather";
    topology topology;
    topology.add(input_layout("InputDictionary", input1->get_layout()));
    topology.add(input_layout("InputText", input2->get_layout()));
    topology.add(
        gather(key_prim_id, input_info("InputDictionary"), input_info("InputText"), axis, ov::Shape{3, 2, 3, 3, 2}, batch_dim, negative_indexes)
    );

    auto prog = program::build_program(engine, topology, ExecutionConfig{});
    network net(prog, 0);
    const auto  prim_inst = net.get_primitive(key_prim_id);
    const auto  primitve  = prim_inst->desc();
    const auto& prog_node = net.get_program()->get_node(key_prim_id);

    const auto primitive_hash = primitve->hash();
    const auto params_hash = prog_node.get_kernel_impl_params()->hash();

    ASSERT_EQ(primitive_hash, 93320679543770233UL);
    ASSERT_EQ(params_hash, 18126277300376770566UL);
}

TEST(check_hash_value, gemm_basic) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 3 } });
    auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 1 } });

    auto key_prim_id = "gemm";
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(input_layout("input2", input2->get_layout()));
    topology.add(crop("crop.1", input_info("input"), { 1, 1, 4, 3 }, { 0, 1, 0, 0 }));
    topology.add(gemm(key_prim_id, { input_info("crop.1"), input_info("input2") }, data_types::f32, false, true));

    auto prog = program::build_program(engine, topology, ExecutionConfig{});
    network net(prog, 0);
    const auto  prim_inst = net.get_primitive(key_prim_id);
    const auto  primitve  = prim_inst->desc();
    const auto& prog_node = net.get_program()->get_node(key_prim_id);

    const auto primitive_hash = primitve->hash();
    const auto params_hash = prog_node.get_kernel_impl_params()->hash();

    ASSERT_EQ(primitive_hash, 8009877756431655269UL);
    ASSERT_EQ(params_hash, 2966249915421110547UL);
}

TEST(check_hash_value, permute_basic) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 2, 2, 3, 2 } });

    auto key_prim_id = "permute";
    topology topology(
        input_layout("input", input->get_layout()),
        permute(key_prim_id, input_info("input"), { 0, 1, 2, 3 }));

    auto prog = program::build_program(engine, topology, ExecutionConfig{});
    network net(prog, 0);
    const auto  prim_inst = net.get_primitive(key_prim_id);
    const auto  primitve  = prim_inst->desc();
    const auto& prog_node = net.get_program()->get_node(key_prim_id);

    const auto primitive_hash = primitve->hash();
    const auto params_hash = prog_node.get_kernel_impl_params()->hash();

    ASSERT_EQ(primitive_hash, 4658575237077439700UL);
    ASSERT_EQ(params_hash, 4319508487906266226UL);
}

TEST(check_hash_value, reorder_basic) {
    auto& engine = get_test_engine();

    const int32_t b_in = 1;
    const int32_t f_in = 8 * 4;
    const int32_t y_in = 4;
    const int32_t x_in = 8 * 2;

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { b_in,f_in,x_in,y_in } });
    layout output_layout(data_types::f32, format::b_fs_yx_fsv16, { b_in,f_in,x_in,y_in });

    auto key_prim_id = "reorder";
    topology topology(
        input_layout("input", input->get_layout()),
        reorder(key_prim_id, input_info("input"), output_layout));

    auto prog = program::build_program(engine, topology, ExecutionConfig{});
    network net(prog, 0);
    const auto  prim_inst = net.get_primitive(key_prim_id);
    const auto  primitve  = prim_inst->desc();
    const auto& prog_node = net.get_program()->get_node(key_prim_id);

    const auto primitive_hash = primitve->hash();
    const auto params_hash = prog_node.get_kernel_impl_params()->hash();

    ASSERT_EQ(primitive_hash, 16293979194373117693UL);
    ASSERT_EQ(params_hash, 1719378641386629286UL);
}

TEST(check_hash_value, reshape_basic) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 2, 2 } });

    auto key_prim_id = "reshape";
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    auto padded_input_layout = input->get_layout();
    padded_input_layout.data_padding = padding();
    topology.add(reorder("reorder", input_info("input"), padded_input_layout));
    topology.add(reshape(key_prim_id, input_info("reorder"), tensor( 1, 1, 4, 1 ), cldnn::reshape::reshape_mode::base, padding({0, 0, 2, 2})));

    auto prog = program::build_program(engine, topology, ExecutionConfig{});
    network net(prog, 0);
    const auto  prim_inst = net.get_primitive(key_prim_id);
    const auto  primitve  = prim_inst->desc();
    const auto& prog_node = net.get_program()->get_node(key_prim_id);

    const auto primitive_hash = primitve->hash();
    const auto params_hash = prog_node.get_kernel_impl_params()->hash();

    ASSERT_EQ(primitive_hash, 1534749073560581535UL);
    ASSERT_EQ(params_hash, 1686780870642992006UL);
}

TEST(check_hash_value, conv_basic) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { 1, 1, 4, 4, 4 } });
    auto weights = engine.allocate_memory({ data_types::f32, format::bfzyx, { 1, 1, 2, 2, 2 } });
    auto biases = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 1, 1, 1 } });

    auto key_prim_id = "convolution";
    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        data("biases", biases),
        convolution(key_prim_id, input_info("input"), { "weights" }, { "biases" }, {1, 1, 1}, {0, 0, 0}, {1, 1, 1}));

    auto prog = program::build_program(engine, topology, ExecutionConfig{});
    network net(prog, 0);
    const auto  prim_inst = net.get_primitive(key_prim_id);
    const auto  primitve  = prim_inst->desc();
    const auto& prog_node = net.get_program()->get_node(key_prim_id);

    const auto primitive_hash = primitve->hash();
    const auto params_hash = prog_node.get_kernel_impl_params()->hash();

    ASSERT_EQ(primitive_hash, 14591385718963138714UL);
    ASSERT_EQ(params_hash, 6876197578014654797UL);
}

TEST(check_hash_value, quantize_basic) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 16, 2, 2}});
    auto input_low = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto input_high = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 16, 1, 1 } });
    auto output_low = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });
    auto output_high = engine.allocate_memory({ data_types::f32,format::bfyx,{ 1, 1, 1, 1 } });

    auto key_prim_id = "quantize";
    topology topology;
    topology.add(
        input_layout("input", input->get_layout()),
        data("input_low", input_low),
        data("input_high", input_high),
        data("output_low", output_low),
        data("output_high", output_high),
        quantize(key_prim_id, input_info("input"), input_info("input_low"), input_info("input_high"), input_info("output_low"), input_info("output_high"), 256, data_types::u8)
    );

    auto prog = program::build_program(engine, topology, ExecutionConfig{});
    network net(prog, 0);
    const auto  prim_inst = net.get_primitive(key_prim_id);
    const auto  primitve  = prim_inst->desc();
    const auto& prog_node = net.get_program()->get_node(key_prim_id);

    const auto primitive_hash = primitve->hash();
    const auto params_hash = prog_node.get_kernel_impl_params()->hash();

    ASSERT_EQ(primitive_hash, 4135863035456568493UL);
    ASSERT_EQ(params_hash, 13898649554943348250UL);
}
