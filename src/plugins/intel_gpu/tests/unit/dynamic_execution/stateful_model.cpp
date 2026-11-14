// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/kv_cache.hpp>
#include <intel_gpu/primitives/read_value.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/runtime/internal_properties.hpp>
#include <limits>

#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/runtime/shape_predictor.hpp"
#include "program_wrapper.h"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace stateful_model_tests {
TEST(stateful_model, skip_gather_at_runtime) {
    auto& engine = get_test_engine();

    auto input_kv_lay = layout{ov::PartialShape{-1, 32, -1, 128}, data_types::f32, format::bfyx};
    auto input_present_lay = layout{ov::PartialShape{-1, 32, -1, 128}, data_types::f32, format::bfyx};
    auto input_beam_idx_lay = layout{ov::PartialShape{-1}, data_types::i32, format::bfyx};

    topology topology(input_layout("kv_cache", input_kv_lay),
                      input_layout("beam_idx", input_beam_idx_lay),
                      input_layout("present", input_present_lay),
                      gather("gather",
                             input_info("kv_cache"),
                             input_info("beam_idx"),
                             0,                                       // axis
                             input_kv_lay.get_partial_shape().size(), // input rank
                             ov::Shape{},                             // output shape
                             0,                                       // batch_dim
                             true),                                   // support_neg_ind
                      concatenation("concat", {input_info("gather"), input_info("present")}, 0),
                      reorder("reorder", input_info("concat"), format::bfyx, data_types::f32)); /*output padding*/

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    auto gather_inst = network.get_primitive("gather");
    ASSERT_EQ(gather_inst->get_node().can_be_optimized(), true);
    ASSERT_EQ(gather_inst->can_be_optimized(), true);

    auto KV_SIZE = 24;
    auto BATCH_SIZE = 1;
    auto kv_cache_mem = engine.allocate_memory({{KV_SIZE, 32, BATCH_SIZE, 128}, data_types::f32, format::bfyx});
    auto present_mem = engine.allocate_memory({{1, 32, BATCH_SIZE, 128}, data_types::f32, format::bfyx});
    auto beam_idx_mem = engine.allocate_memory({{KV_SIZE}, data_types::i32, format::bfyx});
    std::vector<float> kv_input_data(kv_cache_mem->get_layout().count());
    std::vector<float> present_input_data(present_mem->get_layout().count());
    std::vector<int32_t> beam_idx_input_data(beam_idx_mem->get_layout().count());
    std::iota(kv_input_data.begin(), kv_input_data.end(), 0.f);
    std::iota(present_input_data.begin(), present_input_data.end(), 0.f);
    std::iota(beam_idx_input_data.begin(), beam_idx_input_data.end(), 0);
    set_values(kv_cache_mem, kv_input_data);
    set_values(present_mem, present_input_data);
    set_values(beam_idx_mem, beam_idx_input_data);

    network.set_input_data("kv_cache", kv_cache_mem);
    network.set_input_data("present", present_mem);
    network.set_input_data("beam_idx", beam_idx_mem);
    network.execute();
    ASSERT_EQ(gather_inst->can_be_optimized(), true);
    auto gather_output_mem = network.get_output_memory("gather");
    cldnn::mem_lock<float, mem_lock_type::read> gather_output_ptr(gather_output_mem, get_test_stream());
    for (size_t i = 0; i < gather_output_mem->get_layout().count(); ++i) {
        ASSERT_EQ(gather_output_ptr[i], kv_input_data[i]);
    }
}

TEST(stateful_model, not_skip_gather_at_runtime) {
    auto& engine = get_test_engine();

    auto input_kv_lay = layout{ov::PartialShape{-1, 32, -1, 128}, data_types::f32, format::bfyx};
    auto input_present_lay = layout{ov::PartialShape{-1, 32, -1, 128}, data_types::f32, format::bfyx};
    auto input_beam_idx_lay = layout{ov::PartialShape{-1}, data_types::i32, format::bfyx};

    topology topology(input_layout("kv_cache", input_kv_lay),
                      input_layout("beam_idx", input_beam_idx_lay),
                      input_layout("present", input_present_lay),
                      gather("gather",
                             input_info("kv_cache"),
                             input_info("beam_idx"),
                             0,                                       // axis
                             input_kv_lay.get_partial_shape().size(), // input rank
                             ov::Shape{},                             // output shape
                             0,                                       // batch_dim
                             true),                                   // support_neg_ind
                      concatenation("concat", {input_info("gather"), input_info("present")}, 0),
                      reorder("reorder", input_info("concat"), format::bfyx, data_types::f32)); /*output padding*/

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    auto gather_inst = network.get_primitive("gather");
    ASSERT_EQ(gather_inst->get_node().can_be_optimized(), true);
    ASSERT_EQ(gather_inst->can_be_optimized(), true);

    auto KV_SIZE = 24;
    auto BATCH_SIZE = 1;
    auto kv_cache_mem = engine.allocate_memory({{KV_SIZE, 32, BATCH_SIZE, 128}, data_types::f32, format::bfyx});
    auto present_mem = engine.allocate_memory({{1, 32, BATCH_SIZE, 128}, data_types::f32, format::bfyx});
    auto beam_idx_mem = engine.allocate_memory({{KV_SIZE}, data_types::i32, format::bfyx});
    std::vector<float> kv_input_data(kv_cache_mem->get_layout().count());
    std::vector<float> present_input_data(present_mem->get_layout().count());
    std::vector<int32_t> beam_idx_input_data(beam_idx_mem->get_layout().count());
    std::iota(kv_input_data.begin(), kv_input_data.end(), 0.f);
    std::iota(present_input_data.begin(), present_input_data.end(), 0.f);
    std::iota(beam_idx_input_data.begin(), beam_idx_input_data.end(), 0);
    std::swap(beam_idx_input_data[0], beam_idx_input_data[1]);
    set_values(kv_cache_mem, kv_input_data);
    set_values(present_mem, present_input_data);
    set_values(beam_idx_mem, beam_idx_input_data);

    network.set_input_data("kv_cache", kv_cache_mem);
    network.set_input_data("present", present_mem);
    network.set_input_data("beam_idx", beam_idx_mem);
    network.execute();
    ASSERT_EQ(gather_inst->can_be_optimized(), false);
}

TEST(stateful_model, not_skip_gather_in_cpuimpl) {
    auto& engine = get_test_engine();

    auto input_kv_lay = layout{ov::PartialShape{-1, 32, -1, 128}, data_types::f32, format::bfyx};
    auto input_present_lay = layout{ov::PartialShape{-1, 32, -1, 128}, data_types::f32, format::bfyx};
    auto input_beam_idx_lay = layout{ov::PartialShape{-1}, data_types::i32, format::bfyx};

    topology topology(input_layout("kv_cache", input_kv_lay),
                      input_layout("beam_idx", input_beam_idx_lay),
                      input_layout("present", input_present_lay),
                      gather("gather",
                             input_info("kv_cache"),
                             input_info("beam_idx"),
                             0,                                       // axis
                             input_kv_lay.get_partial_shape().size(), // input rank
                             ov::Shape{},                             // output shape
                             0,                                       // batch_dim
                             true),                                   // support_neg_ind
                      concatenation("concat", {input_info("gather"), input_info("present")}, 0),
                      reorder("reorder", input_info("concat"), format::bfyx, data_types::f32)); /*output padding*/

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"gather", {format::bfyx, "", impl_types::cpu}} }));

    network network(engine, topology, config);
    auto gather_inst = network.get_primitive("gather");
    ASSERT_EQ(gather_inst->get_node().can_be_optimized(), true);
    ASSERT_EQ(gather_inst->can_be_optimized(), true);

    auto KV_SIZE = 24;
    auto BATCH_SIZE = 1;
    auto kv_cache_mem = engine.allocate_memory({{KV_SIZE, 32, BATCH_SIZE, 128}, data_types::f32, format::bfyx});
    auto present_mem = engine.allocate_memory({{1, 32, BATCH_SIZE, 128}, data_types::f32, format::bfyx});
    auto beam_idx_mem = engine.allocate_memory({{KV_SIZE}, data_types::i32, format::bfyx});
    std::vector<float> kv_input_data(kv_cache_mem->get_layout().count());
    std::vector<float> present_input_data(present_mem->get_layout().count());
    std::vector<int32_t> beam_idx_input_data(beam_idx_mem->get_layout().count());
    std::iota(kv_input_data.begin(), kv_input_data.end(), 0.f);
    std::iota(present_input_data.begin(), present_input_data.end(), 0.f);
    std::iota(beam_idx_input_data.begin(), beam_idx_input_data.end(), 0);
    set_values(kv_cache_mem, kv_input_data);
    set_values(present_mem, present_input_data);
    set_values(beam_idx_mem, beam_idx_input_data);

    network.set_input_data("kv_cache", kv_cache_mem);
    network.set_input_data("present", present_mem);
    network.set_input_data("beam_idx", beam_idx_mem);
    network.execute();
    ASSERT_EQ(gather_inst->can_be_optimized(), true);
    auto gather_output_mem = network.get_output_memory("gather");
    cldnn::mem_lock<float, mem_lock_type::read> gather_output_ptr(gather_output_mem, get_test_stream());
    for (size_t i = 0; i < gather_output_mem->get_layout().count(); ++i) {
        ASSERT_EQ(gather_output_ptr[i], kv_input_data[i]);
    }
}

TEST(stateful_model, check_dynamic_pad_for_kv_cache) {
    auto& engine = get_test_engine();

    auto input_beam_idx_lay = layout{ov::PartialShape{-1}, data_types::i32, format::bfyx};
    auto input_present_lay = layout{ov::PartialShape{-1, 32, -1, 128}, data_types::f32, format::bfyx};

    ov::op::util::VariableInfo info{ov::PartialShape{-1, 32, -1, 128}, data_types::f32, "v0"};
    auto input_kv_lay = layout{info.data_shape, info.data_type, format::bfyx};
    topology topology(input_layout("beam_idx", input_beam_idx_lay),
                      input_layout("present", input_present_lay),
                      read_value("kv_cache", std::vector<input_info>{}, info.variable_id, {input_kv_lay}),
                      gather("gather",
                             input_info("kv_cache"),
                             input_info("beam_idx"),
                             0,                                       // axis
                             input_kv_lay.get_partial_shape().size(), // input rank
                             ov::Shape{},                             // output shape
                             0,                                       // batch_dim
                             true),                                   // support_neg_ind
                      kv_cache("concat", {input_info("gather"), input_info("present")}, info, 0, 0, false, false, false),
                      reorder("reorder", input_info("concat"), format::bfyx, data_types::f32)); /*output padding*/

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    auto gather_inst = network.get_primitive("gather");
    auto read_value_inst = network.get_primitive("kv_cache");
    auto kv_cache_inst = network.get_primitive("concat");

    ASSERT_EQ(gather_inst->get_node().can_be_optimized(), true);
    ASSERT_EQ(gather_inst->can_be_optimized(), true);

    auto pad = tensor(0);
    pad.batch[0] = 1;


    {
        std::vector<tensor::value_type> dynamic_pad_mask;
        const auto& dynamic_pad_dims = read_value_inst->get_output_layout(0).data_padding._dynamic_dims_mask;
        for (size_t i = 0; i < dynamic_pad_dims.size(); i++)
            dynamic_pad_mask.push_back(dynamic_pad_dims[i]);
        ASSERT_EQ(tensor(dynamic_pad_mask, 0), pad);
    }
    {
        std::vector<tensor::value_type> dynamic_pad_mask;
        const auto& dynamic_pad_dims = gather_inst->get_output_layout(0).data_padding._dynamic_dims_mask;
        for (size_t i = 0; i < dynamic_pad_dims.size(); i++)
            dynamic_pad_mask.push_back(dynamic_pad_dims[i]);
        ASSERT_EQ(tensor(dynamic_pad_mask, 0), pad);
    }
    {
        std::vector<tensor::value_type> dynamic_pad_mask;
        const auto& dynamic_pad_dims = kv_cache_inst->get_output_layout(0).data_padding._dynamic_dims_mask;
        for (size_t i = 0; i < dynamic_pad_dims.size(); i++)
            dynamic_pad_mask.push_back(dynamic_pad_dims[i]);
        ASSERT_EQ(tensor(dynamic_pad_mask, 0), pad);
    }
}

TEST(stateful_model, kv_cache_release) {
    auto context = std::make_shared<ov::intel_gpu::RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{get_test_engine().get_device()});
    auto& engine = context->get_engine();

    auto input_lay = layout{ov::PartialShape{1, 32, 1, 128}, data_types::f32, format::bfyx};
    const auto past_data = engine.allocate_memory(input_lay);
    const auto present_data = engine.allocate_memory(input_lay);
    set_random_values<float>(past_data);
    set_random_values<float>(present_data);

    ov::op::util::VariableInfo info{ov::PartialShape{-1, 32, -1, 128}, data_types::f32, "v0"};
    auto input_kv_lay = layout{info.data_shape, info.data_type, format::bfyx};
    topology topology(input_layout("past", input_lay),
                      input_layout("present", input_lay),
                      read_value("kv_cache", {input_info("past")}, info.variable_id, {input_kv_lay}),
                      kv_cache("concat", {input_info("kv_cache"), input_info("present")}, info, 0, 0, false, false, false),
                      reorder("reorder", input_info("concat"), format::bfyx, data_types::f32)); /*output padding*/

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);

    auto read_value_inst = network->get_primitive("kv_cache");
    auto kv_cache_inst = network->get_primitive("concat");
    
    ov::intel_gpu::VariableStateInfo var_info{info.variable_id, input_lay};
    var_info.m_release_variable_inst.emplace_back(std::dynamic_pointer_cast<memory_state::releasable_variable>(read_value_inst));
    var_info.m_release_variable_inst.emplace_back(std::dynamic_pointer_cast<memory_state::releasable_variable>(kv_cache_inst));
    auto variable = std::make_shared<ov::intel_gpu::VariableState>(var_info, context, network->get_shape_predictor());
    network->set_variable("v0", variable);
    network->set_input_data("past", past_data);
    network->set_input_data("present", present_data);

    const auto outputs = network->execute();

    const auto rv_output_count = read_value_inst->outputs_memory_count();
    const auto kv_output_count = read_value_inst->outputs_memory_count();
    for (size_t i = 0; i < rv_output_count; ++i) {
        ASSERT_TRUE(read_value_inst->output_memory_ptr(i));
    }
    for (size_t i = 0; i < kv_output_count; ++i) {
        ASSERT_TRUE(kv_cache_inst->output_memory_ptr(i));
    }

    variable->reset();
    for (size_t i = 0; i < rv_output_count; ++i) {
        ASSERT_FALSE(read_value_inst->output_memory_ptr(i));
    }
    for (size_t i = 0; i < kv_output_count; ++i) {
        ASSERT_FALSE(kv_cache_inst->output_memory_ptr(i));
    }
}

TEST(stateful_model, kv_cache_state_bound_to_output_without_preallocation) {
    auto context = std::make_shared<ov::intel_gpu::RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{get_test_engine().get_device()});
    auto& engine = context->get_engine();
    // Report the device as almost full, so ShapePredictor::can_preallocate() rejects the kv_cache output preallocation
    // The 64 KiB slack keeps the fake usage above the threshold's float rounding margin (<6 KiB for device sizes up to 128 GiB),
    // so preallocation is rejected for any request size on any device.
    engine.add_memory_used(static_cast<uint64_t>(engine.get_device_info().max_global_mem_size * 0.9) + 64 * 1024, allocation_type::usm_device);

    const ov::Shape token_shape{1, 2, 1, 16};  // [batch, heads, seq, head_size]
    const size_t heads = token_shape[1];
    const size_t head_size = token_shape[3];
    const auto token_lay = layout{token_shape, data_types::f32, format::bfyx};
    ov::PartialShape kv_shape = token_shape;
    kv_shape[0] = ov::Dimension::dynamic();
    kv_shape[2] = ov::Dimension::dynamic();

    ov::op::util::VariableInfo info{kv_shape, data_types::f32, "v0"};
    topology topology(input_layout("past", token_lay),
                      input_layout("present", token_lay),
                      read_value("kv_cache", {input_info("past")}, info.variable_id, {layout{kv_shape, data_types::f32, format::bfyx}}),
                      kv_cache("concat", {input_info("kv_cache"), input_info("present")}, info, 2, 0, false, false, false),
                      reorder("reorder", input_info("concat"), format::bfyx, data_types::f32));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    ov::intel_gpu::VariableStateInfo var_info{info.variable_id, token_lay};
    network->set_variable(info.variable_id, std::make_shared<ov::intel_gpu::VariableState>(var_info, context, network->get_shape_predictor()));
    // Pin the precondition so the test fails loudly instead of silently running the padded branch it no longer covers.
    ASSERT_FALSE(network->get_shape_predictor()->can_preallocate(1));

    const auto make_token = [&](float first_value) {
        std::vector<float> values(token_lay.count());
        std::iota(values.begin(), values.end(), first_value);
        auto mem = engine.allocate_memory(token_lay);
        set_values(mem, values);
        return std::make_pair(mem, values);
    };

    auto [past_mem, expected] = make_token(0.f);
    network->set_input_data("past", past_mem);
    size_t seq_len = 1;
    for (size_t iter = 1; iter <= 2; ++iter) {
        auto [present_mem, present] = make_token(1000.f * iter);
        network->set_input_data("present", present_mem);
        auto outputs = network->execute();

        std::vector<float> concatenated;
        for (size_t h = 0; h < heads; ++h) {
            concatenated.insert(concatenated.end(), expected.begin() + h * seq_len * head_size, expected.begin() + (h + 1) * seq_len * head_size);
            concatenated.insert(concatenated.end(), present.begin() + h * head_size, present.begin() + (h + 1) * head_size);
        }
        expected = std::move(concatenated);
        ++seq_len;

        auto output_mem = outputs.at("reorder").get_memory();
        ASSERT_EQ(output_mem->get_layout().get_shape(), (ov::Shape{1, heads, seq_len, head_size}));
        cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output_mem, network->get_stream());
        for (size_t i = 0; i < expected.size(); ++i) {
            ASSERT_EQ(output_ptr[i], expected[i]) << "iteration " << iter << ", index " << i;
        }
    }
}

// Compressed (asymmetric planar) kv-cache binds three states. In the degraded branch of realloc_outputs() every
// state must stay bound to its own kv_cache output buffer, which requires the zero-point state to keep the zp
// output data type: a scales-typed (f16) layout is twice the zp buffer size, so the state would be split into a
// separate 2x allocation and the output->state copy would read twice the zp output buffer on every iteration.
TEST(stateful_model, kv_cache_compressed_state_bound_to_output_without_preallocation) {
    if (!get_test_engine().get_device_info().supports_immad) {
        GTEST_SKIP() << "planar zero-point compressed kv-cache is only produced for supports_immad devices";
    }
    auto context = std::make_shared<ov::intel_gpu::RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{get_test_engine().get_device()});
    auto& engine = context->get_engine();
    engine.add_memory_used(static_cast<uint64_t>(engine.get_device_info().max_global_mem_size * 0.9) + 64 * 1024, allocation_type::usm_device);

    const ov::Shape token_shape{1, 2, 1, 16};
    const ov::Shape past0_shape{1, 2, 0, 16};
    const ov::Shape scales0_shape{1, 2, 0, 1};
    const auto present_lay = layout{token_shape, data_types::f16, format::bfyx};
    const auto past_lay = layout{past0_shape, data_types::i8, format::bfyx};
    const auto scales_lay = layout{scales0_shape, data_types::f16, format::bfyx};
    const auto zp_lay = layout{scales0_shape, data_types::i8, format::bfyx};
    const auto beam_idx_lay = layout{ov::Shape{1}, data_types::i32, format::bfyx};

    ov::PartialShape kv_shape = token_shape;
    kv_shape[0] = ov::Dimension::dynamic();
    kv_shape[2] = ov::Dimension::dynamic();
    ov::PartialShape scales_pshape = token_shape;
    scales_pshape[0] = ov::Dimension::dynamic();
    scales_pshape[2] = ov::Dimension::dynamic();
    scales_pshape[3] = 1;

    ov::op::util::VariableInfo info{kv_shape, data_types::i8, "v0"};

    using DQ = ov::op::internal::DynamicQuantize;
    DQ::Attributes attrs;
    attrs.group_sizes = std::vector<uint64_t>{1, 1, 1, std::numeric_limits<uint64_t>::max()};
    attrs.scales_zp_output_order = std::vector<uint64_t>{0, 1, 2, 3};
    attrs.quantization_type = DQ::QuantizationType::Asymmetric;
    attrs.output_storage_type = DQ::OutputStorageType::Planar;
    attrs.quantization_dt = ov::element::i8;
    attrs.scale_dt = ov::element::f16;
    attrs.zp_dt = ov::element::i8;

    auto kv_cache_prim =
        cldnn::kv_cache("kv_cache",
                        {input_info("read_value"), input_info("present"), input_info("beam_idx"), input_info("read_value", 1), input_info("read_value", 2)},
                        info,
                        2,
                        0,
                        true /*indirect*/,
                        false /*trim*/,
                        false /*update_kv*/);
    kv_cache_prim.compressed = true;
    kv_cache_prim.quantization_attributes = attrs;
    kv_cache_prim.num_outputs = 4;
    kv_cache_prim.output_data_types.resize(4);

    // State layouts must stay dynamic like in production: static layouts skip shape_info entries without advancing
    // the write offset, which shifts all later records and corrupts kernel addressing.
    const layout kv_dyn_lay{kv_shape, data_types::i8, format::bfyx};
    const layout scales_dyn_lay{scales_pshape, data_types::f16, format::bfyx};
    const layout zp_dyn_lay{scales_pshape, data_types::i8, format::bfyx};

    topology topology(input_layout("past", past_lay),
                      input_layout("present", present_lay),
                      input_layout("beam_idx", beam_idx_lay),
                      input_layout("past_scales", scales_lay),
                      input_layout("past_zp", zp_lay),
                      read_value("read_value",
                                 {input_info("past"), input_info("past_scales"), input_info("past_zp")},
                                 info.variable_id,
                                 {kv_dyn_lay, scales_dyn_lay, zp_dyn_lay}),
                      kv_cache_prim,
                      reorder("out_kv", input_info("kv_cache"), format::bfyx, data_types::i8),
                      reorder("out_scales", input_info("kv_cache", 2), format::bfyx, data_types::f16),
                      reorder("out_zp", input_info("kv_cache", 3), format::bfyx, data_types::i8));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);
    ASSERT_FALSE(network->get_shape_predictor()->can_preallocate(1));

    ov::intel_gpu::VariableStateInfo var_info{info.variable_id, kv_dyn_lay};
    auto compressed_variable =
        std::make_shared<ov::intel_gpu::VariableStateIndirectKVCacheCompressed>(var_info,
                                                                                context,
                                                                                network->get_shape_predictor(),
                                                                                std::vector<cldnn::layout>{kv_dyn_lay, scales_dyn_lay, zp_dyn_lay},
                                                                                0 /*beam_idx*/,
                                                                                2 /*concat_idx*/,
                                                                                true /*has_zp_state*/);
    network->set_variable(info.variable_id, compressed_variable);

    network->set_input_data("past", engine.allocate_memory(past_lay));
    network->set_input_data("past_scales", engine.allocate_memory(scales_lay));
    network->set_input_data("past_zp", engine.allocate_memory(zp_lay));
    std::vector<int32_t> beam_idx_values{0};
    auto beam_idx_mem = engine.allocate_memory(beam_idx_lay);
    set_values(beam_idx_mem, beam_idx_values);
    network->set_input_data("beam_idx", beam_idx_mem);

    std::vector<ov::float16> present_values(present_lay.count());
    for (size_t i = 0; i < present_values.size(); ++i) {
        present_values[i] = ov::float16{static_cast<float>(i)};
    }
    auto present_mem = engine.allocate_memory(present_lay);
    set_values(present_mem, present_values);

    auto* kv_inst = network->get_primitive("kv_cache").get();
    std::vector<int8_t> prev_kv;
    std::vector<int8_t> prev_zp;
    const size_t heads = token_shape[1];
    const size_t head_kv = token_shape[3];
    const size_t head_zp = scales0_shape[3];
    // Per-iteration inputs must differ non-affinely and cross zero: quantized codes are invariant to affine
    // transforms of a row, so e.g. a plain ramp quantizes identically every iteration and lost or swapped rows
    // would stay hidden. Here every (head, iteration) row yields distinct quantized data.
    for (size_t iter = 1; iter <= 2; ++iter) {
        for (size_t i = 0; i < present_values.size(); ++i) {
            present_values[i] = ov::float16{static_cast<float>(((i * 37 + iter * 101) % 64) - 32)};
        }
        set_values(present_mem, present_values);
        network->set_input_data("present", present_mem);
        auto outputs = network->execute();

        auto kv_mem = outputs.at("out_kv").get_memory();
        auto zp_mem = outputs.at("out_zp").get_memory();
        ASSERT_EQ(kv_mem->get_layout().get_shape(), (ov::Shape{1, heads, iter, head_kv}));
        ASSERT_EQ(zp_mem->get_layout().get_shape(), (ov::Shape{1, heads, iter, head_zp}));

        std::vector<int8_t> cur_kv(kv_mem->get_layout().count());
        std::vector<int8_t> cur_zp(zp_mem->get_layout().count());
        {
            cldnn::mem_lock<int8_t, mem_lock_type::read> out(kv_mem, network->get_stream());
            for (size_t i = 0; i < cur_kv.size(); ++i) {
                cur_kv[i] = out[i];
            }
        }
        {
            cldnn::mem_lock<int8_t, mem_lock_type::read> out(zp_mem, network->get_stream());
            for (size_t i = 0; i < cur_zp.size(); ++i) {
                cur_zp[i] = out[i];
            }
        }
        // bfyx row growth shifts the start of every following head segment, so compare the previous iteration's
        // rows head by head at the grown strides instead of as a flat prefix; per row the appended token is the
        // last one, so every earlier position must still hold the previous iteration's value.
        for (size_t f = 0; f < heads; ++f) {
            const auto prev_kv_head = prev_kv.begin() + static_cast<ptrdiff_t>(f) * static_cast<ptrdiff_t>(iter - 1) * static_cast<ptrdiff_t>(head_kv);
            const auto cur_kv_head = cur_kv.begin() + static_cast<ptrdiff_t>(f) * static_cast<ptrdiff_t>(iter) * static_cast<ptrdiff_t>(head_kv);
            EXPECT_TRUE(std::equal(prev_kv_head, prev_kv_head + (iter - 1) * head_kv, cur_kv_head))
                << "iteration " << iter << " head " << f << " kv round-trip";
            const auto prev_zp_head = prev_zp.begin() + static_cast<ptrdiff_t>(f) * static_cast<ptrdiff_t>(iter - 1) * static_cast<ptrdiff_t>(head_zp);
            const auto cur_zp_head = cur_zp.begin() + static_cast<ptrdiff_t>(f) * static_cast<ptrdiff_t>(iter) * static_cast<ptrdiff_t>(head_zp);
            EXPECT_TRUE(std::equal(prev_zp_head, prev_zp_head + (iter - 1) * head_zp, cur_zp_head))
                << "iteration " << iter << " head " << f << " zp round-trip";
        }
        // Non-vacuity: outputs must carry data, not a zero- or constant-filled buffer, otherwise any
        // "previous rows are preserved" comparison would be satisfied trivially.
        EXPECT_NE(std::count_if(cur_kv.begin(),
                                cur_kv.end(),
                                [](int8_t v) {
                                    return v != 0;
                                }),
                  0)
            << "iteration " << iter << " kv output must not be zero-filled";
        EXPECT_NE(std::count(cur_kv.begin(), cur_kv.end(), cur_kv.front()), static_cast<long>(cur_kv.size()))
            << "iteration " << iter << " kv output must not be constant";
        EXPECT_NE(std::count(cur_zp.begin(), cur_zp.end(), int8_t{0}), static_cast<long>(cur_zp.size()))
            << "iteration " << iter << " zp output must not be zero-filled";
        prev_kv = std::move(cur_kv);
        prev_zp = std::move(cur_zp);

        // Branch 2 contract: every state stays bound to its own output buffer with its own element type.
        auto* cv = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&network->get_variable(info.variable_id));
        ASSERT_NE(cv, nullptr);
        auto zp_state = cv->get_compression_zp_state();
        auto scale_state = cv->get_compression_scale_state();
        EXPECT_EQ(zp_state->get_layout().data_type, data_types::i8) << "iteration " << iter;
        EXPECT_EQ(scale_state->get_layout().data_type, data_types::f16) << "iteration " << iter;
        EXPECT_TRUE(engine.is_the_same_buffer(*zp_state->get_memory(), *kv_inst->output_memory_ptr(3))) << "iteration " << iter << " zp alias";
        EXPECT_TRUE(engine.is_the_same_buffer(*scale_state->get_memory(), *kv_inst->output_memory_ptr(2))) << "iteration " << iter << " scale alias";
        EXPECT_TRUE(engine.is_the_same_buffer(*cv->get_memory(), *kv_inst->output_memory_ptr(0))) << "iteration " << iter << " kv alias";
    }
}

// Branch 3 of realloc_outputs() runs for dynamic kv_cache nodes whose sequence axis was never marked dynamic
// (e.g. optimize_data(false), or prepare_buffer_fusing early-return). States keep their own buffers and are
// refreshed by per-iteration output->state copies; the zp state layout must still carry the zp output type,
// otherwise the state is reinterpreted as scales-sized (2N) data and the oversized copy reads past the zp output.
TEST(stateful_model, kv_cache_compressed_state_layout_keeps_zp_type_without_dynamic_mask) {
    if (!get_test_engine().get_device_info().supports_immad) {
        GTEST_SKIP() << "planar zero-point compressed kv-cache is only produced for supports_immad devices";
    }
    auto context = std::make_shared<ov::intel_gpu::RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{get_test_engine().get_device()});
    auto& engine = context->get_engine();
    // Reject preallocation so layouts are exact-sized; branch selection itself only depends on the missing dynamic mask.
    engine.add_memory_used(static_cast<uint64_t>(engine.get_device_info().max_global_mem_size * 0.9) + 64 * 1024, allocation_type::usm_device);

    const ov::Shape token_shape{1, 2, 1, 16};
    const ov::Shape past0_shape{1, 2, 0, 16};
    const ov::Shape scales0_shape{1, 2, 0, 1};
    const auto present_lay = layout{token_shape, data_types::f16, format::bfyx};
    const auto past_lay = layout{past0_shape, data_types::i8, format::bfyx};
    const auto scales_lay = layout{scales0_shape, data_types::f16, format::bfyx};
    const auto zp_lay = layout{scales0_shape, data_types::i8, format::bfyx};
    const auto beam_idx_lay = layout{ov::Shape{1}, data_types::i32, format::bfyx};

    ov::PartialShape kv_shape = token_shape;
    kv_shape[0] = ov::Dimension::dynamic();
    kv_shape[2] = ov::Dimension::dynamic();
    ov::PartialShape scales_pshape = token_shape;
    scales_pshape[0] = ov::Dimension::dynamic();
    scales_pshape[2] = ov::Dimension::dynamic();
    scales_pshape[3] = 1;

    ov::op::util::VariableInfo info{kv_shape, data_types::i8, "v0"};

    using DQ = ov::op::internal::DynamicQuantize;
    DQ::Attributes attrs;
    attrs.group_sizes = std::vector<uint64_t>{1, 1, 1, std::numeric_limits<uint64_t>::max()};
    attrs.scales_zp_output_order = std::vector<uint64_t>{0, 1, 2, 3};
    attrs.quantization_type = DQ::QuantizationType::Asymmetric;
    attrs.output_storage_type = DQ::OutputStorageType::Planar;
    attrs.quantization_dt = ov::element::i8;
    attrs.scale_dt = ov::element::f16;
    attrs.zp_dt = ov::element::i8;

    auto kv_cache_prim =
        cldnn::kv_cache("kv_cache",
                        {input_info("read_value"), input_info("present"), input_info("beam_idx"), input_info("read_value", 1), input_info("read_value", 2)},
                        info,
                        2,
                        0,
                        true /*indirect*/,
                        false /*trim*/,
                        false /*update_kv*/);
    kv_cache_prim.compressed = true;
    kv_cache_prim.quantization_attributes = attrs;
    kv_cache_prim.num_outputs = 4;
    kv_cache_prim.output_data_types.resize(4);

    const layout kv_dyn_lay{kv_shape, data_types::i8, format::bfyx};
    const layout scales_dyn_lay{scales_pshape, data_types::f16, format::bfyx};
    const layout zp_dyn_lay{scales_pshape, data_types::i8, format::bfyx};

    topology topology(input_layout("past", past_lay),
                      input_layout("present", present_lay),
                      input_layout("beam_idx", beam_idx_lay),
                      input_layout("past_scales", scales_lay),
                      input_layout("past_zp", zp_lay),
                      read_value("read_value",
                                 {input_info("past"), input_info("past_scales"), input_info("past_zp")},
                                 info.variable_id,
                                 {kv_dyn_lay, scales_dyn_lay, zp_dyn_lay}),
                      kv_cache_prim,
                      reorder("out_kv", input_info("kv_cache"), format::bfyx, data_types::i8),
                      reorder("out_scales", input_info("kv_cache", 2), format::bfyx, data_types::f16),
                      reorder("out_zp", input_info("kv_cache", 3), format::bfyx, data_types::i8));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    // No data optimization: prepare_buffer_fusing never stamps the dynamic pad mask, so realloc_outputs()
    // takes the branch that only calls set_layout() on the states.
    config.set_property(ov::intel_gpu::optimize_data(false));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);

    ov::intel_gpu::VariableStateInfo var_info{info.variable_id, kv_dyn_lay};
    auto compressed_variable =
        std::make_shared<ov::intel_gpu::VariableStateIndirectKVCacheCompressed>(var_info,
                                                                                context,
                                                                                network->get_shape_predictor(),
                                                                                std::vector<cldnn::layout>{kv_dyn_lay, scales_dyn_lay, zp_dyn_lay},
                                                                                0 /*beam_idx*/,
                                                                                2 /*concat_idx*/,
                                                                                true /*has_zp_state*/);
    network->set_variable(info.variable_id, compressed_variable);

    network->set_input_data("past", engine.allocate_memory(past_lay));
    network->set_input_data("past_scales", engine.allocate_memory(scales_lay));
    network->set_input_data("past_zp", engine.allocate_memory(zp_lay));
    std::vector<int32_t> beam_idx_values{0};
    auto beam_idx_mem = engine.allocate_memory(beam_idx_lay);
    set_values(beam_idx_mem, beam_idx_values);
    network->set_input_data("beam_idx", beam_idx_mem);

    std::vector<ov::float16> present_values(present_lay.count());
    for (size_t i = 0; i < present_values.size(); ++i) {
        present_values[i] = ov::float16{static_cast<float>(i)};
    }
    auto present_mem = engine.allocate_memory(present_lay);
    set_values(present_mem, present_values);

    for (size_t iter = 1; iter <= 2; ++iter) {
        network->set_input_data("present", present_mem);
        network->execute();

        auto* cv = dynamic_cast<ov::intel_gpu::VariableStateIndirectKVCacheCompressed*>(&network->get_variable(info.variable_id));
        ASSERT_NE(cv, nullptr);
        auto zp_state = cv->get_compression_zp_state();
        auto scale_state = cv->get_compression_scale_state();
        EXPECT_EQ(zp_state->get_layout().data_type, data_types::i8) << "iteration " << iter;
        EXPECT_EQ(scale_state->get_layout().data_type, data_types::f16) << "iteration " << iter;
        // Branch 3 keeps separate state buffers (no binding); the zp layout must stay exactly the output size.
        EXPECT_EQ(zp_state->get_layout().bytes_count(), sizeof(int8_t) * 2 * iter)
            << "iteration " << iter << " zp layout " << zp_state->get_layout().to_short_string() << " mem_size=" << zp_state->get_memory()->size()
            << " actual_mem=" << zp_state->get_actual_mem_size();
    }
}

}  // namespace stateful_model_tests
