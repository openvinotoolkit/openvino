// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/kv_cache.hpp>
#include <intel_gpu/primitives/read_value.hpp>
#include <intel_gpu/runtime/internal_properties.hpp>

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

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
                      kv_cache("concat", {input_info("gather"), input_info("present")}, info, 0, 0, false),
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

}  // stateful_model_tests
