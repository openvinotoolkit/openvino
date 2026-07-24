// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "openvino/util/file_util.hpp"

#include <intel_gpu/runtime/debug_configuration.hpp>
#include <intel_gpu/primitives/input_layout.hpp>

#include <intel_gpu/primitives/scaled_dot_product_attention.hpp>
#include <intel_gpu/primitives/vl_sdpa.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/data.hpp>
#include "scaled_dot_product_attention_inst.h"
#include "vl_sdpa_inst.h"

#include <cstddef>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>

using namespace cldnn;
using namespace ::tests;
namespace  {
struct vlsdpa_test_params {
    int32_t head_size;
    int32_t num_heads;
    std::vector<int32_t> cu_seqlens;

    vlsdpa_test_params(int h_size, int n_heads, std::vector<int32_t> _cu_seqlens)
        : head_size(h_size), num_heads(n_heads), cu_seqlens(_cu_seqlens) {}
};

struct vlsdpa_gpu_test : public ::testing::TestWithParam<vlsdpa_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void load_input(cldnn::memory::ptr mem, size_t idx) {
        auto shapes = mem->get_layout().get_shape();
        size_t size = ov::shape_size(shapes);
        auto input_data = rg.generate_random_1d<ov::float16>(size, -1.0f, 1.0f);
        set_values(mem, input_data);
    }

    void get_attention_mask(cldnn::memory::ptr mem, std::vector<int32_t> cu_seqlens) {
        const auto shapes = mem->get_layout().get_shape();
        const size_t size = ov::shape_size(shapes);
        const auto hidden_states_size = shapes.back();

        std::vector<ov::float16> attention_mask_data(size, -std::numeric_limits<ov::float16>::infinity());
        for (size_t i = 1; i < cu_seqlens.size(); ++i) {
            size_t start = cu_seqlens[i-1];
            size_t end = cu_seqlens[i];
            for (size_t row = start; row < end; ++row) {
                for (size_t col = start; col < end; ++col) {
                    attention_mask_data[row * hidden_states_size + col] = ov::float16(0.0f);
                }
            }
        }
        set_values(mem, attention_mask_data);
    }

    void get_cu_seqlens(cldnn::memory::ptr mem, std::vector<int32_t> cu_seqlens) {
        set_values(mem, cu_seqlens);
    }

    topology create_ref_topology(layout q_layout, layout k_layout, layout v_layout, layout attn_mask_layout) {
        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", k_layout));
        topo.add(input_layout("v", v_layout));
        topo.add(input_layout("attn", attn_mask_layout));  // "attention_mask"

        auto sdpa_prim = scaled_dot_product_attention("sdpa", {input_info("q"), input_info("k"), input_info("v"), input_info("attn")},
            false, -1, order_q, order_k, order_v, {0, 1, 2}, {}, false);
        topo.add(sdpa_prim);

        topo.add(permute("permute_o", input_info("sdpa"), order_o2));

        topo.add(reorder("result",input_info("permute_o"), format::bfyx, data_types::f16));
        return topo;
    };

    topology create_topology(layout q_layout, layout k_layout, layout v_layout, layout cu_seqlen_layout) {
        topology topo;
        topo.add(input_layout("q", q_layout));
        topo.add(input_layout("k", k_layout));
        topo.add(input_layout("v", v_layout));
        topo.add(input_layout("attn", cu_seqlen_layout));  // "cu_seqlens"

        auto vlsdpa_prim = vl_sdpa("vlsdpa", {input_info("q"), input_info("k"), input_info("v"), input_info("attn")},
            order_q, order_k, order_v, order_o);

        topo.add(vlsdpa_prim);
        topo.add(reorder("result",input_info("vlsdpa"), format::bfyx, data_types::f16));
        return topo;
    }

    // Build a Split(axis=1, num_splits=3) crop chain fanning a fused [L, 3*H, S]
    // parent into named "q"/"k"/"v" slices. Enabling optimize_data on the resulting
    // topology causes prepare_buffer_fusing to keep the crops in-place, so each
    // slice's runtime layout carries a non-zero linear_offset and the parent's row
    // pitch (3 * num_heads * head_size). This is exactly the discontinuous-token-axis
    // input shape that the CM vl_sdpa kernel's token_offset_{q,k,v} / {q,k,v}_token_pitch
    // scalars are designed to handle (see cm_sdpa_vlen.cm memory-layout contract).
    void add_qkv_split(topology& topo,
                       const primitive_id& fused_id,
                       const primitive_id& axis_data_id,
                       int32_t num_heads,
                       int32_t head_size) {
        const auto slice_shape = tensor(batch(1), feature(num_heads), spatial(head_size, 1));
        topo.add(crop("q", { input_info(fused_id), input_info(axis_data_id) },
                      slice_shape, tensor(feature(0 * num_heads)),
                      crop_ngraph_op_mode::split, 0, /*axis*/1, /*num_splits*/3));
        topo.add(crop("k", { input_info(fused_id), input_info(axis_data_id) },
                      slice_shape, tensor(feature(1 * num_heads)),
                      crop_ngraph_op_mode::split, 1, /*axis*/1, /*num_splits*/3));
        topo.add(crop("v", { input_info(fused_id), input_info(axis_data_id) },
                      slice_shape, tensor(feature(2 * num_heads)),
                      crop_ngraph_op_mode::split, 2, /*axis*/1, /*num_splits*/3));
    }

    topology create_ref_topology_fused(layout fused_layout,
                                       layout attn_mask_layout,
                                       memory::ptr axis_data_mem,
                                       int32_t num_heads,
                                       int32_t head_size) {
        topology topo;
        topo.add(input_layout("fused", fused_layout));
        topo.add(data("split_axis_ref", axis_data_mem));
        add_qkv_split(topo, "fused", "split_axis_ref", num_heads, head_size);

        topo.add(input_layout("attn", attn_mask_layout));  // "attention_mask"

        auto sdpa_prim = scaled_dot_product_attention("sdpa", {input_info("q"), input_info("k"), input_info("v"), input_info("attn")},
            false, -1, order_q, order_k, order_v, {0, 1, 2}, {}, false);
        topo.add(sdpa_prim);
        topo.add(permute("permute_o", input_info("sdpa"), order_o2));
        topo.add(reorder("result", input_info("permute_o"), format::bfyx, data_types::f16));
        return topo;
    }

    topology create_topology_fused(layout fused_layout,
                                   layout cu_seqlen_layout,
                                   memory::ptr axis_data_mem,
                                   int32_t num_heads,
                                   int32_t head_size) {
        topology topo;
        topo.add(input_layout("fused", fused_layout));
        topo.add(data("split_axis_opt", axis_data_mem));
        add_qkv_split(topo, "fused", "split_axis_opt", num_heads, head_size);

        topo.add(input_layout("attn", cu_seqlen_layout));  // "cu_seqlens"

        auto vlsdpa_prim = vl_sdpa("vlsdpa", {input_info("q"), input_info("k"), input_info("v"), input_info("attn")},
            order_q, order_k, order_v, order_o);
        topo.add(vlsdpa_prim);
        topo.add(reorder("result", input_info("vlsdpa"), format::bfyx, data_types::f16));
        return topo;
    }

    std::tuple<cldnn::memory::ptr, cldnn::network::ptr> run_network(topology &topo,
            cldnn::memory::ptr q_mem,
            cldnn::memory::ptr k_mem,
            cldnn::memory::ptr v_mem,
            cldnn::memory::ptr attn_mem,
            const bool is_caching_test) {
        auto& engine = get_test_engine();

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        net->set_input_data("q", q_mem);
        net->set_input_data("k", k_mem);
        net->set_input_data("v", v_mem);
        net->set_input_data("attn", attn_mem);

        auto outputs = net->execute();
        auto output = outputs.at("result").get_memory();
        return {output, net};
    }

    void execute(vlsdpa_test_params& p, const bool is_caching_test = false) {
        const auto head_size = p.head_size;
        const auto num_heads = p.num_heads;
        const auto cu_seqlens = p.cu_seqlens;
        const auto cumsum = cu_seqlens.back();
        const auto seq_length_q = cumsum;
        const auto seq_length_kv = cumsum;

        assert(cu_seqlens.front() == 0);

        auto& engine = get_test_engine();

        // create topologies
        cldnn::layout q_dyn_layout({-1, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout k_dyn_layout({-1, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout v_dyn_layout({-1, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout attn_mask_dyn_layout({1, -1, -1}, data_types::f16, format::bfyx);
        cldnn::layout cu_seqlens_dyn_layout({-1}, data_types::i32, format::bfyx);

        topology ref_topo = create_ref_topology(q_dyn_layout, k_dyn_layout, v_dyn_layout, attn_mask_dyn_layout);
        topology opt_topo = create_topology(q_dyn_layout, k_dyn_layout, v_dyn_layout, cu_seqlens_dyn_layout);

        // allocate memories
        cldnn::layout q_static_layout({seq_length_q,  num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout k_static_layout({seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout v_static_layout({seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout attn_mask_static_layout({1, seq_length_q, seq_length_kv}, data_types::f16, format::bfyx);
        cldnn::layout cu_seqlens_static_layout({static_cast<ov::Dimension::value_type>(cu_seqlens.size())}, data_types::i32, format::bfyx);

        auto q_mem = engine.allocate_memory(q_static_layout);
        auto k_mem = engine.allocate_memory(k_static_layout);
        auto v_mem = engine.allocate_memory(v_static_layout);
        auto attn_mask_mem = engine.allocate_memory(attn_mask_static_layout);
        auto cu_seqlens_mem = engine.allocate_memory(cu_seqlens_static_layout);

        load_input(q_mem, 0);
        load_input(k_mem, 1);
        load_input(v_mem, 2);
        get_attention_mask(attn_mask_mem, cu_seqlens);
        get_cu_seqlens(cu_seqlens_mem, cu_seqlens);

        // execute networks
        auto [mem_ref_ptr, net_ref_ptr] = run_network(ref_topo,
                                        q_mem, k_mem, v_mem, attn_mask_mem,
                                        is_caching_test);
        auto [mem_opt_ptr, net_opt_ptr] = run_network(opt_topo,
                                        q_mem, k_mem, v_mem, cu_seqlens_mem,
                                        is_caching_test);

        // validate results
        cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_data(mem_ref_ptr, get_test_stream());
        cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_data(mem_opt_ptr, get_test_stream());
        {
            for (size_t idx = 0; idx < ref_data.size(); idx++) {
                ASSERT_FALSE(std::isnan(opt_data[idx]) || std::isnan(ref_data[idx])) << "NaN found at index " << idx;
            }
            auto ret = cosineSimilarity(ref_data, opt_data);
            ASSERT_GE(ret, 0.95f);
        }
    }

    void execute_fused_qkv(vlsdpa_test_params& p, const bool is_caching_test = false) {
        const auto head_size = p.head_size;
        const auto num_heads = p.num_heads;
        const auto cu_seqlens = p.cu_seqlens;
        const auto cumsum = cu_seqlens.back();
        const auto seq_length_q = cumsum;
        const auto seq_length_kv = cumsum;

        assert(cu_seqlens.front() == 0);

        auto& engine = get_test_engine();

        // Parent-layout dims: {L, 3*H, S}. After optimize_data-driven buffer fusing,
        // the three Split(axis=1, num_splits=3) crops become in-place views of this
        // parent, producing the discontinuous token-axis layout the CM kernel expects.
        const int32_t fused_feature = 3 * num_heads;
        cldnn::layout fused_dyn_layout({-1, fused_feature, head_size}, data_types::f16, format::bfyx);
        cldnn::layout attn_mask_dyn_layout({1, -1, -1}, data_types::f16, format::bfyx);
        cldnn::layout cu_seqlens_dyn_layout({-1}, data_types::i32, format::bfyx);

        // Split-axis constant shared by both topologies (Split expects axis as i64 scalar).
        auto axis_data_mem = engine.allocate_memory({{}, data_types::i64, format::bfyx});
        set_values<int64_t>(axis_data_mem, {1});

        topology ref_topo = create_ref_topology_fused(fused_dyn_layout, attn_mask_dyn_layout,
                                                     axis_data_mem, num_heads, head_size);
        topology opt_topo = create_topology_fused(fused_dyn_layout, cu_seqlens_dyn_layout,
                                                  axis_data_mem, num_heads, head_size);

        // Concrete input allocations. The fused parent is filled once and reused by both
        // networks, so any output divergence isolates the vl_sdpa kernel's handling of
        // the non-zero token_offset_{q,k,v} / {q,k,v}_token_pitch scalar path.
        cldnn::layout fused_static_layout({seq_length_q, fused_feature, head_size}, data_types::f16, format::bfyx);
        cldnn::layout attn_mask_static_layout({1, seq_length_q, seq_length_kv}, data_types::f16, format::bfyx);
        cldnn::layout cu_seqlens_static_layout({static_cast<ov::Dimension::value_type>(cu_seqlens.size())}, data_types::i32, format::bfyx);

        auto fused_mem = engine.allocate_memory(fused_static_layout);
        auto attn_mask_mem = engine.allocate_memory(attn_mask_static_layout);
        auto cu_seqlens_mem = engine.allocate_memory(cu_seqlens_static_layout);

        load_input(fused_mem, 0);
        get_attention_mask(attn_mask_mem, cu_seqlens);
        get_cu_seqlens(cu_seqlens_mem, cu_seqlens);

        auto run = [&](topology& topo, cldnn::memory::ptr attn_mem, bool optimize) {
            ExecutionConfig config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            // optimize=false keeps the reference crops materialized (contiguous Q/K/V).
            // optimize=true lets prepare_buffer_fusing fold the vl_sdpa crops in-place,
            // which is the whole point of this test.
            config.set_property(ov::intel_gpu::optimize_data(optimize));

            cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);
            net->set_input_data("fused", fused_mem);
            net->set_input_data("attn", attn_mem);

            auto outputs = net->execute();
            return outputs.at("result").get_memory();
        };

        auto mem_ref_ptr = run(ref_topo, attn_mask_mem, /*optimize=*/false);
        auto mem_opt_ptr = run(opt_topo, cu_seqlens_mem, /*optimize=*/true);

        cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_data(mem_ref_ptr, get_test_stream());
        cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_data(mem_opt_ptr, get_test_stream());
        for (size_t idx = 0; idx < ref_data.size(); idx++) {
            ASSERT_FALSE(std::isnan(opt_data[idx]) || std::isnan(ref_data[idx])) << "NaN found at index " << idx;
        }
        auto ret = cosineSimilarity(ref_data, opt_data);
        ASSERT_GE(ret, 0.95f);
    }

    static std::string
    PrintToStringParamName(const testing::TestParamInfo<vlsdpa_test_params>& info) {
        std::string result = "vlsdpa_gpu_test_" + std::to_string(info.param.head_size) + "_" +
               std::to_string(info.param.num_heads) + "_" +
               vec2str<int32_t>(info.param.cu_seqlens);

        return result;
    }

    static bool check_vlsdpa_available() {
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        if (!cldnn::check_cm_jit_support(engine, config) || !engine.get_device_info().supports_immad) {
            return false;
        }

        return true;
    }

    const std::vector<int64_t> order_q = {1, 0, 2};
    const std::vector<int64_t> order_k = {1, 0, 2};
    const std::vector<int64_t> order_v = {1, 0, 2};
    const std::vector<int64_t> order_o = {1, 0, 2};
    const std::vector<uint16_t> order_o2 = {1, 0, 2};  // primitive permute constructor asks "ushort"
};

TEST_P(vlsdpa_gpu_test, basic) {
    if (!check_vlsdpa_available())
        GTEST_SKIP();

    auto p = GetParam();
    execute(p);
}

TEST_P(vlsdpa_gpu_test, basic_caching) {
    if (!check_vlsdpa_available())
        GTEST_SKIP();

    auto p = GetParam();
    execute(p, true);
}

// Exercises the token_offset_{q,k,v} / {q,k,v}_token_pitch runtime scalars that
// the CM vl_sdpa kernel gained to handle discontinuous Q/K/V slices of a fused
// [L, 3*H, S] parent buffer (Split(axis=1, num_splits=3) with in-place buffer fusing).
TEST_P(vlsdpa_gpu_test, discontinuous_qkv_fused) {
    if (!check_vlsdpa_available())
        GTEST_SKIP();

    auto p = GetParam();
    execute_fused_qkv(p);
}

TEST_P(vlsdpa_gpu_test, discontinuous_qkv_fused_caching) {
    if (!check_vlsdpa_available())
        GTEST_SKIP();

    auto p = GetParam();
    execute_fused_qkv(p, true);
}

INSTANTIATE_TEST_SUITE_P(smoke_vlsdpa_gpu_test,
    vlsdpa_gpu_test,
    ::testing::Values(
        vlsdpa_test_params{64 /*head_size*/, 1 /*num_head*/, {0, 16} /*cu_seqlens*/},
        vlsdpa_test_params{72 /*head_size*/, 1 /*num_head*/, {0, 16} /*cu_seqlens*/},
        vlsdpa_test_params{128 /*head_size*/, 1 /*num_head*/, {0, 16} /*cu_seqlens*/},
        vlsdpa_test_params{64 /*head_size*/, 2 /*num_head*/, {0, 16} /*cu_seqlens*/},
        vlsdpa_test_params{72 /*head_size*/, 2 /*num_head*/, {0, 16} /*cu_seqlens*/},
        vlsdpa_test_params{64 /*head_size*/, 1 /*num_head*/, {0, 16, 32} /*cu_seqlens*/},
        vlsdpa_test_params{64 /*head_size*/, 2 /*num_head*/, {0, 16, 32} /*cu_seqlens*/},
        vlsdpa_test_params{72 /*head_size*/, 2 /*num_head*/, {0, 16, 32} /*cu_seqlens*/}
    ),
    vlsdpa_gpu_test::PrintToStringParamName
);

} // namespace
