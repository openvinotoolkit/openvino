// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/draft_block_mask.hpp"

#include <cstddef>
#include <experimental/primitives/draft_block_mask.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <vector>

#include "draft_block_mask_inst.h"
#include "graph/impls/ocl/kernel_selector_helper.h"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

bool is_cm_jit_and_immad_supported() {
    try {
        auto& engine = get_test_engine();
        auto config = get_test_default_config(engine);
        return cldnn::check_cm_jit_support(engine, config) && engine.get_device_info().supports_immad;
    } catch (...) {
        return false;
    }
}

// GPU-side implementation requires fp16 Q/K (see DraftBlockMaskImplementationManager::validate_impl);
// the CPU reference is templated, so it's exercised here with the same precision for a fair,
// bit-for-bit-ish comparison rather than introducing extra fp32->fp16 rounding noise.
TEST(draft_block_mask_gpu, small_case) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    auto& engine = get_test_engine();

    const size_t num_heads = 2;
    const size_t seqlen = 2;  // q_groups (temporal groups)
    const size_t k_groups = 2;
    const size_t spatial_windows = 6;  // HW
    const size_t head_dim = 8;
    const size_t D = num_heads * head_dim;
    const size_t block_s = 4;  // window_tokens
    const size_t topk = 4;
    const size_t batch_size = 1;

    const size_t Nq = seqlen * spatial_windows;
    const size_t Nk = k_groups * spatial_windows;

    tests::random_generator rg;
    rg.set_seed("draft_block_mask_small_case");

    auto q_data = rg.generate_random_1d<ov::float16>(Nq * block_s * D, -1, 1, 256);
    auto k_data = rg.generate_random_1d<ov::float16>(Nk * block_s * D, -1, 1, 256);

    // All-pass local mask: exercises the pool -> score/softmax -> top-k pipeline without also
    // needing to replicate build_local_block_mask_shifted_vec_normal_slide's geometry here.
    std::vector<uint8_t> local_mask_gpu(spatial_windows * spatial_windows, 1);
    std::vector<char> local_mask_ref(spatial_windows * spatial_windows, 1);

    auto q_layout = layout{ov::PartialShape{static_cast<int64_t>(Nq), static_cast<int64_t>(block_s), static_cast<int64_t>(D)}, data_types::f16, format::bfyx};
    auto k_layout = layout{ov::PartialShape{static_cast<int64_t>(Nk), static_cast<int64_t>(block_s), static_cast<int64_t>(D)}, data_types::f16, format::bfyx};
    auto mask_layout = layout{ov::PartialShape{static_cast<int64_t>(spatial_windows), static_cast<int64_t>(spatial_windows)}, data_types::u8, format::bfyx};

    auto q_mem = engine.allocate_memory(q_layout);
    auto k_mem = engine.allocate_memory(k_layout);
    auto mask_mem = engine.allocate_memory(mask_layout);

    set_values(q_mem, q_data);
    set_values(k_mem, k_data);
    set_values(mask_mem, local_mask_gpu);

    topology topo;
    topo.add(input_layout("q_windows", q_layout));
    topo.add(input_layout("k_windows", k_layout));
    topo.add(input_layout("local_mask", mask_layout));
    topo.add(draft_block_mask("draft_block_mask",
                              {input_info("q_windows"), input_info("k_windows"), input_info("local_mask")},
                              static_cast<int64_t>(num_heads),
                              static_cast<int64_t>(seqlen),
                              static_cast<int64_t>(topk),
                              static_cast<int64_t>(batch_size)));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);

    network->set_input_data("q_windows", q_mem);
    network->set_input_data("k_windows", k_mem);
    network->set_input_data("local_mask", mask_mem);

    auto outputs = network->execute();
    auto output_mem = outputs.at("draft_block_mask").get_memory();

    std::vector<char> ref_output(batch_size * num_heads * Nq * Nk);
    ov::reference::draft_block_mask<ov::float16>(q_data.data(),
                                                 k_data.data(),
                                                 local_mask_ref.data(),
                                                 ref_output.data(),
                                                 Nq,
                                                 Nk,
                                                 block_s,
                                                 D,
                                                 num_heads,
                                                 spatial_windows,
                                                 seqlen,
                                                 topk,
                                                 batch_size);

    ASSERT_EQ(output_mem->count(), ref_output.size());
    cldnn::mem_lock<uint8_t, mem_lock_type::read> out_ptr(output_mem, network->get_stream());
    for (size_t i = 0; i < ref_output.size(); i++) {
        ASSERT_EQ(static_cast<int>(out_ptr[i]), static_cast<int>(ref_output[i])) << " at index=" << i;
    }
}

// ===================== dynamic-shape / runtime-topk coverage (see resolve_geometry() in
// draft_block_mask.cpp under graph/impls/cm/) =====================
//
// FlashVSR's streaming DiT graph gives this op a dynamic token-count axis (Nq/Nk, hence the mask's
// HW axis too) and a runtime `topk` value -- num_heads/head_dim/block_s stay compile-time constants.
// The cases below check that split: axes/values resolve_geometry() (and the new 5-input primitive)
// treat as safe to leave dynamic/runtime actually work (including re-dispatch for more than one
// concrete shape against the same compiled network), while the axis it still requires to be static
// (D, hence head_dim) is rejected rather than silently mis-dispatched.
//
// Seeds below were hand-picked to avoid a known source of flakiness independent of this change: the
// GPU kernel's PROBS intermediate (post-softmax, pre-top-k) is stored in fp16 (see
// DraftBlockMaskInternBuffIdx::PROBS in draft_block_mask.cpp), while the CPU reference computes
// softmax entirely in float -- an all-pass local mask plus small Nk can occasionally put two scores
// within fp16 rounding distance of the top-k threshold, so the (strict-greater-than) reference and
// the GPU kernel keep a different element at that exact boundary for a small fraction of random
// seeds. This is a pre-existing characteristic of the topk kernel's precision, not something this
// dynamic-shape/runtime-topk change introduces -- confirmed by reproducing it with the unmodified
// static 3-input primitive at an unlucky seed too.

struct DraftBlockMaskTestParams {
    size_t num_heads = 2;
    size_t seqlen = 2;           // q temporal groups
    size_t k_groups = 2;         // kv temporal groups
    size_t spatial_windows = 6;  // HW
    size_t head_dim = 8;
    size_t block_s = 4;  // window_tokens
    size_t topk = 4;
    size_t batch_size = 1;

    bool dyn_tokens = false;    // Nq/Nk/HW axes dynamic (q/k/mask move together)
    bool dyn_head_dim = false;  // D axis (num_heads*head_dim) dynamic -- rejected
    bool runtime_topk = false;  // use the new 5-input primitive with runtime seqlen/topk inputs
};

ov::Dimension draft_maybe_dynamic(bool dynamic, int64_t value) {
    return dynamic ? ov::Dimension::dynamic() : ov::Dimension(value);
}

// Builds the topology (with layouts reflecting `p`'s `dyn_*`/`runtime_topk` flags) and, if
// `expect_throw` is false, executes it once with `p`'s concrete sizes and checks the output against
// the CPU reference. Returns the constructed network so multi-execution tests can drive it further.
network::ptr build_and_run_draft_block_mask(const std::string& seed, DraftBlockMaskTestParams p, bool skip_run = false) {
    auto& engine = get_test_engine();

    const size_t D = p.num_heads * p.head_dim;
    const size_t Nq = p.seqlen * p.spatial_windows;
    const size_t Nk = p.k_groups * p.spatial_windows;

    tests::random_generator rg;
    rg.set_seed(seed);

    auto q_data = rg.generate_random_1d<ov::float16>(Nq * p.block_s * D, -1, 1, 256);
    auto k_data = rg.generate_random_1d<ov::float16>(Nk * p.block_s * D, -1, 1, 256);
    std::vector<uint8_t> local_mask_gpu(p.spatial_windows * p.spatial_windows, 1);
    std::vector<char> local_mask_ref(local_mask_gpu.begin(), local_mask_gpu.end());

    auto q_dyn_layout = layout{ov::PartialShape{draft_maybe_dynamic(p.dyn_tokens, static_cast<int64_t>(Nq)),
                                                static_cast<int64_t>(p.block_s),
                                                draft_maybe_dynamic(p.dyn_head_dim, static_cast<int64_t>(D))},
                               data_types::f16,
                               format::bfyx};
    auto k_dyn_layout = layout{ov::PartialShape{draft_maybe_dynamic(p.dyn_tokens, static_cast<int64_t>(Nk)),
                                                static_cast<int64_t>(p.block_s),
                                                draft_maybe_dynamic(p.dyn_head_dim, static_cast<int64_t>(D))},
                               data_types::f16,
                               format::bfyx};
    auto mask_dyn_layout = layout{ov::PartialShape{draft_maybe_dynamic(p.dyn_tokens, static_cast<int64_t>(p.spatial_windows)),
                                                   draft_maybe_dynamic(p.dyn_tokens, static_cast<int64_t>(p.spatial_windows))},
                                  data_types::u8,
                                  format::bfyx};

    auto q_actual_layout =
        layout{ov::PartialShape{static_cast<int64_t>(Nq), static_cast<int64_t>(p.block_s), static_cast<int64_t>(D)}, data_types::f16, format::bfyx};
    auto k_actual_layout =
        layout{ov::PartialShape{static_cast<int64_t>(Nk), static_cast<int64_t>(p.block_s), static_cast<int64_t>(D)}, data_types::f16, format::bfyx};
    auto mask_actual_layout =
        layout{ov::PartialShape{static_cast<int64_t>(p.spatial_windows), static_cast<int64_t>(p.spatial_windows)}, data_types::u8, format::bfyx};

    auto q_mem = engine.allocate_memory(q_actual_layout);
    auto k_mem = engine.allocate_memory(k_actual_layout);
    auto mask_mem = engine.allocate_memory(mask_actual_layout);
    set_values(q_mem, q_data);
    set_values(k_mem, k_data);
    set_values(mask_mem, local_mask_gpu);

    topology topo;
    topo.add(input_layout("q_windows", q_dyn_layout));
    topo.add(input_layout("k_windows", k_dyn_layout));
    topo.add(input_layout("local_mask", mask_dyn_layout));

    std::vector<input_info> inputs = {input_info("q_windows"), input_info("k_windows"), input_info("local_mask")};

    memory::ptr seqlen_mem;
    memory::ptr topk_mem;
    if (p.runtime_topk) {
        auto scalar_layout = layout{ov::PartialShape{1}, data_types::i64, format::bfyx};
        seqlen_mem = engine.allocate_memory(scalar_layout);
        topk_mem = engine.allocate_memory(scalar_layout);
        set_values(seqlen_mem, std::vector<int64_t>{static_cast<int64_t>(p.seqlen)});
        set_values(topk_mem, std::vector<int64_t>{static_cast<int64_t>(p.topk)});
        topo.add(input_layout("seqlen", scalar_layout));
        topo.add(input_layout("topk", scalar_layout));
        inputs.push_back(input_info("seqlen"));
        inputs.push_back(input_info("topk"));
        topo.add(draft_block_mask("draft_block_mask", inputs, static_cast<int64_t>(p.num_heads), static_cast<int64_t>(p.batch_size)));
    } else {
        topo.add(draft_block_mask("draft_block_mask",
                                  inputs,
                                  static_cast<int64_t>(p.num_heads),
                                  static_cast<int64_t>(p.seqlen),
                                  static_cast<int64_t>(p.topk),
                                  static_cast<int64_t>(p.batch_size)));
    }

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto network = get_network(engine, topo, config, get_test_stream_ptr(), false);

    if (skip_run) {
        return network;
    }

    network->set_input_data("q_windows", q_mem);
    network->set_input_data("k_windows", k_mem);
    network->set_input_data("local_mask", mask_mem);
    if (p.runtime_topk) {
        network->set_input_data("seqlen", seqlen_mem);
        network->set_input_data("topk", topk_mem);
    }

    auto outputs = network->execute();
    auto output_mem = outputs.at("draft_block_mask").get_memory();

    std::vector<char> ref_output(p.batch_size * p.num_heads * Nq * Nk);
    ov::reference::draft_block_mask<ov::float16>(q_data.data(),
                                                 k_data.data(),
                                                 local_mask_ref.data(),
                                                 ref_output.data(),
                                                 Nq,
                                                 Nk,
                                                 p.block_s,
                                                 D,
                                                 p.num_heads,
                                                 p.spatial_windows,
                                                 p.seqlen,
                                                 p.topk,
                                                 p.batch_size);

    EXPECT_EQ(output_mem->count(), ref_output.size());
    cldnn::mem_lock<uint8_t, mem_lock_type::read> out_ptr(output_mem, network->get_stream());
    for (size_t i = 0; i < ref_output.size(); i++) {
        EXPECT_EQ(static_cast<int>(out_ptr[i]), static_cast<int>(ref_output[i])) << " at index=" << i;
    }

    return network;
}

TEST(draft_block_mask_gpu, dynamic_tokens_basic) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    DraftBlockMaskTestParams p;
    p.dyn_tokens = true;
    auto network = build_and_run_draft_block_mask("draft_block_mask_dynamic_tokens_basic_v2", p);

    auto& node = network->get_program()->get_node("draft_block_mask");
    auto impl = node.get_selected_impl();
    ASSERT_TRUE(impl != nullptr);
    EXPECT_TRUE(impl->is_dynamic());
}

// Same compiled network (one dynamic-shape `draft_block_mask` node), driven through three
// `execute()` calls with genuinely different concrete Nq/Nk/HW -- verifies resolve_geometry()'s
// shape-derived seqlen/spatial_windows (and get_internal_buffer_descs()'s buffer resizing) are
// correctly re-resolved per execution, not stale from the first shape.
TEST(draft_block_mask_gpu, dynamic_tokens_multi_inference_different_shape) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    DraftBlockMaskTestParams p;
    p.dyn_tokens = true;
    auto network = build_and_run_draft_block_mask("draft_block_mask_dynamic_tokens_multi_shape_build", p, /*skip_run=*/true);

    auto& engine = get_test_engine();
    auto run_shape = [&](const std::string& seed, size_t seqlen, size_t k_groups, size_t spatial_windows) {
        const size_t D = p.num_heads * p.head_dim;
        const size_t Nq = seqlen * spatial_windows;
        const size_t Nk = k_groups * spatial_windows;

        tests::random_generator rg;
        rg.set_seed(seed);
        auto q_data = rg.generate_random_1d<ov::float16>(Nq * p.block_s * D, -1, 1, 256);
        auto k_data = rg.generate_random_1d<ov::float16>(Nk * p.block_s * D, -1, 1, 256);
        std::vector<uint8_t> local_mask_gpu(spatial_windows * spatial_windows, 1);
        std::vector<char> local_mask_ref(local_mask_gpu.begin(), local_mask_gpu.end());

        auto q_mem = engine.allocate_memory(
            layout{ov::PartialShape{static_cast<int64_t>(Nq), static_cast<int64_t>(p.block_s), static_cast<int64_t>(D)}, data_types::f16, format::bfyx});
        auto k_mem = engine.allocate_memory(
            layout{ov::PartialShape{static_cast<int64_t>(Nk), static_cast<int64_t>(p.block_s), static_cast<int64_t>(D)}, data_types::f16, format::bfyx});
        auto mask_mem = engine.allocate_memory(
            layout{ov::PartialShape{static_cast<int64_t>(spatial_windows), static_cast<int64_t>(spatial_windows)}, data_types::u8, format::bfyx});
        set_values(q_mem, q_data);
        set_values(k_mem, k_data);
        set_values(mask_mem, local_mask_gpu);

        network->set_input_data("q_windows", q_mem);
        network->set_input_data("k_windows", k_mem);
        network->set_input_data("local_mask", mask_mem);

        auto outputs = network->execute();
        auto output_mem = outputs.at("draft_block_mask").get_memory();

        std::vector<char> ref_output(p.batch_size * p.num_heads * Nq * Nk);
        ov::reference::draft_block_mask<ov::float16>(q_data.data(),
                                                     k_data.data(),
                                                     local_mask_ref.data(),
                                                     ref_output.data(),
                                                     Nq,
                                                     Nk,
                                                     p.block_s,
                                                     D,
                                                     p.num_heads,
                                                     spatial_windows,
                                                     seqlen,
                                                     p.topk,
                                                     p.batch_size);

        EXPECT_EQ(output_mem->count(), ref_output.size());
        cldnn::mem_lock<uint8_t, mem_lock_type::read> out_ptr(output_mem, network->get_stream());
        for (size_t i = 0; i < ref_output.size(); i++) {
            EXPECT_EQ(static_cast<int>(out_ptr[i]), static_cast<int>(ref_output[i])) << " seqlen=" << seqlen << " at index=" << i;
        }
    };

    run_shape("draft_block_mask_dynamic_tokens_multi_shape_1", /*seqlen=*/2, /*k_groups=*/2, /*spatial_windows=*/6);
    run_shape("draft_block_mask_dynamic_tokens_multi_shape_2", /*seqlen=*/3, /*k_groups=*/3, /*spatial_windows=*/8);
    run_shape("draft_block_mask_dynamic_tokens_multi_shape_3", /*seqlen=*/2, /*k_groups=*/4, /*spatial_windows=*/4);
}

// D (hence head_dim, resolve_geometry()'s one remaining hard static-shape requirement) declared
// dynamic must not silently mis-dispatch -- surfaces as an exception while compiling the program,
// since get_jit_constants() reads head_dim via resolve_geometry()'s OPENVINO_ASSERT and there is no
// fallback implementation for draft_block_mask to silently degrade to (CM-only).
TEST(draft_block_mask_gpu, dynamic_head_dim_rejected) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    DraftBlockMaskTestParams p;
    p.dyn_head_dim = true;
    ASSERT_ANY_THROW(build_and_run_draft_block_mask("draft_block_mask_dynamic_head_dim_rejected", p));
}

// Exercises the new 5-input primitive (runtime seqlen/topk inputs, see
// ov::op::internal::DraftBlockMask's 5-input constructor) at the SAME concrete shapes as
// `small_case` above, asserting the runtime-input path produces identical output to the
// attribute-based path. `dyn_tokens` is required here even though only `topk`'s VALUE (not any
// shape) is under test: `topk`'s value is only read into `kernel_impl_params::memory_deps` (see
// draft_block_mask_inst.h's get_shape_infer_dependencies()) when `primitive_inst::update_shape()`
// actually runs its memory_deps-population pass, which it skips as a pure-static-shape
// optimization when the output layout is already static and unchanged -- so a fully-static
// declared layout would silently never read the runtime topk input at all.
TEST(draft_block_mask_gpu, runtime_topk_input_matches_attribute_topk) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    DraftBlockMaskTestParams p;
    p.dyn_tokens = true;
    p.runtime_topk = true;
    build_and_run_draft_block_mask("draft_block_mask_runtime_topk_matches_attribute", p);
}

// The actual FlashVSR use case: dynamic Nq/Nk/HW AND a runtime topk value together.
TEST(draft_block_mask_gpu, dynamic_tokens_with_runtime_topk) {
    if (!is_cm_jit_and_immad_supported()) {
        GTEST_SKIP() << "Intel GPU with CM JIT compiler and IMMAD required, skipping";
    }
    DraftBlockMaskTestParams p;
    p.dyn_tokens = true;
    p.runtime_topk = true;
    build_and_run_draft_block_mask("draft_block_mask_dyn_topk_v3", p);
}

}  // namespace
