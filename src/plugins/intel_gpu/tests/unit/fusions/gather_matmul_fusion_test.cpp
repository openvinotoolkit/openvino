// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Unit fusion test for GatherMatmul + SwiGLU fusion.
// Verifies that prepare_primitive_fusing::fuse_swiglu() absorbs SwiGLU into GatherMatmul
// and that the fused result matches the non-fused reference.

#ifdef ENABLE_ONEDNN_FOR_GPU

#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/gather_matmul.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/swiglu.hpp>
#include <ov_ops/glu.hpp>

using namespace cldnn;
using namespace ::tests;

struct gather_matmul_fusion_test_params {
    ov::PartialShape in_shape;       // [n_activated_experts, n_tokens, K]
    ov::PartialShape weight_shape;   // [n_all_experts, 2N, K]  (2N = gate+value)
    ov::PartialShape indices_shape;  // [n_tokens, top_k]
    data_types data_type;
    data_types weights_type;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class GatherMatmulSwiGLUFusingTest : public ::BaseFusingTest<gather_matmul_fusion_test_params> {
public:
    void execute(gather_matmul_fusion_test_params& p) {
        cfg_not_fused.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        cfg_fused.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        auto input_prim = this->get_mem(layout{p.in_shape, p.data_type, format::bfyx});

        // Indices: fill with valid expert IDs
        auto idx_layout = layout{p.indices_shape, data_types::i32, format::bfyx};
        auto indices_prim = engine.allocate_memory(idx_layout);
        {
            auto n_experts = static_cast<int32_t>(p.weight_shape[0].get_length());
            auto count = static_cast<size_t>(ov::shape_size(p.indices_shape.to_shape()));
            std::vector<int32_t> idx_data(count);
            for (size_t i = 0; i < count; i++)
                idx_data[i] = static_cast<int32_t>(i % n_experts);
            set_values(indices_prim, idx_data);
        }

        network network_not_fused(this->engine, this->topology_non_fused, this->cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, this->cfg_fused);

        network_fused.set_input_data("input", input_prim);
        network_fused.set_input_data("indices", indices_prim);
        network_not_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("indices", indices_prim);

        this->compare(network_not_fused, network_fused, p);
    }
};

TEST_P(GatherMatmulSwiGLUFusingTest, basic) {
    // GatherMatmul requires systolic (IMMAD) support
    if (!engine.get_device_info().supports_immad)
        return;
    if (engine.get_device_info().execution_units_count < 128)
        return;

    auto p = GetParam();

    auto n_all_experts = p.weight_shape[0].get_length();
    auto n_2N = p.weight_shape[1].get_length();
    int64_t swiglu_length = static_cast<int64_t>(n_2N / 2);

    // Dynamic input layout for activations
    auto dynamic_input_layout = layout{ov::PartialShape::dynamic(p.in_shape.size()), p.data_type, format::bfyx};
    auto dynamic_indices_layout = layout{ov::PartialShape::dynamic(p.indices_shape.size()), data_types::i32, format::bfyx};

    // Scale: [n_all_experts, 2N, 1] — per-channel
    ov::Shape scale_shape = {static_cast<size_t>(n_all_experts), static_cast<size_t>(n_2N), 1};

    // Build topology: gather_matmul → swiglu → reorder(f32)
    create_topologies(
        input_layout("input", dynamic_input_layout),
        input_layout("indices", dynamic_indices_layout),
        data("weights", get_mem(layout{p.weight_shape, p.weights_type, format::bfyx})),
        data("bias", get_mem(layout{ov::PartialShape{0}, data_types::u8, format::bfyx}, 0)),
        data("scales", get_mem(layout{scale_shape, data_types::f16, format::bfyx}, 0.01f)),
        data("zp", get_mem(layout{ov::PartialShape{0}, data_types::u8, format::bfyx}, 0)),
        gather_matmul("bgm",
                      {input_info("input"), input_info("weights"), input_info("indices"),
                       input_info("bias"), input_info("scales"), input_info("zp")},
                      false,  // has_bias
                      false,  // has_zp
                      1),     // n_activated_experts
        swiglu("swiglu",
               input_info("bgm"),
               -1,                                            // axis (last)
               swiglu_length,                                 // glu_stride
               ov::op::internal::GLU::GluType::Swish,         // glu_type
               0,                                             // gate_idx
               tensor()),                                     // output_size (inferred)
        reorder("reorder_bfyx", input_info("swiglu"), format::bfyx, data_types::f32));

    tolerance = 1.0f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, GatherMatmulSwiGLUFusingTest, ::testing::ValuesIn(std::vector<gather_matmul_fusion_test_params>{
    // {in_shape, weight_shape, indices_shape, data_type, weights_type, expected_fused, expected_not_fused}
    // Per-channel, single token
    {
        {1, 1, 128},     // in: [1, 1, 128]
        {4, 128, 128},   // weights: [4, 2N=128, K=128] → N=64 after SwiGLU
        {1, 2},          // indices: [1, top_k=2]
        data_types::f16,
        data_types::u4,
        2,               // fused: bgm (with swiglu absorbed) + reorder
        3,               // not fused: bgm + swiglu + reorder
    },
    // Per-channel, multiple tokens
    {
        {1, 4, 128},     // in: [1, 4, 128]
        {4, 128, 128},   // weights: [4, 2N=128, K=128]
        {4, 2},          // indices: [4, top_k=2]
        data_types::f16,
        data_types::u4,
        2,
        3,
    },
}));

#endif  // ENABLE_ONEDNN_FOR_GPU
