// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <numeric>
#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/moe_gemm.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include "intel_gpu/op/moe_compressed.hpp"

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

enum class PHASE { PREFILL, UP, DOWN };
struct moe_gemm_test_params {
    PHASE phase;
    size_t num_tokens;
    size_t num_total_experts;
    size_t num_experts_per_token;
    size_t num_actually_used_experts;
    size_t hidden_size;
    size_t out_N;
    bool has_bias;
    cldnn::data_types input_dt = cldnn::data_types::f16;
    cldnn::data_types weight_dt = cldnn::data_types::f16;
    cldnn::data_types scale_dt = cldnn::data_types::f16;
    int32_t scale_group_size;
    bool weight_symmetric_quant;
    bool is_pa;
};

template <typename T>
void get_reference(const std::vector<ov::float16>& input,
                   const std::vector<T>& weight,
                   std::vector<ov::float16>& ref_out,
                   const std::vector<int32_t>& experts_ids,
                   const std::vector<int32_t>& input_offset_per_expert,
                   const std::vector<int32_t>& input_tokens_lens,
                   int32_t N,
                   int32_t K,
                   const std::vector<ov::float16>& weight_scale,
                   const std::vector<ov::float16>& weight_zp,
                   int32_t group_size,
                   bool is_weight_compressed,
                   bool is_weight_symmetric_quant,
                   cldnn::data_types weight_dt,
                   const std::vector<ov::float16>& bias) {
    std::cout << "get_reference" << std::endl;
    size_t elements_per_byte = (ov::element::Type(weight_dt).bitwidth() == 4) ? 2 : 1;
    auto ld_w = K / elements_per_byte;
    auto ld_in = K;
    auto ld_out = N;
    auto batch = input_offset_per_expert.size();
    auto expert_stride = ld_w * N;
    for (size_t b = 0; b < batch; b++) {
        int32_t expert_id = experts_ids[b];
        auto weight_ptr = &weight[expert_id * expert_stride];
        auto input_ptr = &input[input_offset_per_expert[b] * ld_in];
        auto ref_out_ptr = &ref_out[input_offset_per_expert[b] * ld_out];
        auto cur_m = input_tokens_lens[b];
        for (int j = 0; j < cur_m; j++) {
            for (int n = 0; n < N; n++) {
                auto weight_r = weight_ptr + n * ld_w;
                auto input_r = input_ptr + j * ld_in;
                float acc = 0.0f;
                for (int32_t k = 0; k < K; ++k) {
                    acc += static_cast<float>(weight_r[k]) * static_cast<float>(input_r[k]);
                }
                ref_out_ptr[j * ld_out + n] = static_cast<ov::float16>(acc);
            }
        }
    }
}
template <>
void get_reference(const std::vector<ov::float16>& input,
                   const std::vector<uint8_t>& weight,
                   std::vector<ov::float16>& ref_out,
                   const std::vector<int32_t>& experts_ids,
                   const std::vector<int32_t>& input_offset_per_expert,
                   const std::vector<int32_t>& input_tokens_lens,
                   int32_t N,
                   int32_t K,
                   const std::vector<ov::float16>& weight_scale,
                   const std::vector<ov::float16>& weight_zp,
                   int32_t group_size,
                   bool is_weight_compressed,
                   bool is_weight_symmetric_quant,
                   cldnn::data_types weight_dt,
                   const std::vector<ov::float16>& bias) {
    std::cout << "get_reference" << std::endl;
    size_t elements_per_byte = (ov::element::Type(weight_dt).bitwidth() == 4) ? 2 : 1;
    auto ld_w = K / elements_per_byte;
    auto ld_in = K;
    auto ld_out = N;
    auto batch = input_offset_per_expert.size();
    auto expert_stride = ld_w * N;
    int num_scale_groups = K / group_size;
    bool has_bias = bias.size() > 0;
    for (size_t b = 0; b < batch; b++) {
        int32_t expert_id = experts_ids[b];
        auto weight_ptr = &weight[expert_id * expert_stride];
        auto input_ptr = &input[input_offset_per_expert[b] * ld_in];
        auto ref_out_ptr = &ref_out[input_offset_per_expert[b] * ld_out];
        auto cur_m = input_tokens_lens[b];
        for (int j = 0; j < cur_m; j++) {
            for (int n = 0; n < N; n++) {
                auto weight_r = weight_ptr + n * ld_w;
                auto input_r = input_ptr + j * ld_in;
                float acc = 0.0f;
                int scale_group = 0;
                auto bias_val = has_bias ? bias[expert_id * N + n] : ov::float16(0.0f);
                if (is_weight_compressed) {
                    for (size_t k = 0; k < ld_w; k += (group_size / elements_per_byte)) {
                        for (size_t ki = k; ki < k + (group_size / elements_per_byte); ++ki) {
                            // decompress i4
                            if (weight_dt == cldnn::data_types::i4 && is_weight_symmetric_quant) {
                                int8_t q0 = static_cast<int8_t>(weight_r[ki] & 0x0F);
                                int8_t q1 = static_cast<int8_t>(weight_r[ki] >> 4) & 0x0F;
                                if (q0 > 7)
                                    q0 -= 16;
                                if (q1 > 7)
                                    q1 -= 16;
                                float scale = float(weight_scale[expert_id * N * num_scale_groups + n * num_scale_groups + scale_group]);
                                ov::float16 fa0 = ov::float16((float)q0 * (float)scale);
                                ov::float16 fa1 = ov::float16((float)q1 * (float)scale);
                                acc += (float)fa0 * input_r[2 * ki];
                                acc += (float)fa1 * input_r[2 * ki + 1];
                            } else if (weight_dt == cldnn::data_types::u4 && !is_weight_symmetric_quant) {
                                uint8_t q0 = (static_cast<uint8_t>(weight_r[ki]) & 0x0F);
                                uint8_t q1 = (static_cast<uint8_t>(weight_r[ki]) >> 4) & 0x0F;
                                float scale = float(weight_scale[expert_id * N * num_scale_groups + n * num_scale_groups + scale_group]);
                                float zp = float(weight_zp[expert_id * N * num_scale_groups + n * num_scale_groups + scale_group]);
                                float fa0 = (float(q0) - zp) * scale;
                                float fa1 = (float(q1) - zp) * scale;
                                acc += fa0 * input_r[2 * ki];
                                acc += fa1 * input_r[2 * ki + 1];
                            } else {
                                OPENVINO_ASSERT("Not implemented dt");
                            }
                        }
                        scale_group++;
                    }
                } else {
                    for (int32_t k = 0; k < K; ++k) {
                        acc += static_cast<float>(weight_r[k]) * static_cast<float>(input_r[k]);
                    }
                }
                ref_out_ptr[j * ld_out + n] = static_cast<ov::float16>(acc + bias_val);
            }
        }
    }
}

template <typename T>
struct MoEGemmTest : public ::testing::TestWithParam<T> {
    public:
    random_generator rg;
    cldnn::engine& engine = get_test_engine();

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }


    void quantize_4bit(std::vector<ov::float16>& weight_fp,
                       std::vector<uint8_t>& weight_quantized,
                       cldnn::data_types q_dt,
                       int B,
                       int N,
                       int K,
                       int group_size,
                       int is_symmetric,
                       std::vector<ov::float16>& weight_scale,
                       std::vector<ov::float16>& weight_zp) {
        const int K_q = K / 2;
        const int num_elements_per_byte = 2;
        const int num_scale_groups = K / group_size;

        for (int b = 0; b < B; b++) {
            for (int n = 0; n < N; n++) {
                int group_iter = 0;
                while (group_iter * group_size < K) {
                    ov::float16 amax = std::numeric_limits<ov::float16>::lowest();
                    ov::float16 amin = std::numeric_limits<ov::float16>::max();
                    for (int ki = 0; ki < group_size; ki++) {
                        ov::float16 v = weight_fp[b * N * K + n * K + group_iter * group_size + ki];
                        amax = std::max(amax, v);
                        amin = std::min(amin, v);
                    }
                    if (q_dt == cldnn::data_types::u4) {
                        OPENVINO_ASSERT(!is_symmetric);
                        const uint8_t q_max = 15;
                        const uint8_t q_min = 0;
                        float range = (float)amax - (float)amin;
                        if (range <= 1e-5f)
                            range = 1e-2f;
                        float inv_scale = (q_max - q_min) / range;
                        float zp_tmp = (float) (q_min - amin * inv_scale);
                        ov::float16 zp = zp_tmp;
                        // quantize asym u4
                        for (int ki = 0; ki < group_size / num_elements_per_byte; ki++)  {
                            ov::float16 v0 = weight_fp[(b * N * K) + (n * K) + (group_iter * group_size) + num_elements_per_byte * ki];
                            ov::float16 v1 = weight_fp[(b * N * K) + (n * K) + (group_iter * group_size) + num_elements_per_byte * ki + 1];
                            uint8_t q0 = std::min(std::max((uint8_t)(float(v0) * inv_scale + (float)zp), (uint8_t)0), q_max); // u4
                            uint8_t q1 = std::min(std::max((uint8_t)(float(v1) * inv_scale + (float)zp), (uint8_t)0), q_max); // u4
            
                            uint8_t q0q1 = (q1 << 4) | (q0 & 0x0F);
                            weight_quantized[(b * N * K_q) + (n * K_q) + (group_iter * group_size / num_elements_per_byte) + ki] = uint8_t(q0q1);
                        }
                        ov::float16 scale = 1 / inv_scale;
                        weight_scale[b * N + n * num_scale_groups + group_iter ] = scale;
                        weight_zp[b * N + n * num_scale_groups + group_iter] = zp;
                    } else if (q_dt == cldnn::data_types::i4) {
                        OPENVINO_ASSERT(is_symmetric);
                        const int8_t q_max = 7;
                        const int8_t q_min = -8;
                        float abs_max = std::max(std::abs(float(amax)), std::abs(float(amin)));
                        float inv_scale = (float)q_max / abs_max;
                        // quantize sym i4
                        for (int ki = 0; ki < group_size / num_elements_per_byte; ki++) {
                            ov::float16 v0 = weight_fp[(b * N * K) + (n * K) + (group_iter * group_size) + num_elements_per_byte * ki];
                            ov::float16 v1 = weight_fp[(b * N * K) + (n * K) + (group_iter * group_size) + num_elements_per_byte * ki + 1];
                            int8_t q0 = std::min(std::max((int8_t)(std::round(float(v0) * inv_scale)), q_min), q_max);  // u4
                            int8_t q1 = std::min(std::max((int8_t)(std::round(float(v1) * inv_scale)), q_min), q_max);  // u4
                            uint8_t q0q1 = ((uint8_t)q1 << 4) | ((uint8_t)q0 & 0x0F);
                            weight_quantized[b * N * K_q + (n * K_q) + (group_iter * group_size / num_elements_per_byte) + ki] = q0q1;
                        }
                        ov::float16 scale = 1 / inv_scale;
                        weight_scale[b * N * num_scale_groups + n * num_scale_groups + group_iter] = scale;
                    }
                    group_iter++;
                }
            }
        }
    }

    void create_weight_data_and_topology(T& p, topology& topo, std::vector<ov::float16>& experts_data_f16, std::vector<uint8_t>& experts_data_quant,
                         std::vector<ov::float16>& scales_data, std::vector<ov::float16>& zp_data, bool is_weight_compressed) {
        ov::intel_gpu::op::MOECompressed::Config moe_config;
        moe_config.top_k = p.num_experts_per_token;
        moe_config.num_expert = p.num_total_experts;
        moe_config.has_batch_dim = !p.is_pa;
        moe_config.hidden_size = p.hidden_size;
        moe_config.inter_size = p.out_N;
        auto num_scale_groups = p.hidden_size / p.scale_group_size;
        auto input_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(p.hidden_size)};
        auto input_act_layout = layout{input_shape, p.input_dt, format::bfyx};

        auto experts_shape = num_scale_groups > 1
                                 ? ov::PartialShape{ov::Dimension(p.num_total_experts), ov::Dimension(p.out_N), ov::Dimension(num_scale_groups), ov::Dimension(p.scale_group_size)}
                                 : ov::PartialShape{ov::Dimension(p.num_total_experts), ov::Dimension(p.out_N), ov::Dimension(p.hidden_size)};
        auto experts_f16_layout = layout{experts_shape, data_types::f16, format::bfyx};

        auto experts_ids_shape = ov::PartialShape{ov::Dimension::dynamic()};
        auto experts_ids_layout = layout{experts_ids_shape, data_types::i32, format::bfyx};
   
        auto input_offset_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
        auto input_offset_per_expert_layout = layout{input_offset_per_expert_shape, data_types::i32, format::bfyx};

        auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension::dynamic()};
        auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};

        auto input_prim = input_layout("input", input_act_layout);
        auto experts_ids_prim = input_layout("experts_ids", experts_ids_layout);
        auto input_offset_per_expert_prim = input_layout("input_offset_per_expert", input_offset_per_expert_layout);
        auto input_tokens_lens_prim = input_layout("input_tokens_lens", input_tokens_lens_layout);

        topo.add(input_prim);
        topo.add(experts_ids_prim);
        topo.add(input_offset_per_expert_prim);
        topo.add(input_tokens_lens_prim);
        // prepare experts weights
        layout experts_layout;
        if (is_weight_compressed) {
            experts_layout = layout{experts_shape, p.weight_dt, format::bfyx};
        } else {
            experts_layout = experts_f16_layout;
        }
        auto experts_mem = engine.allocate_memory(experts_layout);
        experts_data_f16.resize(p.num_total_experts * p.out_N * p.hidden_size);
        auto gen_f16_weight = [&](size_t num_total_experts, size_t expert_out_N, size_t hidden_size) {
            for (size_t e = 0; e < num_total_experts; ++e) {
                for (size_t n = 0; n < expert_out_N; ++n) {
                    for (size_t h = 0; h < hidden_size; ++h) {
                        size_t idx = e * expert_out_N * hidden_size + n * hidden_size + h;
                        experts_data_f16[idx] = static_cast<ov::float16>((e + (n % 3) + (h % 5) + 1) / 10.0f);
                    }
                }
            }
        };
        gen_f16_weight(p.num_total_experts, p.out_N, p.hidden_size);

        if (is_weight_compressed) {
            experts_data_quant.resize(p.num_total_experts * p.hidden_size * p.out_N / 2);
            scales_data.resize(p.num_total_experts * p.out_N * num_scale_groups);
            if (!p.weight_symmetric_quant) {
                zp_data.resize(p.num_total_experts * p.out_N * num_scale_groups);
            }
            quantize_4bit(experts_data_f16, experts_data_quant, p.weight_dt, p.num_total_experts, p.out_N, p.hidden_size, p.scale_group_size, p.weight_symmetric_quant, scales_data, zp_data);
            set_values(experts_mem, experts_data_quant);
            auto scale_shape = num_scale_groups > 1 ? ov::PartialShape{ov::Dimension(p.num_total_experts), ov::Dimension(p.out_N), ov::Dimension(num_scale_groups), ov::Dimension(1)} : 
                                                    ov::PartialShape{ov::Dimension(p.num_total_experts), ov::Dimension(p.out_N), ov::Dimension(1)};
            auto scale_layout = layout{scale_shape, data_types::f16, format::bfyx};
            auto scale_mem = engine.allocate_memory(scale_layout);
            set_values(scale_mem, scales_data);

            auto moe_experts_prim = data("moe_experts", experts_mem);
            auto moe_experts_scale_prim = data("moe_experts_scale", scale_mem);
            topo.add(moe_experts_prim);
            topo.add(moe_experts_scale_prim);
            if (!p.weight_symmetric_quant) {
                auto zp_shape = num_scale_groups > 1 ? ov::PartialShape{ov::Dimension(p.num_total_experts), ov::Dimension(p.out_N), ov::Dimension(num_scale_groups), ov::Dimension(1)} : 
                                                       ov::PartialShape{ov::Dimension(p.num_total_experts), ov::Dimension(p.out_N), ov::Dimension(1)};
                auto zp_layout = layout{zp_shape, data_types::f16, format::bfyx};
                auto zp_mem = engine.allocate_memory(zp_layout);
                set_values(zp_mem, zp_data);
                auto moe_experts_zp_prim = data("moe_experts_zp", zp_mem);
                topo.add(moe_experts_zp_prim);
                std::vector<input_info> inputs = {
                    input_info("input"),
                    input_info("moe_experts"),
                    input_info("experts_ids"),
                    input_info("input_offset_per_expert"),
                    input_info("input_tokens_lens"),
                    input_info("moe_experts_scale"),
                    input_info("moe_experts_zp"),
                };
                auto moe_gemm_prim = moe_gemm("moe_gemm", inputs, moe_config);
                moe_gemm_prim.has_bias = false;
                topo.add(moe_gemm_prim);
            } else {
                std::vector<input_info> inputs = {
                    input_info("input"),
                    input_info("moe_experts"),
                    input_info("experts_ids"),
                    input_info("input_offset_per_expert"),
                    input_info("input_tokens_lens"),
                    input_info("moe_experts_scale"),
                };

                auto moe_gemm_prim = moe_gemm("moe_gemm", inputs, moe_config);
                topo.add(moe_gemm_prim);
            }
        } else {
            set_values(experts_mem, experts_data_f16);
            auto moe_experts_prim = data("moe_experts", experts_mem);
            topo.add(moe_experts_prim);
            std::vector<input_info> inputs = {input_info("input"),
                                              input_info("moe_experts"),
                                              input_info("experts_ids"),
                                              input_info("input_offset_per_expert"),
                                              input_info("input_tokens_lens")};
            auto moe_gemm_prim = moe_gemm("moe_gemm", inputs, moe_config);
            topo.add(moe_gemm_prim);
        }
    }

    void execute(T& p) {
        topology topo;
        std::vector<ov::float16> experts_data_f16;
        std::vector<uint8_t> experts_data_q4;
        std::vector<ov::float16> scales_data;
        std::vector<ov::float16> zp_data;
        std::vector<cldnn::data_types> quant_types = {data_types::i4, data_types::u4, data_types::i8, data_types::u8};
        if (!engine.get_device_info().supports_immad || engine.get_device_info().arch != gpu_arch::xe2)
            return;

        bool is_weight_compressed = std::any_of(quant_types.begin(), quant_types.end(), [=](const cldnn::data_types& t) -> bool {
            return t == p.weight_dt;
        });
        create_weight_data_and_topology(p, topo, experts_data_f16, experts_data_q4, scales_data, zp_data, is_weight_compressed);
        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        network network(engine, topo, config);

        auto get_input_data = [] (size_t M, size_t K, random_generator& rg) {
            std::vector<ov::float16> input_data(M * K);
            for (size_t m = 0; m < M; ++m) {
                for (size_t k = 0; k < K; ++k) {
                    input_data[m * K + k] = static_cast<ov::float16>((k % 20 + 1.0f) / 10);
                }
            }
            return input_data;
        };

        const auto M = p.phase == PHASE::UP ? p.num_tokens : p.num_tokens * p.num_experts_per_token;
        auto input_data = get_input_data(M, p.hidden_size, rg);
        auto input_data_shape = p.is_pa ? ov::PartialShape{ov::Dimension(M), ov::Dimension(1), ov::Dimension(p.hidden_size)}
                                        : ov::PartialShape{ov::Dimension(1), ov::Dimension(M), ov::Dimension(p.hidden_size)};
        auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
        auto input_mem = engine.allocate_memory(input_data_layout);
        set_values(input_mem, input_data);

        std::vector<int32_t> experts_ids_data(p.num_actually_used_experts);
        const auto step = p.num_total_experts / p.num_actually_used_experts;
        for (size_t e = 0; e < p.num_actually_used_experts; e++) {
            experts_ids_data[e] = static_cast<int32_t>(e * step);
        }
        auto experts_ids_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(p.num_actually_used_experts))};
        auto experts_ids_data_layout = layout{experts_ids_data_shape, data_types::i32, format::bfyx};
        auto experts_ids_mem = engine.allocate_memory(experts_ids_data_layout);
        set_values(experts_ids_mem, experts_ids_data);

        std::vector<int32_t> input_offset_per_expert_data(p.num_actually_used_experts);
        std::vector<int32_t> input_tokens_lens(p.num_actually_used_experts);
        const auto avg_len = (M / p.num_actually_used_experts);
        for (size_t e = 0; e < p.num_actually_used_experts; e++) {
            size_t len = (e == p.num_actually_used_experts -1) ? M - (avg_len * e) : avg_len;
            input_tokens_lens[e] = static_cast<int32_t>(len);
            input_offset_per_expert_data[e] = static_cast<int32_t>(e * avg_len);
        }
        auto input_tokens_lens_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(p.num_actually_used_experts))};
        auto input_tokens_lens_data_layout = layout{input_tokens_lens_data_shape, data_types::i32, format::bfyx};
        auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_data_layout);
        set_values(input_tokens_lens_mem, input_tokens_lens);

        auto input_offset_per_expert_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(p.num_actually_used_experts))};
        auto input_offset_per_expert_data_layout = layout{input_offset_per_expert_data_shape, data_types::i32, format::bfyx};
        auto input_offset_per_expert_mem = engine.allocate_memory(input_offset_per_expert_data_layout);
        set_values(input_offset_per_expert_mem, input_offset_per_expert_data);

        network.set_input_data("input", input_mem);
        network.set_input_data("experts_ids", experts_ids_mem);
        network.set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
        network.set_input_data("input_tokens_lens", input_tokens_lens_mem);
        auto outputs = network.execute();
        auto output = outputs.at("moe_gemm").get_memory();
        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
        std::vector<ov::float16> output_ref(M * p.out_N, 0.0f);
        if (is_weight_compressed) {
            get_reference<uint8_t>(input_data,
                                   experts_data_q4,
                                   output_ref,
                                   experts_ids_data,
                                   input_offset_per_expert_data,
                                   input_tokens_lens,
                                   static_cast<int32_t>(p.out_N),
                                   static_cast<int32_t>(p.hidden_size),
                                   scales_data,
                                   zp_data,
                                   p.scale_group_size,
                                   is_weight_compressed,
                                   p.weight_symmetric_quant,
                                   p.weight_dt,
                                   {});
        } else {
            get_reference<ov::float16>(input_data,
                                   experts_data_f16,
                                   output_ref,
                                   experts_ids_data,
                                   input_offset_per_expert_data,
                                   input_tokens_lens,
                                   static_cast<int32_t>(p.out_N),
                                   static_cast<int32_t>(p.hidden_size),
                                   scales_data,
                                   zp_data,
                                   p.scale_group_size,
                                   is_weight_compressed,
                                   p.weight_symmetric_quant,
                                   p.weight_dt,
                                   {});
        }

        for (size_t i = 0; i < M * p.out_N; i++) {
            auto tolerance = std::max(std::abs(output_ref[i] * 0.01f), 0.1f);
            ASSERT_NEAR(output_ptr[i], output_ref[i], tolerance);
        }
    }
};




class moe_gemm_test : public MoEGemmTest<moe_gemm_test_params> {};
TEST_P(moe_gemm_test, basic) {
    auto p = GetParam();
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(smoke_moe_gemm,
                         moe_gemm_test,
                         ::testing::ValuesIn(std::vector<moe_gemm_test_params>{// f16 / prefill
                                                                               moe_gemm_test_params{
                                                                                   PHASE::PREFILL /*phase*/,
                                                                                   30,                     /*num_tokens*/
                                                                                   32,                     /*num_total_experts*/
                                                                                   2,                      /*num_experts_per_token*/
                                                                                   3,                      /*num_actually_used_experts*/
                                                                                   128,                    /*hidden_size*/
                                                                                   64,                     /*out_N*/
                                                                                   false,                  /*has_bias*/
                                                                                   cldnn::data_types::f16, /*input_dt*/
                                                                                   cldnn::data_types::f16, /*weight_dt*/
                                                                                   cldnn::data_types::f16,
                                                                                   /*scale_dt*/  // not used
                                                                                   -1,
                                                                                   /*scale_group_size*/  // not used
                                                                                   false                 /*weight_symmetric_quant*/
                                                                               },
                                                                               // f16 / up
                                                                               moe_gemm_test_params{
                                                                                   PHASE::UP /*phase*/,
                                                                                   1,                      /*num_tokens*/
                                                                                   32,                     /*num_total_experts*/
                                                                                   4,                      /*num_experts_per_token*/
                                                                                   4,                      /*num_actually_used_experts*/
                                                                                   128,                    /*hidden_size*/
                                                                                   64,                     /*out_N*/
                                                                                   false,                  /*has_bias*/
                                                                                   cldnn::data_types::f16, /*input_dt*/
                                                                                   cldnn::data_types::f16, /*weight_dt*/
                                                                                   cldnn::data_types::f16,
                                                                                   /*scale_dt*/  // not used
                                                                                   -1,
                                                                                   /*scale_group_size*/  // not used
                                                                                   false                 /*weight_symmetric_quant*/
                                                                               },
                                                                               // f16 / down
                                                                               moe_gemm_test_params{
                                                                                   PHASE::UP /*phase*/,
                                                                                   1,                      /*num_tokens*/
                                                                                   32,                     /*num_total_experts*/
                                                                                   4,                      /*num_experts_per_token*/
                                                                                   4,                      /*num_actually_used_experts*/
                                                                                   128,                    /*hidden_size*/
                                                                                   64,                     /*out_N*/
                                                                                   false,                  /*has_bias*/
                                                                                   cldnn::data_types::f16, /*input_dt*/
                                                                                   cldnn::data_types::f16, /*weight_dt*/
                                                                                   cldnn::data_types::f16,
                                                                                   /*scale_dt*/  // not used
                                                                                   -1,
                                                                                   /*scale_group_size*/  // not used
                                                                                   false                 /*weight_symmetric_quant*/
                                                                               },
                                                                               // i4 / symmetric/ group size 32 / prefill
                                                                               moe_gemm_test_params{
                                                                                   PHASE::PREFILL /*phase*/,
                                                                                   30,                     /*num_tokens*/
                                                                                   32,                     /*num_total_experts*/
                                                                                   2,                      /*num_experts_per_token*/
                                                                                   3,                      /*num_actually_used_experts*/
                                                                                   128,                    /*hidden_size*/
                                                                                   64,                     /*out_N*/
                                                                                   false,                  /*has_bias*/
                                                                                   cldnn::data_types::f16, /*input_dt*/
                                                                                   cldnn::data_types::i4,  /*weight_dt*/
                                                                                   cldnn::data_types::f16, /*scale_dt*/
                                                                                   32,                     /*scale_group_size*/
                                                                                   true,                   /*weight_symmetric_quant*/
                                                                                   true,                   /*is_pa*/
                                                                               },
                                                                               // i4 / symmetric/ group size 32 / up
                                                                               moe_gemm_test_params{
                                                                                   PHASE::UP /*phase*/,
                                                                                   1,                      /*num_tokens*/
                                                                                   32,                     /*num_total_experts*/
                                                                                   4,                      /*num_experts_per_token*/
                                                                                   4,                      /*num_actually_used_experts*/
                                                                                   128,                    /*hidden_size*/
                                                                                   64,                     /*out_N*/
                                                                                   false,                  /*has_bias*/
                                                                                   cldnn::data_types::f16, /*input_dt*/
                                                                                   cldnn::data_types::i4,  /*weight_dt*/
                                                                                   cldnn::data_types::f16, /*scale_dt*/
                                                                                   32,                     /*scale_group_size*/
                                                                                   true                    /*weight_symmetric_quant*/
                                                                               },
                                                                               // i4 / symmetric/ group size 32 / down
                                                                               moe_gemm_test_params{
                                                                                   PHASE::DOWN /*phase*/,
                                                                                   1,                      /*num_tokens*/
                                                                                   32,                     /*num_total_experts*/
                                                                                   4,                      /*num_experts_per_token*/
                                                                                   4,                      /*num_actually_used_experts*/
                                                                                   128,                    /*hidden_size*/
                                                                                   64,                     /*out_N*/
                                                                                   false,                  /*has_bias*/
                                                                                   cldnn::data_types::f16, /*input_dt*/
                                                                                   cldnn::data_types::i4,  /*weight_dt*/
                                                                                   cldnn::data_types::f16, /*scale_dt*/
                                                                                   32,                     /*scale_group_size*/
                                                                                   true                    /*weight_symmetric_quant*/
                                                                               },
                                                                               // u4 / asymmetric/ per_token / prefill
                                                                               moe_gemm_test_params{
                                                                                   PHASE::PREFILL /*phase*/,
                                                                                   1,                      /*num_tokens*/
                                                                                   32,                     /*num_total_experts*/
                                                                                   4,                      /*num_experts_per_token*/
                                                                                   4,                      /*num_actually_used_experts*/
                                                                                   2880,                    /*hidden_size*/
                                                                                   5760,                     /*out_N*/
                                                                                   false,                  /*has_bias*/
                                                                                   cldnn::data_types::f16, /*input_dt*/
                                                                                   cldnn::data_types::u4,  /*weight_dt*/
                                                                                   cldnn::data_types::f16, /*scale_dt*/
                                                                                   2880,                    /*scale_group_size*/
                                                                                   false                   /*weight_symmetric_quant*/
                                                                               },
                                                                               // u4 / asymmetric/ group size 32 / prefill
                                                                               moe_gemm_test_params{
                                                                                   PHASE::PREFILL /*phase*/,
                                                                                   50,                     /*num_tokens*/
                                                                                   32,                     /*num_total_experts*/
                                                                                   2,                      /*num_experts_per_token*/
                                                                                   15,                     /*num_actually_used_experts*/
                                                                                   128,                    /*hidden_size*/
                                                                                   64,                     /*out_N*/
                                                                                   false,                  /*has_bias*/
                                                                                   cldnn::data_types::f16, /*input_dt*/
                                                                                   cldnn::data_types::u4,  /*weight_dt*/
                                                                                   cldnn::data_types::f32, /*scale_dt*/
                                                                                   32,                     /*scale_group_size*/
                                                                                   false                   /*weight_symmetric_quant*/
                                                                               },
                                                                               // u4 / asymmetric/ group size 32 / up
                                                                               moe_gemm_test_params{
                                                                                   PHASE::UP /*phase*/,
                                                                                   1,                      /*num_tokens*/
                                                                                   32,                     /*num_total_experts*/
                                                                                   4,                      /*num_experts_per_token*/
                                                                                   4,                      /*num_actually_used_experts*/
                                                                                   128,                    /*hidden_size*/
                                                                                   64,                     /*out_N*/
                                                                                   false,                  /*has_bias*/
                                                                                   cldnn::data_types::f16, /*input_dt*/
                                                                                   cldnn::data_types::u4,  /*weight_dt*/
                                                                                   cldnn::data_types::f32, /*scale_dt*/
                                                                                   32,                     /*scale_group_size*/
                                                                                   false                   /*weight_symmetric_quant*/
                                                                               },
                                                                               // u4 / asymmetric/ group size 32 / down
                                                                               moe_gemm_test_params{
                                                                                   PHASE::DOWN /*phase*/,
                                                                                   1,                      /*num_tokens*/
                                                                                   32,                     /*num_total_experts*/
                                                                                   4,                      /*num_experts_per_token*/
                                                                                   4,                      /*num_actually_used_experts*/
                                                                                   128,                    /*hidden_size*/
                                                                                   64,                     /*out_N*/
                                                                                   false,                  /*has_bias*/
                                                                                   cldnn::data_types::f16, /*input_dt*/
                                                                                   cldnn::data_types::u4,  /*weight_dt*/
                                                                                   cldnn::data_types::f32, /*scale_dt*/
                                                                                   32,                     /*scale_group_size*/
                                                                                   false                   /*weight_symmetric_quant*/
                                                                               }
                                                                            }));

TEST(moe_gemm_real_data_i4, basic) {
    cldnn::engine& engine = get_test_engine();
    ov::intel_gpu::op::MOECompressed::Config moe_config;
    moe_config.top_k = 1;
    moe_config.num_expert = 4;
    moe_config.has_batch_dim = false;
    moe_config.hidden_size = 2880;
    moe_config.inter_size = 5760;
    size_t scale_group_size = 32; 
    size_t num_tokens = 6;

    auto num_scale_groups = moe_config.hidden_size / scale_group_size;
    auto input_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(moe_config.hidden_size)};
    auto input_act_layout = layout{input_shape, data_types::f16, format::bfyx};

    auto experts_shape = ov::PartialShape{ov::Dimension(moe_config.num_expert), ov::Dimension(moe_config.inter_size), ov::Dimension(num_scale_groups), ov::Dimension(scale_group_size)};
    auto experts_ids_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto experts_ids_layout = layout{experts_ids_shape, data_types::i32, format::bfyx};

    auto input_offset_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto input_offset_per_expert_layout = layout{input_offset_per_expert_shape, data_types::i32, format::bfyx};

    auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};

    auto input_prim = input_layout("input", input_act_layout);
    auto experts_ids_prim = input_layout("experts_ids", experts_ids_layout);
    auto input_offset_per_expert_prim = input_layout("input_offset_per_expert", input_offset_per_expert_layout);
    auto input_tokens_lens_prim = input_layout("input_tokens_lens", input_tokens_lens_layout);

    topology topo;
    topo.add(input_prim);
    topo.add(experts_ids_prim);
    topo.add(input_offset_per_expert_prim);
    topo.add(input_tokens_lens_prim);


    std::ifstream experts_file("/home/taylor/work/openvino/src/plugins/intel_gpu/tests/unit/test_cases/exp1_weight_i4.txt");
    std::ifstream scales_file("/home/taylor/work/openvino/src/plugins/intel_gpu/tests/unit/test_cases/exp1_scale_i4.txt");
    std::ifstream input_file("/home/taylor/work/openvino/src/plugins/intel_gpu/tests/unit/test_cases/input_token_5_i4.txt");
    std::ifstream bias_file("/home/taylor/work/openvino/src/plugins/intel_gpu/tests/unit/test_cases/exp1_bias_int4.txt");
    // load experts weights
    layout experts_layout = layout{experts_shape, data_types::i4, format::bfyx};
    auto experts_mem = engine.allocate_memory(experts_layout);
    std::vector<uint8_t> experts_data_quant; // (moe_config.num_expert * moe_config.inter_size * moe_config.hidden_size);
    std::vector<uint8_t> experts_data_packed; // (moe_config.num_expert * moe_config.inter_size * moe_config.hidden_size / 2);

    // read weight
    if (!experts_file.is_open()) {
        std::cerr << "Error could not open file" << std::endl;
    }
    int32_t weight_val;
    while (experts_file >> weight_val) {
        experts_data_quant.push_back(static_cast<int8_t>(weight_val));
    }
//    std::srand(std::time(nullptr));
    std::srand(1);
    size_t expert_size = moe_config.inter_size * moe_config.hidden_size;
    for (size_t i = 1; i < moe_config.num_expert; ++i) {
        experts_data_quant.insert(experts_data_quant.end(),
                                   experts_data_quant.begin(),
                                   experts_data_quant.begin() + expert_size);
        for (size_t j = 0; j < expert_size; ++j) {
            std::swap(experts_data_quant[i * expert_size + j], experts_data_quant[rand() % expert_size]);
        }
    }

    if (experts_data_quant.size() != moe_config.num_expert * moe_config.inter_size * moe_config.hidden_size) {
        std::cerr << "Error: experts data size mismatch : " << experts_data_quant.size() << std::endl;
    }

    for (size_t i = 0; i < experts_data_quant.size(); i+=2) {
        int8_t q1 = experts_data_quant[i];
        int8_t q2 = experts_data_quant[i + 1];
        uint8_t packed = ((uint8_t) q2 << 4) | ((uint8_t) q1 & 0x0F);
        experts_data_packed.push_back(packed);
    }
    if (experts_data_packed.size() != moe_config.num_expert * moe_config.inter_size * moe_config.hidden_size / 2) {
        std::cerr << "Error: packed experts data size mismatch" << std::endl;
    }
    set_values(experts_mem, experts_data_packed);
    //read scale
    std::vector<ov::float16> scales_data; // (moe_config.num_expert * moe_config.inter_size * num_scale_groups);
    if (!scales_file.is_open()) {
        std::cerr << "Error could not open file" << std::endl;
    }
    float scale_val;
    while (scales_file >> scale_val) {
        scales_data.push_back(static_cast<ov::float16>(scale_val));
    }

    size_t scales_size_per_expert = moe_config.inter_size * num_scale_groups;
    for (size_t i = 1; i < moe_config.num_expert; ++i) {
        scales_data.insert(scales_data.end(), scales_data.begin(), scales_data.begin() + scales_size_per_expert);
        for (size_t j = 0; j < scales_size_per_expert; ++j) {
            std::swap(scales_data[i * scales_size_per_expert + j], scales_data[rand() % scales_size_per_expert]);
        }
    }

    if (scales_data.size() != moe_config.num_expert * moe_config.inter_size * num_scale_groups) {
        std::cerr << "Error: scales data size mismatch" << std::endl;
    }

    auto scale_shape =
        ov::PartialShape{ov::Dimension(moe_config.num_expert), ov::Dimension(moe_config.inter_size), ov::Dimension(num_scale_groups), ov::Dimension(1)};
    auto scale_layout = layout{scale_shape, data_types::f16, format::bfyx};
    auto scale_mem = engine.allocate_memory(scale_layout);
    set_values(scale_mem, scales_data);

    // read bias
    std::vector<ov::float16> bias_data; // (moe_config.num_expert * moe_config.inter_size * num_scale_groups);
    if (!bias_file.is_open()) {
        std::cerr << "Error could not open file" << std::endl;
    }
    float bias_val;
    while (bias_file >> bias_val) {
        bias_data.push_back(static_cast<ov::float16>(bias_val));
    }
    size_t bias_size_per_expert = moe_config.inter_size;
    for (size_t i = 1; i < moe_config.num_expert; ++i) {
        bias_data.insert(bias_data.end(), bias_data.begin(), bias_data.begin() + bias_size_per_expert);
        for (size_t j = 0; j < bias_size_per_expert; ++j) {
            std::swap(bias_data[i * bias_size_per_expert + j], bias_data[rand() % bias_size_per_expert]);
        }
    }


    if (bias_data.size() != moe_config.num_expert * moe_config.inter_size) {
        std::cerr << "Error: bias data size mismatch" << std::endl;
    }

    auto bias_shape = ov::PartialShape{ov::Dimension(moe_config.num_expert), ov::Dimension(moe_config.inter_size), ov::Dimension(1)};
    auto bias_layout = layout{bias_shape, data_types::f16, format::bfyx};
    auto bias_mem = engine.allocate_memory(bias_layout);
    set_values(bias_mem, bias_data);


    // Create topology
    auto moe_experts_prim = data("moe_experts", experts_mem);
    auto moe_experts_scale_prim = data("moe_experts_scale", scale_mem);
    auto moe_experts_bias_prim = data("moe_experts_bias", bias_mem);

    topo.add(moe_experts_prim);
    topo.add(moe_experts_scale_prim);
    if (getenv("ADD_BIAS") != nullptr) {
        topo.add(moe_experts_bias_prim);
        std::vector<input_info> inputs = {
            input_info("input"),
            input_info("moe_experts"),
            input_info("experts_ids"),
            input_info("input_offset_per_expert"),
            input_info("input_tokens_lens"),
            input_info("moe_experts_bias"),
            input_info("moe_experts_scale"),
        };
        auto moe_gemm_prim = moe_gemm("moe_gemm", inputs, moe_config);
        moe_gemm_prim.has_bias = true;
        topo.add(moe_gemm_prim);
    } else {
        topo.add(moe_experts_bias_prim);
        std::vector<input_info> inputs = {
            input_info("input"),
            input_info("moe_experts"),
            input_info("experts_ids"),
            input_info("input_offset_per_expert"),
            input_info("input_tokens_lens"),
            input_info("moe_experts_scale"),
        };
        auto moe_gemm_prim = moe_gemm("moe_gemm", inputs, moe_config);
        moe_gemm_prim.has_bias = false;
        topo.add(moe_gemm_prim);
    }

    // Construct network
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topo, config);

    // read input
    std::vector<ov::float16> input_data;
    float input_val;
    while (input_file >> input_val) {
        input_data.push_back(static_cast<ov::float16>(input_val));
    }
    for (size_t i = 1; i < num_tokens; ++i) {
        input_data.insert(input_data.end(), input_data.begin(), input_data.begin() + moe_config.hidden_size);
        for (size_t j = 0; j < moe_config.hidden_size; ++j) {
            std::swap(input_data[i * moe_config.hidden_size + j], input_data[rand() % moe_config.hidden_size]);
        }
    }
    if (input_data.size() != num_tokens * moe_config.hidden_size) {
        std::cerr << "Error: input data size mismatch" << std::endl;
    }
    auto input_data_shape = ov::PartialShape{ov::Dimension(num_tokens), ov::Dimension(1), ov::Dimension(moe_config.hidden_size)};
    auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
    auto input_mem = engine.allocate_memory(input_data_layout);
    set_values(input_mem, input_data);

    // set experts info
    std::vector<int32_t> experts_ids_data = {1, 0};
    auto experts_ids_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(experts_ids_data.size()))};
    auto experts_ids_data_layout = layout{experts_ids_data_shape, data_types::i32, format::bfyx};
    auto experts_ids_mem = engine.allocate_memory(experts_ids_data_layout);
    set_values(experts_ids_mem, experts_ids_data);

    std::vector<int32_t> input_offset_per_expert_data = {0, 1};
    auto input_offset_per_expert_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_offset_per_expert_data.size()))};
    auto input_offset_per_expert_data_layout = layout{input_offset_per_expert_data_shape, data_types::i32, format::bfyx};
    auto input_offset_per_expert_mem = engine.allocate_memory(input_offset_per_expert_data_layout);
    set_values(input_offset_per_expert_mem, input_offset_per_expert_data);

    std::vector<int32_t> input_tokens_lens = {1, 1};
    auto input_tokens_lens_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(input_tokens_lens.size()))};
    auto input_tokens_lens_data_layout = layout{input_tokens_lens_data_shape, data_types::i32, format::bfyx};
    auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_data_layout);
    set_values(input_tokens_lens_mem, input_tokens_lens);

    std::cout << "input : " << input_data_layout.to_short_string() << std::endl;
    std::cout << "weight : " << experts_layout.to_short_string() << std::endl;
    std::cout << "scale : " << scale_layout.to_short_string() << std::endl;
    std::cout << "bias : " << bias_layout.to_short_string() << std::endl;
    network.set_input_data("input", input_mem);
    network.set_input_data("experts_ids", experts_ids_mem);
    network.set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
    network.set_input_data("input_tokens_lens", input_tokens_lens_mem);
    auto outputs = network.execute();
    auto output = outputs.at("moe_gemm").get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
    std::cout << "output_ptr[0]" << output_ptr[0] << std::endl;

    std::vector<ov::float16> output_ref(num_tokens * moe_config.inter_size, 0.0f);

    if (getenv("ADD_BIAS") != nullptr) {
        get_reference<uint8_t>(input_data,
                               experts_data_packed,
                               output_ref,
                               experts_ids_data,
                               input_offset_per_expert_data,
                               input_tokens_lens,
                               static_cast<int32_t>(moe_config.inter_size),
                               static_cast<int32_t>(moe_config.hidden_size),
                               scales_data,
                               {},
                               scale_group_size,
                               true,
                               true,
                               cldnn::data_types::i4,
                               bias_data);
    } else {
        get_reference<uint8_t>(input_data,
                               experts_data_packed,
                               output_ref,
                               experts_ids_data,
                               input_offset_per_expert_data,
                               input_tokens_lens,
                               static_cast<int32_t>(moe_config.inter_size),
                               static_cast<int32_t>(moe_config.hidden_size),
                               scales_data,
                               {},
                               scale_group_size,
                               true,
                               true,
                               cldnn::data_types::i4,
                               {});
    }
    std::cout << "output : " << output_ref.size() << std::endl;
    for (size_t i = 0; i < output_ref.size(); i++) {
        auto tolerance = std::max(std::abs(output_ref[i] * 0.01f), 0.01f);
        if (i >= 5760 && i < 5770) {
            std::cout << "out [" << i << "]: " << output_ptr[i] << ", ref : " << output_ref[i] << " diff : " << abs(output_ptr[i] - output_ref[i]) << " bias : " << bias_data[i] << std::endl;
        }
        ASSERT_NEAR(output_ptr[i], output_ref[i], tolerance);
    }
}

TEST(moe_gemm_real_data_u4, basic) {
    ov::intel_gpu::op::MOECompressed::Config moe_config;
    moe_config.top_k = 1;
    moe_config.num_expert = 1;
    moe_config.has_batch_dim = false;
    moe_config.hidden_size = 2880;
    moe_config.inter_size = 5760;

    cldnn::engine& engine = get_test_engine();
    size_t scale_group_size = 2880; 
    auto input_shape = ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), ov::Dimension(moe_config.hidden_size)};
    auto input_act_layout = layout{input_shape, data_types::f16, format::bfyx};

    auto experts_shape = ov::PartialShape{ov::Dimension(1), ov::Dimension(moe_config.inter_size), ov::Dimension(moe_config.hidden_size)};
    auto experts_ids_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto experts_ids_layout = layout{experts_ids_shape, data_types::i32, format::bfyx};

    auto input_offset_per_expert_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto input_offset_per_expert_layout = layout{input_offset_per_expert_shape, data_types::i32, format::bfyx};

    auto input_tokens_lens_shape = ov::PartialShape{ov::Dimension::dynamic()};
    auto input_tokens_lens_layout = layout{input_tokens_lens_shape, data_types::i32, format::bfyx};

    auto input_prim = input_layout("input", input_act_layout);
    auto experts_ids_prim = input_layout("experts_ids", experts_ids_layout);
    auto input_offset_per_expert_prim = input_layout("input_offset_per_expert", input_offset_per_expert_layout);
    auto input_tokens_lens_prim = input_layout("input_tokens_lens", input_tokens_lens_layout);
    topology topo;
    topo.add(input_prim);
    topo.add(experts_ids_prim);
    topo.add(input_offset_per_expert_prim);
    topo.add(input_tokens_lens_prim);
    // load experts weights
    layout experts_layout = layout{experts_shape, data_types::u4, format::bfyx};
    auto experts_mem = engine.allocate_memory(experts_layout);
    std::vector<uint8_t> experts_data_quant; // (moe_config.num_expert * moe_config.inter_size * moe_config.hidden_size);
    std::vector<uint8_t> experts_data_packed; // (moe_config.num_expert * moe_config.inter_size * moe_config.hidden_size / 2);
    // read weight
    std::ifstream experts_file("/home/taylor/work/openvino/src/plugins/intel_gpu/tests/unit/test_cases/exp1_weight_u4.txt");
    std::ifstream scales_file("/home/taylor/work/openvino/src/plugins/intel_gpu/tests/unit/test_cases/exp1_scale_u4.txt");
    std::ifstream zp_file("/home/taylor/work/openvino/src/plugins/intel_gpu/tests/unit/test_cases/exp1_zp_u4.txt");
    std::ifstream input_file("/home/taylor/work/openvino/src/plugins/intel_gpu/tests/unit/test_cases/input_token_5_u4.txt");
    if (!experts_file.is_open()) {
        std::cerr << "Error could not open file" << std::endl;
    }
    uint32_t weight_val;
    while (experts_file >> weight_val) {
        experts_data_quant.push_back(static_cast<int8_t>(weight_val));
    }
    if (experts_data_quant.size() != moe_config.num_expert * moe_config.inter_size * moe_config.hidden_size) {
        std::cerr << "Error: experts data size mismatch : " << experts_data_quant.size() << std::endl;
    }
    for (size_t i = 0; i < experts_data_quant.size(); i+=2) {
        uint8_t q1 = experts_data_quant[i];
        uint8_t q2 = experts_data_quant[i + 1];
        uint8_t packed = ((uint8_t) q2 << 4) | ((uint8_t) q1 & 0x0F);
        experts_data_packed.push_back(packed);
    }
    if (experts_data_packed.size() != moe_config.num_expert * moe_config.inter_size * moe_config.hidden_size / 2) {
        std::cerr << "Error: packed experts data size mismatch" << std::endl;
    }
    set_values(experts_mem, experts_data_packed);
    //read scale
    std::vector<ov::float16> scales_data; // (moe_config.num_expert * moe_config.inter_size * num_scale_groups);
    if (!scales_file.is_open()) {
        std::cerr << "Error could not open file" << std::endl;
    }
    float scale_val;
    while (scales_file >> scale_val) {
        scales_data.push_back(static_cast<ov::float16>(scale_val));
    }
    if (scales_data.size() != moe_config.num_expert * moe_config.inter_size) {
        std::cerr << "Error: scales data size mismatch" << std::endl;
    }

    auto scale_shape =
        ov::PartialShape{ov::Dimension(moe_config.num_expert), ov::Dimension(moe_config.inter_size), ov::Dimension(1)};
    auto scale_layout = layout{scale_shape, data_types::f16, format::bfyx};
    auto scale_mem = engine.allocate_memory(scale_layout);
    set_values(scale_mem, scales_data);

    // read zp
    std::vector<ov::float16> zps_data;  // (moe_config.num_expert * moe_config.inter_size * num_scale_groups);
    if (!zp_file.is_open()) {
        std::cerr << "Error could not open file" << std::endl;
    }
    float zp_val;
    while (zp_file >> zp_val) {
        zps_data.push_back(static_cast<ov::float16>(zp_val));
    }
    if (zps_data.size() != moe_config.num_expert * moe_config.inter_size) {
        std::cerr << "Error: scales data size mismatch" << std::endl;
    }
    auto zp_shape =
        ov::PartialShape{ov::Dimension(moe_config.num_expert), ov::Dimension(moe_config.inter_size), ov::Dimension(1)};
    auto zp_layout = layout{zp_shape, data_types::f16, format::bfyx};
    auto zp_mem = engine.allocate_memory(zp_layout);
    set_values(zp_mem, zps_data);

    // Create topology
    auto moe_experts_prim = data("moe_experts", experts_mem);
    auto moe_experts_scale_prim = data("moe_experts_scale", scale_mem);
    auto moe_experts_zp_prim = data("moe_experts_zp", zp_mem);
    topo.add(moe_experts_prim);
    topo.add(moe_experts_scale_prim);
    topo.add(moe_experts_zp_prim);
    std::vector<input_info> inputs = {
        input_info("input"),
        input_info("moe_experts"),
        input_info("experts_ids"),
        input_info("input_offset_per_expert"),
        input_info("input_tokens_lens"),
        input_info("moe_experts_scale"),
        input_info("moe_experts_zp"),
    };
    auto moe_gemm_prim = moe_gemm("moe_gemm", inputs, moe_config);
    topo.add(moe_gemm_prim);

    // Construct network
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topo, config);

    // read input
    std::vector<ov::float16> input_data;
    float input_val;
    while (input_file >> input_val) {
        input_data.push_back(static_cast<ov::float16>(input_val));
    }
    if (input_data.size() != moe_config.hidden_size) {
        std::cerr << "Error: input data size mismatch" << std::endl;
    }
    auto input_data_shape = ov::PartialShape{ov::Dimension(1), ov::Dimension(1), ov::Dimension(moe_config.hidden_size)};
    auto input_data_layout = layout{input_data_shape, data_types::f16, format::bfyx};
    auto input_mem = engine.allocate_memory(input_data_layout);
    set_values(input_mem, input_data);

    // set experts info
    std::vector<int32_t> experts_ids_data = {0};
    auto experts_ids_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(1))};
    auto experts_ids_data_layout = layout{experts_ids_data_shape, data_types::i32, format::bfyx};
    auto experts_ids_mem = engine.allocate_memory(experts_ids_data_layout);
    set_values(experts_ids_mem, experts_ids_data);

    std::vector<int32_t> input_offset_per_expert_data = {0};
    auto input_offset_per_expert_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(1))};
    auto input_offset_per_expert_data_layout = layout{input_offset_per_expert_data_shape, data_types::i32, format::bfyx};
    auto input_offset_per_expert_mem = engine.allocate_memory(input_offset_per_expert_data_layout);
    set_values(input_offset_per_expert_mem, input_offset_per_expert_data);

    std::vector<int32_t> input_tokens_lens = {1};
    auto input_tokens_lens_data_shape = ov::PartialShape{ov::Dimension(static_cast<int64_t>(1))};
    auto input_tokens_lens_data_layout = layout{input_tokens_lens_data_shape, data_types::i32, format::bfyx};
    auto input_tokens_lens_mem = engine.allocate_memory(input_tokens_lens_data_layout);
    set_values(input_tokens_lens_mem, input_tokens_lens);

    network.set_input_data("input", input_mem);
    network.set_input_data("experts_ids", experts_ids_mem);
    network.set_input_data("input_offset_per_expert", input_offset_per_expert_mem);
    network.set_input_data("input_tokens_lens", input_tokens_lens_mem);
    auto outputs = network.execute();
    auto output = outputs.at("moe_gemm").get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
    std::cout << "output_ptr[0]" << output_ptr[0] << std::endl;

    std::vector<ov::float16> output_ref(moe_config.inter_size, 0.0f);
    get_reference<uint8_t>(input_data,
                           experts_data_packed,
                           output_ref,
                           experts_ids_data,
                           input_offset_per_expert_data,
                           input_tokens_lens,
                           static_cast<int32_t>(moe_config.inter_size),
                           static_cast<int32_t>(moe_config.hidden_size),
                           scales_data,
                           zps_data,
                           scale_group_size,
                           true,
                           false,
                           cldnn::data_types::u4,
                           {});
    for (size_t i = 0; i < output_ref.size(); i++) {
        auto tolerance = std::max(std::abs(output_ref[i] * 0.01f), 0.01f);
        if (i < 10)
            std::cout << "out : " << output_ptr[i] << ", ref : "  << output_ref[i] << " diff : " << abs(output_ptr[i] - output_ref[i]) << std::endl;
        std::cout << "i" << i << std::endl;
        ASSERT_NEAR(output_ptr[i], output_ref[i], tolerance);
    }
}