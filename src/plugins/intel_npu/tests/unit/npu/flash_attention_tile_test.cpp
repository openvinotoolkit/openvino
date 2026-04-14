// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/ops/flash_attention_tile.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <random>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/reference/scaled_dot_product_attention.hpp"

using namespace ov::intel_npu::op;

namespace {

// --- Tensor helpers ---

ov::Tensor make_random_tensor(ov::element::Type type,
                              const ov::Shape& shape,
                              uint32_t seed,
                              float lo = -0.5f,
                              float hi = 0.5f) {
    ov::Tensor tensor(type, shape);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    const auto count = tensor.get_size();

    if (type == ov::element::f32) {
        auto* data = tensor.data<float>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = dist(gen);
        }
    } else if (type == ov::element::f16) {
        auto* data = tensor.data<ov::float16>();
        for (size_t i = 0; i < count; ++i) {
            data[i] = ov::float16(dist(gen));
        }
    }
    return tensor;
}

ov::Tensor make_filled_tensor(ov::element::Type type, const ov::Shape& shape, float value) {
    ov::Tensor tensor(type, shape);
    const auto count = tensor.get_size();
    if (type == ov::element::f32) {
        auto* data = tensor.data<float>();
        std::fill_n(data, count, value);
    } else if (type == ov::element::f16) {
        auto* data = tensor.data<ov::float16>();
        std::fill_n(data, count, ov::float16(value));
    }
    return tensor;
}

ov::Tensor tensor_to_f32(const ov::Tensor& src) {
    if (src.get_element_type() == ov::element::f32) {
        return src;
    }
    ov::Tensor dst(ov::element::f32, src.get_shape());
    const auto* s = src.data<const ov::float16>();
    auto* d = dst.data<float>();
    for (size_t i = 0; i < src.get_size(); ++i) {
        d[i] = static_cast<float>(s[i]);
    }
    return dst;
}

// --- FlashAttentionTile evaluation helper ---

struct FlashAttentionResult {
    ov::Tensor output;      // [B, H_q, L, Ev]
    ov::Tensor output_max;  // [B, H_q, L]
    ov::Tensor output_sum;  // [B, H_q, L]
};

FlashAttentionResult run_flash_attention_tile(const ov::Tensor& query,
                                              const ov::Tensor& key,
                                              const ov::Tensor& value,
                                              const ov::Tensor& running_output,
                                              const ov::Tensor& running_max,
                                              const ov::Tensor& running_sum,
                                              const ov::Tensor* mask,
                                              FlashAttentionTile::Config config) {
    auto elem_type = query.get_element_type();
    auto q_param = std::make_shared<ov::op::v0::Parameter>(elem_type, query.get_shape());
    auto k_param = std::make_shared<ov::op::v0::Parameter>(elem_type, key.get_shape());
    auto v_param = std::make_shared<ov::op::v0::Parameter>(elem_type, value.get_shape());
    auto ro_param = std::make_shared<ov::op::v0::Parameter>(elem_type, running_output.get_shape());
    auto rm_param = std::make_shared<ov::op::v0::Parameter>(elem_type, running_max.get_shape());
    auto rs_param = std::make_shared<ov::op::v0::Parameter>(elem_type, running_sum.get_shape());

    std::shared_ptr<FlashAttentionTile> op;
    if (mask) {
        auto mask_param = std::make_shared<ov::op::v0::Parameter>(mask->get_element_type(), mask->get_shape());
        op = std::make_shared<FlashAttentionTile>(q_param,
                                                  k_param,
                                                  v_param,
                                                  ro_param,
                                                  rm_param,
                                                  rs_param,
                                                  mask_param,
                                                  config);
    } else {
        op = std::make_shared<FlashAttentionTile>(q_param, k_param, v_param, ro_param, rm_param, rs_param, config);
    }

    const auto& out_shape = op->get_output_shape(0);
    const auto& max_shape = op->get_output_shape(1);
    const auto& sum_shape = op->get_output_shape(2);

    ov::TensorVector outputs{ov::Tensor(elem_type, out_shape),
                             ov::Tensor(elem_type, max_shape),
                             ov::Tensor(elem_type, sum_shape)};

    ov::TensorVector inputs{query, key, value, running_output, running_max, running_sum};
    if (mask) {
        inputs.push_back(*mask);
    }

    bool ok = op->evaluate(outputs, inputs);
    EXPECT_TRUE(ok);

    return {outputs[0], outputs[1], outputs[2]};
}

FlashAttentionResult run_flash_attention_tile(const ov::Tensor& query,
                                              const ov::Tensor& key,
                                              const ov::Tensor& value,
                                              const ov::Tensor* mask,
                                              FlashAttentionTile::Config config) {
    auto elem_type = query.get_element_type();
    const auto& q_shape = query.get_shape();
    // [B, H_q, L, E] -> output [B, H_q, L, Ev], max/sum [B, H_q, L]
    auto B = q_shape[0];
    auto H_q = q_shape[1];
    auto L = q_shape[2];
    auto Ev = value.get_shape()[3];

    ov::Shape out_shape{B, H_q, L, Ev};
    ov::Shape state_shape{B, H_q, L};

    auto running_output = make_filled_tensor(elem_type, out_shape, 0.0f);
    auto running_max = make_filled_tensor(elem_type, state_shape, -std::numeric_limits<float>::infinity());
    auto running_sum = make_filled_tensor(elem_type, state_shape, 0.0f);

    return run_flash_attention_tile(query, key, value, running_output, running_max, running_sum, mask, config);
}

// --- Normalize running output by running sum ---

std::vector<float> normalize_output(const ov::Tensor& running_output_t, const ov::Tensor& running_sum_t) {
    auto ro = tensor_to_f32(running_output_t);
    auto rs = tensor_to_f32(running_sum_t);

    const auto& shape = ro.get_shape();  // [B, H_q, L, Ev]
    auto B = shape[0];
    auto H = shape[1];
    auto L = shape[2];
    auto Ev = shape[3];

    const auto* out_data = ro.data<const float>();
    const auto* sum_data = rs.data<const float>();

    std::vector<float> result(ro.get_size());
    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t l = 0; l < L; ++l) {
                auto sum_val = sum_data[b * H * L + h * L + l];
                auto inv_sum = (sum_val != 0.0f) ? (1.0f / sum_val) : 0.0f;
                for (size_t ev = 0; ev < Ev; ++ev) {
                    result[b * H * L * Ev + h * L * Ev + l * Ev + ev] =
                        out_data[b * H * L * Ev + h * L * Ev + l * Ev + ev] * inv_sum;
                }
            }
        }
    }
    return result;
}

// --- SDPA reference ---

std::vector<float> run_sdpa_reference(const ov::Tensor& query_t,
                                      const ov::Tensor& key_t,
                                      const ov::Tensor& value_t,
                                      const float* mask_data,
                                      const ov::Shape& mask_shape) {
    auto query = tensor_to_f32(query_t);
    auto key = tensor_to_f32(key_t);
    auto value = tensor_to_f32(value_t);

    auto output_shape = query.get_shape();
    output_shape.back() = value.get_shape().back();  // [B, H, L, Ev]

    std::vector<float> output(ov::shape_size(output_shape), 0.0f);
    const float scale = 1.0f;

    ov::reference::scaled_dot_product_attention<float, float>(query.data<const float>(),
                                                              key.data<const float>(),
                                                              value.data<const float>(),
                                                              mask_data,
                                                              &scale,
                                                              nullptr,  // no sink
                                                              output.data(),
                                                              false,
                                                              query.get_shape(),
                                                              key.get_shape(),
                                                              value.get_shape(),
                                                              mask_shape,
                                                              ov::Shape{},
                                                              output_shape);

    return output;
}

// --- GQA helper: expand KV from [B, H_kv, S, E] to [B, H_q, S, E] ---

ov::Tensor expand_kv_for_gqa(const ov::Tensor& kv, size_t H_q) {
    auto kv_f32 = tensor_to_f32(kv);
    const auto& shape = kv_f32.get_shape();
    auto B = shape[0];
    auto H_kv = shape[1];
    auto S = shape[2];
    auto dim = shape[3];

    auto group_size = H_q / H_kv;
    ov::Shape expanded_shape{B, H_q, S, dim};
    ov::Tensor expanded(ov::element::f32, expanded_shape);

    const auto* src = kv_f32.data<const float>();
    auto* dst = expanded.data<float>();

    for (size_t b = 0; b < B; ++b) {
        for (size_t h_q = 0; h_q < H_q; ++h_q) {
            auto kv_h = h_q / group_size;
            const auto* src_head = src + (b * H_kv + kv_h) * S * dim;
            auto* dst_head = dst + (b * H_q + h_q) * S * dim;
            std::copy_n(src_head, S * dim, dst_head);
        }
    }
    return expanded;
}

// --- Comparison ---

void compare_vectors(const std::vector<float>& actual,
                     const std::vector<float>& expected,
                     float atol,
                     const std::string& context = "") {
    ASSERT_EQ(actual.size(), expected.size()) << context;
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], atol)
            << context << " mismatch at index " << i << ": actual=" << actual[i] << " expected=" << expected[i];
    }
}

// --- Slice tensor along S dimension: [B, H, S, D] -> [B, H, s1:s2, D] ---

ov::Tensor slice_s_dim(const ov::Tensor& tensor, size_t s_begin, size_t s_end) {
    auto t = tensor_to_f32(tensor);
    const auto& shape = t.get_shape();
    auto B = shape[0];
    auto H = shape[1];
    auto S = shape[2];
    auto D = shape[3];
    auto new_S = s_end - s_begin;

    ov::Tensor result(ov::element::f32, {B, H, new_S, D});
    const auto* src = t.data<const float>();
    auto* dst = result.data<float>();

    for (size_t b = 0; b < B; ++b) {
        for (size_t h = 0; h < H; ++h) {
            for (size_t s = 0; s < new_S; ++s) {
                std::copy_n(src + ((b * H + h) * S + (s_begin + s)) * D, D, dst + ((b * H + h) * new_S + s) * D);
            }
        }
    }
    return result;
}

// Slice mask along last dim (S): [..., L, S] -> [..., L, s1:s2]
ov::Tensor slice_mask_s_dim(const ov::Tensor& mask, size_t s_begin, size_t s_end) {
    auto m = tensor_to_f32(mask);
    const auto& shape = m.get_shape();
    auto rank = shape.size();
    auto S = shape[rank - 1];
    auto L = shape[rank - 2];
    auto new_S = s_end - s_begin;

    size_t outer = 1;
    for (size_t i = 0; i + 2 < rank; ++i) {
        outer *= shape[i];
    }

    ov::Shape new_shape = shape;
    new_shape[rank - 1] = new_S;

    ov::Tensor result(ov::element::f32, new_shape);
    const auto* src = m.data<const float>();
    auto* dst = result.data<float>();

    for (size_t o = 0; o < outer; ++o) {
        for (size_t l = 0; l < L; ++l) {
            std::copy_n(src + (o * L + l) * S + s_begin, new_S, dst + (o * L + l) * new_S);
        }
    }
    return result;
}

}  // namespace

// ============================================================================
// Test fixture
// ============================================================================

class FlashAttentionTileTest : public ::testing::Test {};

// ============================================================================
// 1. Basic Correctness (single tile, is_head=true, is_tail=false, then normalize)
// ============================================================================

TEST_F(FlashAttentionTileTest, SingleTile_SmallDims) {
    // B=1, H=1, L=2, S=3, E=4, Ev=4
    auto Q = make_random_tensor(ov::element::f32, {1, 1, 2, 4}, 42);
    auto K = make_random_tensor(ov::element::f32, {1, 1, 3, 4}, 43);
    auto V = make_random_tensor(ov::element::f32, {1, 1, 3, 4}, 44);

    auto result = run_flash_attention_tile(Q, K, V, nullptr, {true, false});
    auto actual = normalize_output(result.output, result.output_sum);
    auto expected = run_sdpa_reference(Q, K, V, nullptr, {});

    compare_vectors(actual, expected, 1e-5f, "SingleTile_SmallDims");
}

TEST_F(FlashAttentionTileTest, SingleTile_LargerBatch) {
    // B=2, H=2, L=4, S=8, E=16, Ev=16
    auto Q = make_random_tensor(ov::element::f32, {2, 2, 4, 16}, 100);
    auto K = make_random_tensor(ov::element::f32, {2, 2, 8, 16}, 101);
    auto V = make_random_tensor(ov::element::f32, {2, 2, 8, 16}, 102);

    auto result = run_flash_attention_tile(Q, K, V, nullptr, {true, false});
    auto actual = normalize_output(result.output, result.output_sum);
    auto expected = run_sdpa_reference(Q, K, V, nullptr, {});

    compare_vectors(actual, expected, 1e-5f, "SingleTile_LargerBatch");
}

TEST_F(FlashAttentionTileTest, SingleTile_AsymmetricEv) {
    // B=1, H=1, L=2, S=3, E=4, Ev=8
    auto Q = make_random_tensor(ov::element::f32, {1, 1, 2, 4}, 200);
    auto K = make_random_tensor(ov::element::f32, {1, 1, 3, 4}, 201);
    auto V = make_random_tensor(ov::element::f32, {1, 1, 3, 8}, 202);

    auto result = run_flash_attention_tile(Q, K, V, nullptr, {true, false});
    auto actual = normalize_output(result.output, result.output_sum);
    auto expected = run_sdpa_reference(Q, K, V, nullptr, {});

    compare_vectors(actual, expected, 1e-5f, "SingleTile_AsymmetricEv");
}

// ============================================================================
// 2. Multi-Tile
// ============================================================================

TEST_F(FlashAttentionTileTest, TwoTiles_MatchesSDPA) {
    // B=1, H=1, L=2, S=6, E=4, Ev=4 -- split S into 3+3
    auto Q = make_random_tensor(ov::element::f32, {1, 1, 2, 4}, 300);
    auto K = make_random_tensor(ov::element::f32, {1, 1, 6, 4}, 301);
    auto V = make_random_tensor(ov::element::f32, {1, 1, 6, 4}, 302);

    auto K1 = slice_s_dim(K, 0, 3);
    auto V1 = slice_s_dim(V, 0, 3);
    auto K2 = slice_s_dim(K, 3, 6);
    auto V2 = slice_s_dim(V, 3, 6);

    // Tile 1: head
    auto r1 = run_flash_attention_tile(Q, K1, V1, nullptr, {true, false});
    // Tile 2: uses running state from tile 1
    auto r2 = run_flash_attention_tile(Q, K2, V2, r1.output, r1.output_max, r1.output_sum, nullptr, {false, false});

    auto actual = normalize_output(r2.output, r2.output_sum);
    auto expected = run_sdpa_reference(Q, K, V, nullptr, {});

    compare_vectors(actual, expected, 1e-5f, "TwoTiles_MatchesSDPA");
}

TEST_F(FlashAttentionTileTest, ThreeTiles_MatchesSDPA) {
    // B=1, H=2, L=3, S=9, E=4, Ev=4 -- split S into 3+3+3
    auto Q = make_random_tensor(ov::element::f32, {1, 2, 3, 4}, 400);
    auto K = make_random_tensor(ov::element::f32, {1, 2, 9, 4}, 401);
    auto V = make_random_tensor(ov::element::f32, {1, 2, 9, 4}, 402);

    auto K1 = slice_s_dim(K, 0, 3);
    auto V1 = slice_s_dim(V, 0, 3);
    auto K2 = slice_s_dim(K, 3, 6);
    auto V2 = slice_s_dim(V, 3, 6);
    auto K3 = slice_s_dim(K, 6, 9);
    auto V3 = slice_s_dim(V, 6, 9);

    auto r1 = run_flash_attention_tile(Q, K1, V1, nullptr, {true, false});
    auto r2 = run_flash_attention_tile(Q, K2, V2, r1.output, r1.output_max, r1.output_sum, nullptr, {false, false});
    auto r3 = run_flash_attention_tile(Q, K3, V3, r2.output, r2.output_max, r2.output_sum, nullptr, {false, false});

    auto actual = normalize_output(r3.output, r3.output_sum);
    auto expected = run_sdpa_reference(Q, K, V, nullptr, {});

    compare_vectors(actual, expected, 1e-5f, "ThreeTiles_MatchesSDPA");
}

// ============================================================================
// 3. GQA (Grouped Query Attention)
// ============================================================================

TEST_F(FlashAttentionTileTest, GQA_TwoToOne) {
    // H_q=4, H_kv=2, group_size=2
    auto Q = make_random_tensor(ov::element::f32, {1, 4, 2, 4}, 500);
    auto K = make_random_tensor(ov::element::f32, {1, 2, 3, 4}, 501);
    auto V = make_random_tensor(ov::element::f32, {1, 2, 3, 4}, 502);

    auto result = run_flash_attention_tile(Q, K, V, nullptr, {true, false});
    auto actual = normalize_output(result.output, result.output_sum);

    // Expand K/V to H_q=4 for SDPA reference
    auto K_exp = expand_kv_for_gqa(K, 4);
    auto V_exp = expand_kv_for_gqa(V, 4);
    auto expected = run_sdpa_reference(Q, K_exp, V_exp, nullptr, {});

    compare_vectors(actual, expected, 1e-5f, "GQA_TwoToOne");
}

TEST_F(FlashAttentionTileTest, GQA_FourToOne) {
    // H_q=4, H_kv=1
    auto Q = make_random_tensor(ov::element::f32, {1, 4, 2, 4}, 510);
    auto K = make_random_tensor(ov::element::f32, {1, 1, 3, 4}, 511);
    auto V = make_random_tensor(ov::element::f32, {1, 1, 3, 4}, 512);

    auto result = run_flash_attention_tile(Q, K, V, nullptr, {true, false});
    auto actual = normalize_output(result.output, result.output_sum);

    auto K_exp = expand_kv_for_gqa(K, 4);
    auto V_exp = expand_kv_for_gqa(V, 4);
    auto expected = run_sdpa_reference(Q, K_exp, V_exp, nullptr, {});

    compare_vectors(actual, expected, 1e-5f, "GQA_FourToOne");
}

// ============================================================================
// 4. Attention Mask
// ============================================================================

TEST_F(FlashAttentionTileTest, Mask_Additive) {
    auto Q = make_random_tensor(ov::element::f32, {1, 1, 2, 4}, 600);
    auto K = make_random_tensor(ov::element::f32, {1, 1, 3, 4}, 601);
    auto V = make_random_tensor(ov::element::f32, {1, 1, 3, 4}, 602);
    auto mask = make_random_tensor(ov::element::f32, {1, 1, 2, 3}, 603, -1.0f, 0.0f);

    auto result = run_flash_attention_tile(Q, K, V, &mask, {true, false});
    auto actual = normalize_output(result.output, result.output_sum);

    auto mask_f32 = tensor_to_f32(mask);
    auto expected = run_sdpa_reference(Q, K, V, mask_f32.data<const float>(), mask.get_shape());

    compare_vectors(actual, expected, 1e-5f, "Mask_Additive");
}

TEST_F(FlashAttentionTileTest, Mask_CausalPattern) {
    // L=S=4, causal: 0 if s<=l, -inf otherwise
    size_t L = 4, S = 4;
    auto Q = make_random_tensor(ov::element::f32, {1, 1, L, 4}, 610);
    auto K = make_random_tensor(ov::element::f32, {1, 1, S, 4}, 611);
    auto V = make_random_tensor(ov::element::f32, {1, 1, S, 4}, 612);

    ov::Shape mask_shape{1, 1, L, S};
    auto mask = ov::Tensor(ov::element::f32, mask_shape);
    auto* mask_data = mask.data<float>();
    for (size_t l = 0; l < L; ++l) {
        for (size_t s = 0; s < S; ++s) {
            mask_data[l * S + s] = (s <= l) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }

    auto result = run_flash_attention_tile(Q, K, V, &mask, {true, false});
    auto actual = normalize_output(result.output, result.output_sum);
    auto expected = run_sdpa_reference(Q, K, V, mask_data, mask_shape);

    compare_vectors(actual, expected, 1e-5f, "Mask_CausalPattern");
}

TEST_F(FlashAttentionTileTest, Mask_Boolean) {
    auto Q = make_random_tensor(ov::element::f32, {1, 1, 2, 4}, 620);
    auto K = make_random_tensor(ov::element::f32, {1, 1, 3, 4}, 621);
    auto V = make_random_tensor(ov::element::f32, {1, 1, 3, 4}, 622);

    // Boolean mask: true=attend, false=mask out
    ov::Shape mask_shape{1, 1, 2, 3};
    auto bool_mask = ov::Tensor(ov::element::boolean, mask_shape);
    auto* bdata = bool_mask.data<char>();
    // Row 0: attend to all 3; Row 1: attend to first 2 only
    bdata[0] = 1;
    bdata[1] = 1;
    bdata[2] = 1;
    bdata[3] = 1;
    bdata[4] = 1;
    bdata[5] = 0;

    auto result = run_flash_attention_tile(Q, K, V, &bool_mask, {true, false});
    auto actual = normalize_output(result.output, result.output_sum);

    // Build equivalent float mask for SDPA
    auto float_mask = ov::Tensor(ov::element::f32, mask_shape);
    auto* fdata = float_mask.data<float>();
    for (size_t i = 0; i < 6; ++i) {
        fdata[i] = bdata[i] ? 0.0f : -std::numeric_limits<float>::infinity();
    }

    auto expected = run_sdpa_reference(Q, K, V, fdata, mask_shape);

    compare_vectors(actual, expected, 1e-5f, "Mask_Boolean");
}

// ============================================================================
// 5. Mask Broadcasting
// ============================================================================

TEST_F(FlashAttentionTileTest, Mask_2D) {
    // Mask [L, S] shared across B and H
    auto Q = make_random_tensor(ov::element::f32, {2, 2, 3, 4}, 700);
    auto K = make_random_tensor(ov::element::f32, {2, 2, 5, 4}, 701);
    auto V = make_random_tensor(ov::element::f32, {2, 2, 5, 4}, 702);
    auto mask = make_random_tensor(ov::element::f32, {3, 5}, 703, -1.0f, 0.0f);

    auto result = run_flash_attention_tile(Q, K, V, &mask, {true, false});
    auto actual = normalize_output(result.output, result.output_sum);

    auto mask_f32 = tensor_to_f32(mask);
    auto expected = run_sdpa_reference(Q, K, V, mask_f32.data<const float>(), mask.get_shape());

    compare_vectors(actual, expected, 1e-5f, "Mask_2D");
}

TEST_F(FlashAttentionTileTest, Mask_3D_PerHead) {
    // Mask [H, L, S] -- different per head, shared across batch
    auto Q = make_random_tensor(ov::element::f32, {2, 2, 3, 4}, 710);
    auto K = make_random_tensor(ov::element::f32, {2, 2, 5, 4}, 711);
    auto V = make_random_tensor(ov::element::f32, {2, 2, 5, 4}, 712);
    auto mask = make_random_tensor(ov::element::f32, {2, 3, 5}, 713, -1.0f, 0.0f);

    auto result = run_flash_attention_tile(Q, K, V, &mask, {true, false});
    auto actual = normalize_output(result.output, result.output_sum);

    auto mask_f32 = tensor_to_f32(mask);
    auto expected = run_sdpa_reference(Q, K, V, mask_f32.data<const float>(), mask.get_shape());

    compare_vectors(actual, expected, 1e-5f, "Mask_3D_PerHead");
}

TEST_F(FlashAttentionTileTest, Mask_4D_BroadcastBatch) {
    // Mask [1, H, L, S] with B=2 -- batch dim broadcast
    auto Q = make_random_tensor(ov::element::f32, {2, 2, 3, 4}, 720);
    auto K = make_random_tensor(ov::element::f32, {2, 2, 5, 4}, 721);
    auto V = make_random_tensor(ov::element::f32, {2, 2, 5, 4}, 722);
    auto mask = make_random_tensor(ov::element::f32, {1, 2, 3, 5}, 723, -1.0f, 0.0f);

    auto result = run_flash_attention_tile(Q, K, V, &mask, {true, false});
    auto actual = normalize_output(result.output, result.output_sum);

    auto mask_f32 = tensor_to_f32(mask);
    auto expected = run_sdpa_reference(Q, K, V, mask_f32.data<const float>(), mask.get_shape());

    compare_vectors(actual, expected, 1e-5f, "Mask_4D_BroadcastBatch");
}

// ============================================================================
// 6. is_tail=true matches manual normalization
// ============================================================================

TEST_F(FlashAttentionTileTest, IsTail_MatchesManualNormalization) {
    auto Q = make_random_tensor(ov::element::f32, {1, 2, 3, 4}, 800);
    auto K = make_random_tensor(ov::element::f32, {1, 2, 5, 4}, 801);
    auto V = make_random_tensor(ov::element::f32, {1, 2, 5, 4}, 802);

    // Run with is_tail=false, then manually normalize
    auto r_no_tail = run_flash_attention_tile(Q, K, V, nullptr, {true, false});
    auto manual_normalized = normalize_output(r_no_tail.output, r_no_tail.output_sum);

    // Run with is_tail=true (internally normalized)
    auto r_tail = run_flash_attention_tile(Q, K, V, nullptr, {true, true});
    auto tail_output = tensor_to_f32(r_tail.output);

    std::vector<float> tail_vec(tail_output.data<const float>(),
                                tail_output.data<const float>() + tail_output.get_size());

    compare_vectors(tail_vec, manual_normalized, 1e-6f, "IsTail_MatchesManualNormalization");
}

// ============================================================================
// 7. f16 Support
// ============================================================================

TEST_F(FlashAttentionTileTest, F16_MatchesF32) {
    auto Q_f32 = make_random_tensor(ov::element::f32, {1, 2, 3, 8}, 900);
    auto K_f32 = make_random_tensor(ov::element::f32, {1, 2, 4, 8}, 901);
    auto V_f32 = make_random_tensor(ov::element::f32, {1, 2, 4, 8}, 902);

    // Create f16 versions (with inherent quantization)
    auto to_f16 = [](const ov::Tensor& src) {
        ov::Tensor dst(ov::element::f16, src.get_shape());
        const auto* s = src.data<const float>();
        auto* d = dst.data<ov::float16>();
        for (size_t i = 0; i < src.get_size(); ++i) {
            d[i] = ov::float16(s[i]);
        }
        return dst;
    };

    auto Q_f16 = to_f16(Q_f32);
    auto K_f16 = to_f16(K_f32);
    auto V_f16 = to_f16(V_f32);

    auto result_f16 = run_flash_attention_tile(Q_f16, K_f16, V_f16, nullptr, {true, false});
    auto actual = normalize_output(result_f16.output, result_f16.output_sum);

    // Use the f16-quantized values as f32 inputs to SDPA for fair comparison
    auto Q_roundtrip = tensor_to_f32(Q_f16);
    auto K_roundtrip = tensor_to_f32(K_f16);
    auto V_roundtrip = tensor_to_f32(V_f16);
    auto expected = run_sdpa_reference(Q_roundtrip, K_roundtrip, V_roundtrip, nullptr, {});

    compare_vectors(actual, expected, 5e-3f, "F16_MatchesF32");
}

// ============================================================================
// 8. Edge Cases
// ============================================================================

TEST_F(FlashAttentionTileTest, EdgeCase_S1) {
    // S=1: softmax of a single element = 1.0, so output should be V
    auto Q = make_random_tensor(ov::element::f32, {1, 1, 2, 4}, 1000);
    auto K = make_random_tensor(ov::element::f32, {1, 1, 1, 4}, 1001);
    auto V = make_random_tensor(ov::element::f32, {1, 1, 1, 4}, 1002);

    auto result = run_flash_attention_tile(Q, K, V, nullptr, {true, false});
    auto actual = normalize_output(result.output, result.output_sum);

    // With S=1, softmax weight is 1.0, so output[l] = V[0] for all l
    const auto* v_data = V.data<const float>();
    std::vector<float> expected(2 * 4);  // L=2, Ev=4
    for (size_t l = 0; l < 2; ++l) {
        for (size_t ev = 0; ev < 4; ++ev) {
            expected[l * 4 + ev] = v_data[ev];
        }
    }

    compare_vectors(actual, expected, 1e-5f, "EdgeCase_S1");
}

TEST_F(FlashAttentionTileTest, EdgeCase_L1) {
    auto Q = make_random_tensor(ov::element::f32, {1, 1, 1, 4}, 1010);
    auto K = make_random_tensor(ov::element::f32, {1, 1, 5, 4}, 1011);
    auto V = make_random_tensor(ov::element::f32, {1, 1, 5, 4}, 1012);

    auto result = run_flash_attention_tile(Q, K, V, nullptr, {true, false});
    auto actual = normalize_output(result.output, result.output_sum);
    auto expected = run_sdpa_reference(Q, K, V, nullptr, {});

    compare_vectors(actual, expected, 1e-5f, "EdgeCase_L1");
}

TEST_F(FlashAttentionTileTest, EdgeCase_LargeValues) {
    // Large values test numerical stability of online softmax
    auto Q = make_random_tensor(ov::element::f32, {1, 1, 2, 4}, 1020, 50.0f, 100.0f);
    auto K = make_random_tensor(ov::element::f32, {1, 1, 3, 4}, 1021, 50.0f, 100.0f);
    auto V = make_random_tensor(ov::element::f32, {1, 1, 3, 4}, 1022, -1.0f, 1.0f);

    auto result = run_flash_attention_tile(Q, K, V, nullptr, {true, false});
    auto actual = normalize_output(result.output, result.output_sum);
    auto expected = run_sdpa_reference(Q, K, V, nullptr, {});

    compare_vectors(actual, expected, 1e-4f, "EdgeCase_LargeValues");
}

// ============================================================================
// Multi-tile with mask
// ============================================================================

TEST_F(FlashAttentionTileTest, TwoTiles_WithMask_MatchesSDPA) {
    // B=1, H=1, L=2, S=6, E=4, Ev=4 with mask [1,1,2,6]
    auto Q = make_random_tensor(ov::element::f32, {1, 1, 2, 4}, 1100);
    auto K = make_random_tensor(ov::element::f32, {1, 1, 6, 4}, 1101);
    auto V = make_random_tensor(ov::element::f32, {1, 1, 6, 4}, 1102);
    auto mask = make_random_tensor(ov::element::f32, {1, 1, 2, 6}, 1103, -2.0f, 0.0f);

    auto K1 = slice_s_dim(K, 0, 3);
    auto V1 = slice_s_dim(V, 0, 3);
    auto K2 = slice_s_dim(K, 3, 6);
    auto V2 = slice_s_dim(V, 3, 6);
    auto mask1 = slice_mask_s_dim(mask, 0, 3);
    auto mask2 = slice_mask_s_dim(mask, 3, 6);

    auto r1 = run_flash_attention_tile(Q, K1, V1, &mask1, {true, false});
    auto r2 = run_flash_attention_tile(Q, K2, V2, r1.output, r1.output_max, r1.output_sum, &mask2, {false, false});

    auto actual = normalize_output(r2.output, r2.output_sum);

    auto mask_f32 = tensor_to_f32(mask);
    auto expected = run_sdpa_reference(Q, K, V, mask_f32.data<const float>(), mask.get_shape());

    compare_vectors(actual, expected, 1e-5f, "TwoTiles_WithMask_MatchesSDPA");
}
