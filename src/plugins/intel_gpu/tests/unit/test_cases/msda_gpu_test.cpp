// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/msda.hpp>
#include <vector>

#include "msda_inst.h"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

struct MsdaCase {
    size_t batch, queries, heads, embed, levels, points;
    std::vector<std::pair<int, int>> spatial;  // (h, w) per level
    size_t keys() const {
        size_t k = 0;
        for (const auto& hw : spatial)
            k += static_cast<size_t>(hw.first) * hw.second;
        return k;
    }
};

// Host model of msda_opt.cl: per (b, q, h, d) accumulate over levels and
// points the attention-weighted bilinear sample of value at the [0,1]
// location, with zero padding outside the feature map and the kernel's
// sampling convention w_im = loc_x * W - 0.5, h_im = loc_y * H - 0.5.
std::vector<float> msda_reference(const MsdaCase& c, const std::vector<float>& value, const std::vector<float>& loc, const std::vector<float>& weights) {
    const size_t B = c.batch, Q = c.queries, H = c.heads, D = c.embed, L = c.levels, P = c.points;
    const size_t K = c.keys();
    std::vector<int32_t> starts;
    int32_t position = 0;
    for (const auto& hw : c.spatial) {
        starts.push_back(position);
        position += hw.first * hw.second;
    }
    std::vector<float> out(B * Q * H * D, 0.f);
    for (size_t b = 0; b < B; ++b) {
        for (size_t q = 0; q < Q; ++q) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t d = 0; d < D; ++d) {
                    float col = 0.f;
                    for (size_t l = 0; l < L; ++l) {
                        const int Hl = c.spatial[l].first, Wl = c.spatial[l].second;
                        for (size_t p = 0; p < P; ++p) {
                            const size_t widx = (((b * Q + q) * H + h) * L + l) * P + p;
                            const size_t lidx = 2 * widx;
                            const float loc_w = loc[lidx], loc_h = loc[lidx + 1];
                            const float weight = weights[widx];
                            const float h_im = loc_h * Hl - 0.5f;
                            const float w_im = loc_w * Wl - 0.5f;
                            if (h_im > -1.f && w_im > -1.f && h_im < Hl && w_im < Wl) {
                                const int h_low = static_cast<int>(std::floor(h_im));
                                const int w_low = static_cast<int>(std::floor(w_im));
                                const int h_high = h_low + 1, w_high = w_low + 1;
                                const float lh = h_im - h_low, lw = w_im - w_low;
                                const float hh = 1.f - lh, hw = 1.f - lw;
                                const auto at = [&](int y, int x) -> float {
                                    if (y < 0 || x < 0 || y >= Hl || x >= Wl)
                                        return 0.f;
                                    const size_t key = starts[l] + static_cast<size_t>(y) * Wl + x;
                                    return value[((b * K + key) * H + h) * D + d];
                                };
                                col += (hh * hw * at(h_low, w_low) + hh * lw * at(h_low, w_high) + lh * hw * at(h_high, w_low) + lh * lw * at(h_high, w_high)) *
                                       weight;
                            }
                        }
                    }
                    out[((b * Q + q) * H + h) * D + d] = col;
                }
            }
        }
    }
    return out;
}

void run_msda_case(const MsdaCase& c) {
    auto& engine = get_test_engine();
    const size_t K = c.keys();

    std::vector<float> value_vec(c.batch * K * c.heads * c.embed);
    for (size_t i = 0; i < value_vec.size(); ++i)
        value_vec[i] = 0.125f * static_cast<float>(i % 97) - 3.0f;
    std::vector<float> loc_vec(c.batch * c.queries * c.heads * c.levels * c.points * 2);
    for (size_t i = 0; i < loc_vec.size(); ++i)
        loc_vec[i] = 0.05f + 0.9f * static_cast<float>(i % 13) / 13.f;
    std::vector<float> weight_vec(c.batch * c.queries * c.heads * c.levels * c.points);
    for (size_t i = 0; i < weight_vec.size(); ++i)
        weight_vec[i] = static_cast<float>((i % 7) + 1) / 8.f;

    std::vector<int32_t> spatial_flat;
    for (const auto& hw : c.spatial)
        spatial_flat.insert(spatial_flat.end(), {hw.first, hw.second});
    std::vector<int32_t> starts;
    int32_t position = 0;
    for (const auto& hw : c.spatial) {
        starts.push_back(position);
        position += hw.first * hw.second;
    }

    auto data_value =
        engine.allocate_memory({ov::PartialShape{int64_t(c.batch), int64_t(K), int64_t(c.heads), int64_t(c.embed)}, data_types::f32, format::bfyx});
    auto data_spatial_shapes = engine.allocate_memory({ov::PartialShape{int64_t(c.levels), 2, 1, 1}, data_types::i32, format::bfyx});
    auto data_level_start = engine.allocate_memory({ov::PartialShape{int64_t(c.levels), 1, 1, 1}, data_types::i32, format::bfyx});
    auto data_sampling_loc = engine.allocate_memory(
        {ov::PartialShape{int64_t(c.batch), int64_t(c.queries), int64_t(c.heads * c.levels * c.points * 2), 1}, data_types::f32, format::bfyx});
    auto data_attn_weight = engine.allocate_memory(
        {ov::PartialShape{int64_t(c.batch), int64_t(c.queries), int64_t(c.heads * c.levels * c.points), 1}, data_types::f32, format::bfyx});
    set_values(data_value, value_vec);
    set_values(data_spatial_shapes, spatial_flat);
    set_values(data_level_start, starts);
    set_values(data_sampling_loc, loc_vec);
    set_values(data_attn_weight, weight_vec);

    topology topology;
    topology.add(input_layout("data_value", data_value->get_layout()));
    topology.add(input_layout("data_spatial_shapes", data_spatial_shapes->get_layout()));
    topology.add(input_layout("data_level_start_idx", data_level_start->get_layout()));
    topology.add(input_layout("data_sampling_loc", data_sampling_loc->get_layout()));
    topology.add(input_layout("data_attn_weight", data_attn_weight->get_layout()));
    topology.add(msda("msda",
                      {input_info("data_value"),
                       input_info("data_spatial_shapes"),
                       input_info("data_level_start_idx"),
                       input_info("data_sampling_loc"),
                       input_info("data_attn_weight")}));

    auto network = cldnn::network(engine, topology, get_test_default_config(engine));
    network.set_input_data("data_value", data_value);
    network.set_input_data("data_spatial_shapes", data_spatial_shapes);
    network.set_input_data("data_level_start_idx", data_level_start);
    network.set_input_data("data_sampling_loc", data_sampling_loc);
    network.set_input_data("data_attn_weight", data_attn_weight);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.count("msda"), size_t(1));

    auto output_memory = outputs.at("msda").get_memory();
    cldnn::mem_lock<float> output_ptr(output_memory, get_test_stream());
    const auto ref = msda_reference(c, value_vec, loc_vec, weight_vec);
    ASSERT_EQ(output_memory->count(), ref.size());
    for (size_t i = 0; i < ref.size(); ++i)
        ASSERT_NEAR(output_ptr[i], ref[i], 2e-4f) << "index " << i;
}

}  // namespace

TEST(msda_gpu, static_reference_two_levels) {
    MsdaCase c{1, 2, 2, 4, 2, 2, {{2, 2}, {2, 4}}};
    run_msda_case(c);
}

TEST(msda_gpu, static_reference_single_level_three_points) {
    MsdaCase c{2, 3, 1, 8, 1, 3, {{5, 3}}};
    run_msda_case(c);
}

TEST(msda_gpu, static_reference_interior_and_border) {
    // Locations at 0 and 1 exercise the map borders of the bilinear sampler.
    MsdaCase c{1, 4, 1, 2, 2, 1, {{3, 3}, {1, 4}}};
    run_msda_case(c);
}
