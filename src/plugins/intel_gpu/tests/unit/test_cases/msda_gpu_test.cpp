// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <sstream>
#include <tuple>
#include <vector>

#include "intel_gpu/primitives/input_layout.hpp"
#include "intel_gpu/primitives/msda.hpp"
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

// Host model of the MSDA semantics: per (b, q, h, d) accumulate over levels and
// points the attention-weighted bilinear sample of value at the pixel
// coordinates x * W - 0.5, y * H - 0.5 of the normalized location (x, y), with
// zero padding outside the feature map.
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

// Rounds to the precision the kernel reads, so the host reference sees the
// same inputs as the device.
std::vector<float> round_to(data_types type, std::vector<float> values) {
    if (type == data_types::f16) {
        for (auto& v : values)
            v = static_cast<float>(ov::float16(v));
    }
    return values;
}

template <typename T>
memory::ptr make_input(engine& engine, const ov::PartialShape& shape, data_types type, const std::vector<float>& values) {
    auto memory = engine.allocate_memory({shape, type, format::get_default_format(shape.size())});
    std::vector<T> converted(values.begin(), values.end());
    set_values(memory, converted);
    return memory;
}

template <typename T>
void run_msda_case(const MsdaCase& c, data_types type, bool is_caching_test) {
    auto& engine = get_test_engine();
    const size_t K = c.keys();

    std::vector<float> value_vec(c.batch * K * c.heads * c.embed);
    for (size_t i = 0; i < value_vec.size(); ++i)
        value_vec[i] = 0.125f * static_cast<float>(i % 97) - 3.0f;
    std::vector<float> loc_vec(c.batch * c.queries * c.heads * c.levels * c.points * 2);
    // Locations from -0.3 to 1.3 put samples inside the levels, on their
    // borders and fully outside them.
    for (size_t i = 0; i < loc_vec.size(); ++i)
        loc_vec[i] = -0.3f + 0.1f * static_cast<float>(i % 17);
    std::vector<float> weight_vec(c.batch * c.queries * c.heads * c.levels * c.points);
    for (size_t i = 0; i < weight_vec.size(); ++i)
        weight_vec[i] = static_cast<float>((i % 7) + 1) / 8.f;
    value_vec = round_to(type, value_vec);
    loc_vec = round_to(type, loc_vec);
    weight_vec = round_to(type, weight_vec);

    std::vector<int32_t> spatial_flat;
    for (const auto& hw : c.spatial)
        spatial_flat.insert(spatial_flat.end(), {hw.first, hw.second});
    std::vector<int32_t> starts;
    int32_t position = 0;
    for (const auto& hw : c.spatial) {
        starts.push_back(position);
        position += hw.first * hw.second;
    }

    const auto B = int64_t(c.batch), Q = int64_t(c.queries), H = int64_t(c.heads), L = int64_t(c.levels), P = int64_t(c.points);
    auto data_value = make_input<T>(engine, ov::PartialShape{B, int64_t(K), H, int64_t(c.embed)}, type, value_vec);
    auto data_spatial_shapes = engine.allocate_memory({ov::PartialShape{L, 2}, data_types::i32, format::bfyx});
    auto data_level_start = engine.allocate_memory({ov::PartialShape{L}, data_types::i32, format::bfyx});
    auto data_sampling_loc = make_input<T>(engine, ov::PartialShape{B, Q, H, L, P, 2}, type, loc_vec);
    auto data_attn_weight = make_input<T>(engine, ov::PartialShape{B, Q, H, L, P}, type, weight_vec);
    set_values(data_spatial_shapes, spatial_flat);
    set_values(data_level_start, starts);

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

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
    network->set_input_data("data_value", data_value);
    network->set_input_data("data_spatial_shapes", data_spatial_shapes);
    network->set_input_data("data_level_start_idx", data_level_start);
    network->set_input_data("data_sampling_loc", data_sampling_loc);
    network->set_input_data("data_attn_weight", data_attn_weight);
    auto outputs = network->execute();
    ASSERT_EQ(outputs.count("msda"), size_t(1));

    auto output_memory = outputs.at("msda").get_memory();
    cldnn::mem_lock<T> output_ptr(output_memory, get_test_stream());
    const auto ref = msda_reference(c, value_vec, loc_vec, weight_vec);
    ASSERT_EQ(output_memory->count(), ref.size());
    // The kernel computes in float for both types; f16 only adds the rounding
    // of the stored result.
    for (size_t i = 0; i < ref.size(); ++i) {
        const float tolerance = type == data_types::f16 ? 1e-3f * std::max(1.f, std::abs(ref[i])) : 2e-4f;
        ASSERT_NEAR(static_cast<float>(output_ptr[i]), ref[i], tolerance) << "index " << i;
    }
}

// Shape case, data type and whether the network goes through export and import.
using MsdaTestParams = std::tuple<MsdaCase, data_types, bool>;

const std::vector<MsdaCase> msda_cases = {
    {1, 2, 2, 4, 2, 2, {{2, 2}, {2, 4}}},
    {2, 3, 1, 8, 1, 3, {{5, 3}}},
    // On 3x3 and 1x4 levels most samples touch a border or fall outside.
    {1, 4, 1, 2, 2, 1, {{3, 3}, {1, 4}}},
    // GroundingDINO 800x1333 stride-8 level: pixel coordinates up to 167 need
    // more precision than f16 offers for the bilinear weights.
    {1, 8, 2, 8, 2, 4, {{100, 167}, {50, 84}}},
};

}  // namespace

class msda_gpu_test : public ::testing::TestWithParam<MsdaTestParams> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<MsdaTestParams>& info) {
        const auto& [c, type, is_caching_test] = info.param;
        std::ostringstream name;
        name << "B" << c.batch << "_Q" << c.queries << "_H" << c.heads << "_D" << c.embed << "_P" << c.points << "_levels";
        for (const auto& [h, w] : c.spatial)
            name << "_" << h << "x" << w;
        name << "_" << ov::element::Type(type) << (is_caching_test ? "_cached" : "");
        return name.str();
    }
};

TEST_P(msda_gpu_test, reference) {
    const auto& [c, type, is_caching_test] = GetParam();
    if (type == data_types::f16)
        run_msda_case<ov::float16>(c, type, is_caching_test);
    else
        run_msda_case<float>(c, type, is_caching_test);
}

INSTANTIATE_TEST_SUITE_P(smoke_msda_gpu_test,
                         msda_gpu_test,
                         ::testing::Combine(::testing::ValuesIn(msda_cases), ::testing::Values(data_types::f32, data_types::f16), ::testing::Values(false)),
                         msda_gpu_test::get_test_case_name);

INSTANTIATE_TEST_SUITE_P(smoke_msda_gpu_test_cached,
                         msda_gpu_test,
                         ::testing::Combine(::testing::Values(msda_cases.front()),
                                            ::testing::Values(data_types::f32, data_types::f16),
                                            ::testing::Values(true)),
                         msda_gpu_test::get_test_case_name);
