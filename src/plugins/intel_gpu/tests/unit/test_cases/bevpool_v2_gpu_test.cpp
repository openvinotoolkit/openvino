// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <intel_gpu/primitives/bevpool_v2.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <limits>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

template <typename T>
void assert_with_error_stats(const memory::ptr& output, const std::vector<float>& expected, float abs_threshold, float rel_threshold) {
    mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr.size(), expected.size());

    float max_abs = 0.f;
    float mean_abs = 0.f;
    float max_rel = 0.f;

    for (size_t i = 0; i < output_ptr.size(); ++i) {
        const float actual = static_cast<float>(output_ptr[i]);
        const float exp = expected[i];

        const float abs_err = std::fabs(actual - exp);
        const float rel_base = std::max(std::fabs(exp), std::numeric_limits<float>::epsilon());
        const float rel_err = abs_err / rel_base;

        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);
        mean_abs += abs_err;

        ASSERT_LE(abs_err, abs_threshold) << "abs_err=" << abs_err << " at index " << i;
        ASSERT_LE(rel_err, rel_threshold) << "rel_err=" << rel_err << " at index " << i;
    }

    mean_abs /= static_cast<float>(output_ptr.size());
    ASSERT_LE(max_abs, abs_threshold) << "worst-case max_abs=" << max_abs << ", mean_abs=" << mean_abs << ", max_rel=" << max_rel;
}

// Reference computation mirroring the bevpool_v2 kernel semantics (single-camera case: dw_len ==
// depth_bins * feature_area, so camera_idx is always 0). Used to independently validate the
// blocked/vectorized OPT4 and OPT8 GPU kernels, which are otherwise never selected by the small
// (interval_count < 8) topologies used by the ref-path tests above.
std::vector<float> reference_bevpool_v2(const std::vector<float>& cf,
                                        const std::vector<float>& dw,
                                        const std::vector<uint32_t>& idx,
                                        const std::vector<uint32_t>& itv,
                                        size_t input_channels,
                                        size_t output_channels,
                                        size_t feature_area,
                                        size_t out_spatial) {
    std::vector<float> out(output_channels * out_spatial, 0.f);
    const size_t interval_count = itv.size() / 3;
    for (size_t interval = 0; interval < interval_count; ++interval) {
        const uint32_t start = itv[interval * 3 + 0];
        const uint32_t end = itv[interval * 3 + 1];
        const uint32_t bev_base = itv[interval * 3 + 2];
        for (size_t c = 0; c < output_channels; ++c) {
            float acc = 0.f;
            for (uint32_t i = start; i < end; ++i) {
                const uint32_t dw_index = idx[i];
                const uint32_t feature_idx = dw_index % static_cast<uint32_t>(feature_area);
                const size_t cf_offset = static_cast<size_t>(feature_idx) * input_channels + c;
                acc += cf[cf_offset] * dw[dw_index];
            }
            out[bev_base + c * out_spatial] = acc;
        }
    }
    return out;
}

TEST(BevPoolV2GpuTest, ref_comp_f32) {
    auto& engine = get_test_engine();
    auto stream = get_test_stream_ptr(get_test_default_config(engine));

    const auto cf = engine.allocate_memory({ov::PartialShape{1, 1, 2, 2}, data_types::f32, format::bfyx}, allocation_type::usm_host);
    const auto dw = engine.allocate_memory({ov::PartialShape{1, 2, 2, 2}, data_types::f32, format::bfyx}, allocation_type::usm_host);
    const auto idx = engine.allocate_memory({ov::PartialShape{5}, data_types::i32, format::bfyx}, allocation_type::usm_host);
    const auto itv = engine.allocate_memory({ov::PartialShape{6}, data_types::i32, format::bfyx}, allocation_type::usm_host);

    set_values<float>(cf, {10.f, 20.f, 30.f, 40.f});
    set_values<float>(dw, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    set_values<int32_t>(idx, {0, 1, 2, 3, 4});
    set_values<int32_t>(itv, {0, 2, 0, 2, 5, 1});

    const bound3d x_bound{-10.f, 10.f, 0.5f};
    const bound3d y_bound{-10.f, 10.f, 0.5f};
    const bound3d z_bound{-5.f, 3.f, 0.5f};
    const bound3d d_bound{0.f, 2.f, 1.f};

    topology topology;
    topology.add(input_layout("cf", cf->get_layout()));
    topology.add(input_layout("dw", dw->get_layout()));
    topology.add(input_layout("idx", idx->get_layout()));
    topology.add(input_layout("itv", itv->get_layout()));
    topology.add(bevpool_v2("bevpool_v2",
                            {input_info("cf"), input_info("dw"), input_info("idx"), input_info("itv")},
                            1,
                            1,
                            2,
                            2,
                            1,
                            2,
                            x_bound,
                            y_bound,
                            z_bound,
                            d_bound));

    auto network = get_network(engine, topology, get_test_default_config(engine), stream, false);
    network->set_input_data("cf", cf);
    network->set_input_data("dw", dw);
    network->set_input_data("idx", idx);
    network->set_input_data("itv", itv);

    const auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t{1});

    const auto output = outputs.at("bevpool_v2").get_memory();
    const std::vector<float> expected = {50.f, 300.f};
    // f32 path uses tighter acceptance to catch silent regressions early.
    assert_with_error_stats<float>(output, expected, 1e-4f, 1e-4f);
}

TEST(BevPoolV2GpuTest, ref_comp_f16_with_error_stats) {
    auto& engine = get_test_engine();
    auto stream = get_test_stream_ptr(get_test_default_config(engine));

    const auto cf = engine.allocate_memory({ov::PartialShape{1, 1, 2, 2}, data_types::f16, format::bfyx}, allocation_type::usm_host);
    const auto dw = engine.allocate_memory({ov::PartialShape{1, 2, 2, 2}, data_types::f16, format::bfyx}, allocation_type::usm_host);
    const auto idx = engine.allocate_memory({ov::PartialShape{5}, data_types::i32, format::bfyx}, allocation_type::usm_host);
    const auto itv = engine.allocate_memory({ov::PartialShape{6}, data_types::i32, format::bfyx}, allocation_type::usm_host);

    set_values<ov::float16>(cf, {ov::float16(10.f), ov::float16(20.f), ov::float16(30.f), ov::float16(40.f)});
    set_values<ov::float16>(
        dw,
        {ov::float16(1.f), ov::float16(2.f), ov::float16(3.f), ov::float16(4.f), ov::float16(5.f), ov::float16(6.f), ov::float16(7.f), ov::float16(8.f)});
    set_values<int32_t>(idx, {0, 1, 2, 3, 4});
    set_values<int32_t>(itv, {0, 2, 0, 2, 5, 1});

    const bound3d x_bound{-10.f, 10.f, 0.5f};
    const bound3d y_bound{-10.f, 10.f, 0.5f};
    const bound3d z_bound{-5.f, 3.f, 0.5f};
    const bound3d d_bound{0.f, 2.f, 1.f};

    topology topology;
    topology.add(input_layout("cf", cf->get_layout()));
    topology.add(input_layout("dw", dw->get_layout()));
    topology.add(input_layout("idx", idx->get_layout()));
    topology.add(input_layout("itv", itv->get_layout()));
    topology.add(bevpool_v2("bevpool_v2",
                            {input_info("cf"), input_info("dw"), input_info("idx"), input_info("itv")},
                            1,
                            1,
                            2,
                            2,
                            1,
                            2,
                            x_bound,
                            y_bound,
                            z_bound,
                            d_bound));

    auto network = get_network(engine, topology, get_test_default_config(engine), stream, false);
    network->set_input_data("cf", cf);
    network->set_input_data("dw", dw);
    network->set_input_data("idx", idx);
    network->set_input_data("itv", itv);

    const auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t{1});

    const auto output = outputs.at("bevpool_v2").get_memory();
    const std::vector<float> expected = {50.f, 300.f};
    // f16 acceptance is intentionally looser due mixed-precision accumulation/rounding.
    assert_with_error_stats<ov::float16>(output, expected, 2e-3f, 2e-3f);
}

TEST(BevPoolV2GpuTest, ref_comp_f32_u32_indices) {
    auto& engine = get_test_engine();
    auto stream = get_test_stream_ptr(get_test_default_config(engine));

    const auto cf = engine.allocate_memory({ov::PartialShape{1, 1, 2, 2}, data_types::f32, format::bfyx}, allocation_type::usm_host);
    const auto dw = engine.allocate_memory({ov::PartialShape{1, 2, 2, 2}, data_types::f32, format::bfyx}, allocation_type::usm_host);
    const auto idx = engine.allocate_memory({ov::PartialShape{5}, data_types::u32, format::bfyx}, allocation_type::usm_host);
    const auto itv = engine.allocate_memory({ov::PartialShape{6}, data_types::u32, format::bfyx}, allocation_type::usm_host);

    set_values<float>(cf, {10.f, 20.f, 30.f, 40.f});
    set_values<float>(dw, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
    set_values<uint32_t>(idx, {0u, 1u, 2u, 3u, 4u});
    set_values<uint32_t>(itv, {0u, 2u, 0u, 2u, 5u, 1u});

    const bound3d x_bound{-10.f, 10.f, 0.5f};
    const bound3d y_bound{-10.f, 10.f, 0.5f};
    const bound3d z_bound{-5.f, 3.f, 0.5f};
    const bound3d d_bound{0.f, 2.f, 1.f};

    topology topology;
    topology.add(input_layout("cf", cf->get_layout()));
    topology.add(input_layout("dw", dw->get_layout()));
    topology.add(input_layout("idx", idx->get_layout()));
    topology.add(input_layout("itv", itv->get_layout()));
    topology.add(bevpool_v2("bevpool_v2",
                            {input_info("cf"), input_info("dw"), input_info("idx"), input_info("itv")},
                            1,
                            1,
                            2,
                            2,
                            1,
                            2,
                            x_bound,
                            y_bound,
                            z_bound,
                            d_bound));

    auto network = get_network(engine, topology, get_test_default_config(engine), stream, false);
    network->set_input_data("cf", cf);
    network->set_input_data("dw", dw);
    network->set_input_data("idx", idx);
    network->set_input_data("itv", itv);

    const auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t{1});

    const auto output = outputs.at("bevpool_v2").get_memory();
    const std::vector<float> expected = {50.f, 300.f};
    assert_with_error_stats<float>(output, expected, 1e-4f, 1e-4f);
}

// output_channels=6 is not a multiple of 4, so this exercises the OPT4 blocked kernel's channel
// tail handling (support_opt_kernel<4> requires output_channels >= 4 && interval_count >= 8).
TEST(BevPoolV2GpuTest, opt4_comp_f32_channel_tail) {
    auto& engine = get_test_engine();
    auto stream = get_test_stream_ptr(get_test_default_config(engine));

    const size_t input_channels = 6;
    const size_t output_channels = 6;
    const size_t image_width = 2;
    const size_t image_height = 2;
    const size_t feature_area = image_width * image_height;
    const size_t depth_bins = 2;
    const size_t dw_len = depth_bins * feature_area;
    const size_t interval_count = 8;
    const size_t out_spatial = interval_count;

    std::vector<float> cf_vals(feature_area * input_channels);
    for (size_t i = 0; i < cf_vals.size(); ++i) {
        cf_vals[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dw_vals(dw_len);
    for (size_t i = 0; i < dw_vals.size(); ++i) {
        dw_vals[i] = static_cast<float>(i + 1);
    }
    std::vector<uint32_t> idx_vals(interval_count);
    for (size_t i = 0; i < idx_vals.size(); ++i) {
        idx_vals[i] = static_cast<uint32_t>(i % dw_len);
    }
    std::vector<uint32_t> itv_vals(interval_count * 3);
    for (size_t i = 0; i < interval_count; ++i) {
        itv_vals[i * 3 + 0] = static_cast<uint32_t>(i);
        itv_vals[i * 3 + 1] = static_cast<uint32_t>(i + 1);
        itv_vals[i * 3 + 2] = static_cast<uint32_t>(i);
    }

    const auto cf = engine.allocate_memory(
        {ov::PartialShape{1, static_cast<int64_t>(input_channels), static_cast<int64_t>(image_height), static_cast<int64_t>(image_width)},
         data_types::f32,
         format::bfyx},
        allocation_type::usm_host);
    const auto dw =
        engine.allocate_memory({ov::PartialShape{1, static_cast<int64_t>(depth_bins), static_cast<int64_t>(image_height), static_cast<int64_t>(image_width)},
                                data_types::f32,
                                format::bfyx},
                               allocation_type::usm_host);
    const auto idx = engine.allocate_memory({ov::PartialShape{static_cast<int64_t>(interval_count)}, data_types::u32, format::bfyx}, allocation_type::usm_host);
    const auto itv =
        engine.allocate_memory({ov::PartialShape{static_cast<int64_t>(interval_count * 3)}, data_types::u32, format::bfyx}, allocation_type::usm_host);

    set_values<float>(cf, cf_vals);
    set_values<float>(dw, dw_vals);
    set_values<uint32_t>(idx, idx_vals);
    set_values<uint32_t>(itv, itv_vals);

    const bound3d x_bound{-10.f, 10.f, 0.5f};
    const bound3d y_bound{-10.f, 10.f, 0.5f};
    const bound3d z_bound{-5.f, 3.f, 0.5f};
    const bound3d d_bound{0.f, static_cast<float>(depth_bins), 1.f};

    topology topology;
    topology.add(input_layout("cf", cf->get_layout()));
    topology.add(input_layout("dw", dw->get_layout()));
    topology.add(input_layout("idx", idx->get_layout()));
    topology.add(input_layout("itv", itv->get_layout()));
    topology.add(bevpool_v2("bevpool_v2",
                            {input_info("cf"), input_info("dw"), input_info("idx"), input_info("itv")},
                            static_cast<uint32_t>(input_channels),
                            static_cast<uint32_t>(output_channels),
                            static_cast<uint32_t>(image_width),
                            static_cast<uint32_t>(image_height),
                            static_cast<uint32_t>(out_spatial),
                            1,
                            x_bound,
                            y_bound,
                            z_bound,
                            d_bound));

    auto network = get_network(engine, topology, get_test_default_config(engine), stream, false);
    network->set_input_data("cf", cf);
    network->set_input_data("dw", dw);
    network->set_input_data("idx", idx);
    network->set_input_data("itv", itv);

    const auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t{1});

    const auto output = outputs.at("bevpool_v2").get_memory();
    const auto expected = reference_bevpool_v2(cf_vals, dw_vals, idx_vals, itv_vals, input_channels, output_channels, feature_area, out_spatial);
    assert_with_error_stats<float>(output, expected, 1e-3f, 1e-3f);
}

// output_channels=8, interval_count=8 satisfies support_opt_kernel<8> for non-fp16 inputs.
TEST(BevPoolV2GpuTest, opt8_comp_f32) {
    auto& engine = get_test_engine();
    auto stream = get_test_stream_ptr(get_test_default_config(engine));

    const size_t input_channels = 8;
    const size_t output_channels = 8;
    const size_t image_width = 2;
    const size_t image_height = 2;
    const size_t feature_area = image_width * image_height;
    const size_t depth_bins = 2;
    const size_t dw_len = depth_bins * feature_area;
    const size_t interval_count = 8;
    const size_t out_spatial = interval_count;

    std::vector<float> cf_vals(feature_area * input_channels);
    for (size_t i = 0; i < cf_vals.size(); ++i) {
        cf_vals[i] = static_cast<float>(i + 1);
    }
    std::vector<float> dw_vals(dw_len);
    for (size_t i = 0; i < dw_vals.size(); ++i) {
        dw_vals[i] = static_cast<float>(i + 1);
    }
    std::vector<uint32_t> idx_vals(interval_count);
    for (size_t i = 0; i < idx_vals.size(); ++i) {
        idx_vals[i] = static_cast<uint32_t>(i % dw_len);
    }
    std::vector<uint32_t> itv_vals(interval_count * 3);
    for (size_t i = 0; i < interval_count; ++i) {
        itv_vals[i * 3 + 0] = static_cast<uint32_t>(i);
        itv_vals[i * 3 + 1] = static_cast<uint32_t>(i + 1);
        itv_vals[i * 3 + 2] = static_cast<uint32_t>(i);
    }

    const auto cf = engine.allocate_memory(
        {ov::PartialShape{1, static_cast<int64_t>(input_channels), static_cast<int64_t>(image_height), static_cast<int64_t>(image_width)},
         data_types::f32,
         format::bfyx},
        allocation_type::usm_host);
    const auto dw =
        engine.allocate_memory({ov::PartialShape{1, static_cast<int64_t>(depth_bins), static_cast<int64_t>(image_height), static_cast<int64_t>(image_width)},
                                data_types::f32,
                                format::bfyx},
                               allocation_type::usm_host);
    const auto idx = engine.allocate_memory({ov::PartialShape{static_cast<int64_t>(interval_count)}, data_types::u32, format::bfyx}, allocation_type::usm_host);
    const auto itv =
        engine.allocate_memory({ov::PartialShape{static_cast<int64_t>(interval_count * 3)}, data_types::u32, format::bfyx}, allocation_type::usm_host);

    set_values<float>(cf, cf_vals);
    set_values<float>(dw, dw_vals);
    set_values<uint32_t>(idx, idx_vals);
    set_values<uint32_t>(itv, itv_vals);

    const bound3d x_bound{-10.f, 10.f, 0.5f};
    const bound3d y_bound{-10.f, 10.f, 0.5f};
    const bound3d z_bound{-5.f, 3.f, 0.5f};
    const bound3d d_bound{0.f, static_cast<float>(depth_bins), 1.f};

    topology topology;
    topology.add(input_layout("cf", cf->get_layout()));
    topology.add(input_layout("dw", dw->get_layout()));
    topology.add(input_layout("idx", idx->get_layout()));
    topology.add(input_layout("itv", itv->get_layout()));
    topology.add(bevpool_v2("bevpool_v2",
                            {input_info("cf"), input_info("dw"), input_info("idx"), input_info("itv")},
                            static_cast<uint32_t>(input_channels),
                            static_cast<uint32_t>(output_channels),
                            static_cast<uint32_t>(image_width),
                            static_cast<uint32_t>(image_height),
                            static_cast<uint32_t>(out_spatial),
                            1,
                            x_bound,
                            y_bound,
                            z_bound,
                            d_bound));

    auto network = get_network(engine, topology, get_test_default_config(engine), stream, false);
    network->set_input_data("cf", cf);
    network->set_input_data("dw", dw);
    network->set_input_data("idx", idx);
    network->set_input_data("itv", itv);

    const auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t{1});

    const auto output = outputs.at("bevpool_v2").get_memory();
    const auto expected = reference_bevpool_v2(cf_vals, dw_vals, idx_vals, itv_vals, input_channels, output_channels, feature_area, out_spatial);
    assert_with_error_stats<float>(output, expected, 1e-3f, 1e-3f);
}

// fp16 with output_channels a multiple of 8 and interval_count=16 satisfies the stricter fp16
// sub-rule inside support_opt_kernel<8> (output_channels % 8 == 0 && interval_count >= 16),
// exercising the OPT8 fp16 vectorized load path.
TEST(BevPoolV2GpuTest, opt8_comp_f16_vectorized) {
    auto& engine = get_test_engine();
    auto stream = get_test_stream_ptr(get_test_default_config(engine));

    const size_t input_channels = 8;
    const size_t output_channels = 8;
    const size_t image_width = 2;
    const size_t image_height = 2;
    const size_t feature_area = image_width * image_height;
    const size_t depth_bins = 4;
    const size_t dw_len = depth_bins * feature_area;
    const size_t interval_count = 16;
    const size_t out_spatial = interval_count;

    std::vector<float> cf_vals(feature_area * input_channels);
    for (size_t i = 0; i < cf_vals.size(); ++i) {
        cf_vals[i] = static_cast<float>((i % 10) + 1);
    }
    std::vector<float> dw_vals(dw_len);
    for (size_t i = 0; i < dw_vals.size(); ++i) {
        dw_vals[i] = static_cast<float>((i % 6) + 1);
    }
    std::vector<uint32_t> idx_vals(interval_count);
    for (size_t i = 0; i < idx_vals.size(); ++i) {
        idx_vals[i] = static_cast<uint32_t>(i % dw_len);
    }
    std::vector<uint32_t> itv_vals(interval_count * 3);
    for (size_t i = 0; i < interval_count; ++i) {
        itv_vals[i * 3 + 0] = static_cast<uint32_t>(i);
        itv_vals[i * 3 + 1] = static_cast<uint32_t>(i + 1);
        itv_vals[i * 3 + 2] = static_cast<uint32_t>(i);
    }

    const auto cf = engine.allocate_memory(
        {ov::PartialShape{1, static_cast<int64_t>(input_channels), static_cast<int64_t>(image_height), static_cast<int64_t>(image_width)},
         data_types::f16,
         format::bfyx},
        allocation_type::usm_host);
    const auto dw =
        engine.allocate_memory({ov::PartialShape{1, static_cast<int64_t>(depth_bins), static_cast<int64_t>(image_height), static_cast<int64_t>(image_width)},
                                data_types::f16,
                                format::bfyx},
                               allocation_type::usm_host);
    const auto idx = engine.allocate_memory({ov::PartialShape{static_cast<int64_t>(interval_count)}, data_types::u32, format::bfyx}, allocation_type::usm_host);
    const auto itv =
        engine.allocate_memory({ov::PartialShape{static_cast<int64_t>(interval_count * 3)}, data_types::u32, format::bfyx}, allocation_type::usm_host);

    std::vector<ov::float16> cf_vals_f16(cf_vals.begin(), cf_vals.end());
    std::vector<ov::float16> dw_vals_f16(dw_vals.begin(), dw_vals.end());
    set_values<ov::float16>(cf, cf_vals_f16);
    set_values<ov::float16>(dw, dw_vals_f16);
    set_values<uint32_t>(idx, idx_vals);
    set_values<uint32_t>(itv, itv_vals);

    const bound3d x_bound{-10.f, 10.f, 0.5f};
    const bound3d y_bound{-10.f, 10.f, 0.5f};
    const bound3d z_bound{-5.f, 3.f, 0.5f};
    const bound3d d_bound{0.f, static_cast<float>(depth_bins), 1.f};

    topology topology;
    topology.add(input_layout("cf", cf->get_layout()));
    topology.add(input_layout("dw", dw->get_layout()));
    topology.add(input_layout("idx", idx->get_layout()));
    topology.add(input_layout("itv", itv->get_layout()));
    topology.add(bevpool_v2("bevpool_v2",
                            {input_info("cf"), input_info("dw"), input_info("idx"), input_info("itv")},
                            static_cast<uint32_t>(input_channels),
                            static_cast<uint32_t>(output_channels),
                            static_cast<uint32_t>(image_width),
                            static_cast<uint32_t>(image_height),
                            static_cast<uint32_t>(out_spatial),
                            1,
                            x_bound,
                            y_bound,
                            z_bound,
                            d_bound));

    auto network = get_network(engine, topology, get_test_default_config(engine), stream, false);
    network->set_input_data("cf", cf);
    network->set_input_data("dw", dw);
    network->set_input_data("idx", idx);
    network->set_input_data("itv", itv);

    const auto outputs = network->execute();
    ASSERT_EQ(outputs.size(), size_t{1});

    const auto output = outputs.at("bevpool_v2").get_memory();
    const auto expected = reference_bevpool_v2(cf_vals, dw_vals, idx_vals, itv_vals, input_channels, output_channels, feature_area, out_spatial);
    // fp16 acceptance is intentionally looser due to mixed-precision accumulation/rounding.
    assert_with_error_stats<ov::float16>(output, expected, 2e-2f, 2e-2f);
}

}  // namespace
