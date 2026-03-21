// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/bevpool_v2.hpp>
#include <intel_gpu/primitives/input_layout.hpp>

#include <cmath>
#include <limits>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

template <typename T>
void assert_with_error_stats(const memory::ptr& output,
                             const std::vector<float>& expected,
                             float abs_threshold,
                             float rel_threshold) {
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

TEST(bevpool_v2_gpu_test, ref_comp_f32) {
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

TEST(bevpool_v2_gpu_test, ref_comp_f16_with_error_stats) {
    auto& engine = get_test_engine();
    auto stream = get_test_stream_ptr(get_test_default_config(engine));

    const auto cf = engine.allocate_memory({ov::PartialShape{1, 1, 2, 2}, data_types::f16, format::bfyx}, allocation_type::usm_host);
    const auto dw = engine.allocate_memory({ov::PartialShape{1, 2, 2, 2}, data_types::f16, format::bfyx}, allocation_type::usm_host);
    const auto idx = engine.allocate_memory({ov::PartialShape{5}, data_types::i32, format::bfyx}, allocation_type::usm_host);
    const auto itv = engine.allocate_memory({ov::PartialShape{6}, data_types::i32, format::bfyx}, allocation_type::usm_host);

    set_values<ov::float16>(cf, {ov::float16(10.f), ov::float16(20.f), ov::float16(30.f), ov::float16(40.f)});
    set_values<ov::float16>(dw, {ov::float16(1.f), ov::float16(2.f), ov::float16(3.f), ov::float16(4.f),
                                 ov::float16(5.f), ov::float16(6.f), ov::float16(7.f), ov::float16(8.f)});
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

TEST(bevpool_v2_gpu_test, ref_comp_f32_u32_indices) {
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

}  // namespace
