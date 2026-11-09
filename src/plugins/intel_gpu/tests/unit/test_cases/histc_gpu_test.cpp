// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/histc.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <limits>
#include <numeric>

#include "openvino/reference/histc.hpp"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

struct HistcTestParams {
    size_t count;
    int64_t bins;
    double min;
    double max;
    bool is_dynamic;
    int resolution;  // values are multiples of 1 / resolution
};

class histc_gpu_test : public ::testing::TestWithParam<HistcTestParams> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<HistcTestParams>& obj) {
        const auto& p = obj.param;
        std::ostringstream name;
        name << "count" << p.count << "_bins" << p.bins << "_" << (p.min == p.max ? "data_range" : "fixed_range") << (p.is_dynamic ? "_dynamic" : "_static")
             << "_res" << p.resolution;
        return name.str();
    }

    void execute(bool is_caching_test) {
        const auto& p = GetParam();
        auto& engine = get_test_engine();

        tests::random_generator rg(GET_SUITE_NAME);
        auto values = rg.generate_random_1d<float>(p.count, -20, 20, p.resolution);
        const layout in_layout{ov::PartialShape{static_cast<int64_t>(p.count)}, data_types::f32, format::bfyx};
        auto input = engine.allocate_memory(in_layout);
        set_values(input, values);

        topology topology(input_layout("input", p.is_dynamic ? layout{ov::PartialShape::dynamic(1), data_types::f32, format::bfyx} : in_layout),
                          histc("histc", input_info("input"), p.bins, p.min, p.max));

        auto config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        auto network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input);
        auto output = network->execute().at("histc").get_memory();

        std::vector<float> expected(p.bins);
        ov::reference::histc(values.data(), expected.data(), values.size(), p.bins, p.min, p.max);

        cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
        ASSERT_EQ(output_ptr.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            ASSERT_EQ(expected[i], output_ptr[i]) << "bin " << i;
        }
    }
};

TEST_P(histc_gpu_test, ref_comp) {
    execute(false);
}

TEST_P(histc_gpu_test, ref_comp_cached) {
    execute(true);
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         histc_gpu_test,
                         testing::ValuesIn(std::vector<HistcTestParams>{
                             // Integer-valued data hits bin edges exactly.
                             {1, 10, 0.0, 0.0, false, 1},
                             {100, 10, -5.0, 5.0, false, 1},
                             {100, 10, 0.0, 0.0, true, 1},
                             {1000000, 100, -10.5, 10.5, false, 1},
                             {1000000, 100, 0.0, 0.0, true, 1},
                             {1000000, 17, 0.0, 0.0, false, 1},
                             // Bin edges between representable values check the division rounding.
                             {1000000, 4096, 0.0, 0.0, false, 1000000},
                             {1000000, 1000, -7.3, 11.9, true, 1000000},
                             // Too many bins for local memory.
                             {300000, 100000, -20.0, 20.0, false, 1},
                             {300000, 100000, 0.0, 0.0, true, 1},
                         }),
                         histc_gpu_test::get_test_case_name);

TEST(histc_gpu, basic) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ov::PartialShape{6}, data_types::f32, format::bfyx});
    set_values(input, {1.f, 2.f, 1.f, 4.f, 2.5f, 7.f});

    topology topology(input_layout("input", input->get_layout()), histc("histc", input_info("input"), 4, 0.0, 4.0));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto output = network.execute().at("histc").get_memory();

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    ASSERT_EQ(std::vector<float>(output_ptr.begin(), output_ptr.end()), (std::vector<float>{0.f, 2.f, 2.f, 1.f}));
}

// preserve_input_type = false: counts above 255 stay exact for u8 input.
TEST(histc_gpu, u8_input_i32_output) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ov::PartialShape{300}, data_types::u8, format::bfyx});
    set_values(input, std::vector<uint8_t>(300, 7));

    auto prim = histc("histc", input_info("input"), 2, 0.0, 0.0);
    prim.output_data_types = {optional_data_type{data_types::i32}};
    topology topology(input_layout("input", input->get_layout()), prim);
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto output = network.execute().at("histc").get_memory();

    cldnn::mem_lock<int32_t, mem_lock_type::read> output_ptr(output, get_test_stream());
    ASSERT_EQ(std::vector<int32_t>(output_ptr.begin(), output_ptr.end()), (std::vector<int32_t>{0, 300}));
}

// A NaN makes the range taken from data invalid, which yields an empty histogram.
TEST(histc_gpu, nan_data_range) {
    auto& engine = get_test_engine();
    // Large enough for the block-read path; NaN is a single element in the middle.
    std::vector<float> values(1000);
    std::iota(values.begin(), values.end(), 0.f);
    values[500] = std::numeric_limits<float>::quiet_NaN();
    auto input = engine.allocate_memory({ov::PartialShape{static_cast<int64_t>(values.size())}, data_types::f32, format::bfyx});
    set_values(input, values);

    topology topology(input_layout("input", input->get_layout()), histc("histc", input_info("input"), 2, 0.0, 0.0));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto output = network.execute().at("histc").get_memory();

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    ASSERT_EQ(std::vector<float>(output_ptr.begin(), output_ptr.end()), (std::vector<float>{0.f, 0.f}));
}

// f16 widens an equal range to [x - 1, x + 1]; nextafter in f16 would give [2047, 2050] and pick bin 0.
TEST(histc_gpu, f16_equal_values_range) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ov::PartialShape{3}, data_types::f16, format::bfyx});
    set_values(input, std::vector<ov::float16>(3, ov::float16(2048.f)));

    topology topology(input_layout("input", input->get_layout()), histc("histc", input_info("input"), 2, 0.0, 0.0));
    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto output = network.execute().at("histc").get_memory();

    cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
    ASSERT_EQ(std::vector<float>(output_ptr.begin(), output_ptr.end()), (std::vector<float>{0.f, 3.f}));
}

}  // namespace
