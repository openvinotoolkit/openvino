// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/normalize.hpp>
#include <cmath>
#include <iostream>
#include <vector>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

template <typename InputType>
struct is_input_and_output_same {
    static const bool value = true;
};

template <>
struct is_input_and_output_same<int8_t> {
    static const bool value = false;
};

template <format::type layoutFormat, typename DataType, bool across_spatial>
struct normalize_input_types {
    static const auto format = layoutFormat;
    using type = DataType;
    using output_type = typename std::conditional<is_input_and_output_same<DataType>::value, DataType, float>::type;
    static const bool normalize_type = across_spatial;
};

template <typename NormalizeInput>
struct normalize_basic : public testing::Test {
    static const auto format = NormalizeInput::format;
    using input_type = typename NormalizeInput::type;
    using output_type = typename NormalizeInput::output_type;
    const ov::element::Type data_type = ov::element::from<input_type>();
    const ov::element::Type output_data_type = ov::element::from<output_type>();
    static const bool across_spatial = NormalizeInput::normalize_type;
    const std::vector<output_type> get_expected_result() {
        return get_expected_result(std::integral_constant<bool, across_spatial>());
    }

    const std::vector<input_type> get_input_values(unsigned b, unsigned f, unsigned y, unsigned x) {
        std::vector<input_type> inputVals(b * f * y * x);
        float n = 0;
        std::generate(inputVals.begin(), inputVals.end(), [&n]() {
            return static_cast<input_type>(n++);
        });
        return inputVals;
    }

    void execute(bool is_caching_test) {
        //  Input  : 1x2x3x3
        //  Output : 1x2x3x3
        auto& engine = get_test_engine();
        const unsigned b = 1;
        const unsigned f = 2;
        const unsigned y = 3;
        const unsigned x = 3;

        auto input = engine.allocate_memory({this->data_type, format::bfyx, {b, f, y, x}});
        auto weights = engine.allocate_memory({data_types::f32, format::bfyx, {1, f, 1, 1}});

        auto inputVals = this->get_input_values(b, f, y, x);
        std::vector<float> weightVals(f);
        for (auto& it : weightVals) {
            it = 1.f;
        }

        set_values(input, inputVals);
        set_values(weights, weightVals);

        topology topology;
        topology.add(input_layout("Input0", input->get_layout()));
        topology.add(data("Input1", weights));
        topology.add(reorder("reordered_Input0", input_info("Input0"), this->format, this->data_type));
        topology.add(reorder("reordered_Input1", input_info("Input1"), this->format, data_types::f32));
        topology.add(normalize("normalize2", input_info("reordered_Input0"), "reordered_Input1", this->across_spatial));
        topology.add(reorder("plane_normalize2", input_info("normalize2"), format::bfyx, this->output_data_type));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("Input0", input);

        auto outputs = network->execute();

        auto output = outputs.at("plane_normalize2").get_memory();
        if (this->data_type == data_types::f16) {
            cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
            auto expected_results = this->get_expected_result();
            for (size_t i = 0; i < expected_results.size(); ++i) {
                ASSERT_NEAR(expected_results[i], output_ptr[i], 0.001);
            }
        } else {
            cldnn::mem_lock<float> output_ptr(output, get_test_stream());
            auto expected_results = this->get_expected_result();
            for (size_t i = 0; i < expected_results.size(); ++i) {
                ASSERT_TRUE(are_equal(expected_results[i], output_ptr[i]));
            }
        }
    }

private:
    static const std::vector<output_type> get_expected_result(std::true_type) {
        static const std::vector<float> result = {0.f,
                                                  0.0236691f,
                                                  0.0473381f,
                                                  0.0710072f,
                                                  0.0946762f,
                                                  0.118345f,
                                                  0.142014f,
                                                  0.165683f,
                                                  0.189352f,
                                                  0.213021f,
                                                  0.236691f,
                                                  0.26036f,
                                                  0.284029f,
                                                  0.307698f,
                                                  0.331367f,
                                                  0.355036f,
                                                  0.378705f,
                                                  0.402374f};
        return std::vector<output_type>(result.begin(), result.end());
    }

    static const std::vector<output_type> get_expected_result(std::false_type) {
        static const std::vector<float> result = {0.f,
                                                  0.0995037f,
                                                  0.178885f,
                                                  0.242536f,
                                                  0.294086f,
                                                  0.336336f,
                                                  0.371391f,
                                                  0.400819f,
                                                  0.425797f,
                                                  1.f,
                                                  0.995037f,
                                                  0.98387f,
                                                  0.970143f,
                                                  0.955779f,
                                                  0.941742f,
                                                  0.928477f,
                                                  0.916157f,
                                                  0.904819f};
        return std::vector<output_type>(result.begin(), result.end());
    }
};

using format_types = testing::Types<normalize_input_types<format::bfyx, float, false>,
                                    normalize_input_types<format::byxf, float, false>,
                                    normalize_input_types<format::yxfb, float, false>,
                                    normalize_input_types<format::b_fs_yx_fsv32, float, false>,
                                    normalize_input_types<format::b_fs_yx_fsv16, float, false>,
                                    normalize_input_types<format::bs_fs_yx_bsv32_fsv16, float, false>,
                                    normalize_input_types<format::bs_fs_yx_bsv16_fsv16, float, false>,
                                    normalize_input_types<format::bs_fs_yx_bsv32_fsv32, float, false>,
                                    normalize_input_types<format::bfyx, ov::float16, false>,
                                    normalize_input_types<format::b_fs_yx_fsv32, ov::float16, false>,
                                    normalize_input_types<format::b_fs_yx_fsv16, ov::float16, false>,
                                    normalize_input_types<format::bs_fs_yx_bsv32_fsv16, ov::float16, false>,
                                    normalize_input_types<format::bs_fs_yx_bsv16_fsv16, ov::float16, false>,
                                    normalize_input_types<format::bs_fs_yx_bsv32_fsv32, ov::float16, false>,
                                    normalize_input_types<format::bfyx, int8_t, false>,
                                    normalize_input_types<format::b_fs_yx_fsv32, int8_t, false>,
                                    normalize_input_types<format::b_fs_yx_fsv16, int8_t, false>,
                                    normalize_input_types<format::bs_fs_yx_bsv32_fsv16, int8_t, false>,
                                    normalize_input_types<format::bs_fs_yx_bsv16_fsv16, int8_t, false>,
                                    normalize_input_types<format::bs_fs_yx_bsv32_fsv32, int8_t, false>,
                                    normalize_input_types<format::bfyx, float, true>,
                                    normalize_input_types<format::byxf, float, true>,
                                    normalize_input_types<format::yxfb, float, true>,
                                    normalize_input_types<format::b_fs_yx_fsv32, float, true>,
                                    normalize_input_types<format::b_fs_yx_fsv16, float, true>,
                                    normalize_input_types<format::bs_fs_yx_bsv32_fsv16, float, true>,
                                    normalize_input_types<format::bs_fs_yx_bsv16_fsv16, float, true>,
                                    normalize_input_types<format::bs_fs_yx_bsv32_fsv32, float, true>,
                                    normalize_input_types<format::bfyx, ov::float16, true>,
                                    normalize_input_types<format::b_fs_yx_fsv32, ov::float16, true>,
                                    normalize_input_types<format::b_fs_yx_fsv16, ov::float16, true>,
                                    normalize_input_types<format::bs_fs_yx_bsv32_fsv16, ov::float16, true>,
                                    normalize_input_types<format::bs_fs_yx_bsv16_fsv16, ov::float16, true>,
                                    normalize_input_types<format::bs_fs_yx_bsv32_fsv32, ov::float16, true>,
                                    normalize_input_types<format::bfyx, int8_t, true>,
                                    normalize_input_types<format::b_fs_yx_fsv32, int8_t, true>,
                                    normalize_input_types<format::b_fs_yx_fsv16, int8_t, true>,
                                    normalize_input_types<format::bs_fs_yx_bsv32_fsv16, int8_t, true>,
                                    normalize_input_types<format::bs_fs_yx_bsv16_fsv16, int8_t, true>,
                                    normalize_input_types<format::bs_fs_yx_bsv32_fsv32, int8_t, true>>;

TYPED_TEST_SUITE(normalize_basic, format_types);

TYPED_TEST(normalize_basic, basic) {
    this->execute(false);
}

// Slices whose sum of squares is far below 1 (e.g. NetVLAD clusters that receive almost no soft-assignment) must still be
// normalized to unit L2 norm instead of being flushed to zero. All-zero slices must stay zero: their scale
// 1/sqrt(eps) = 1e6 exceeds the f16 range, and 0 * inf would be NaN.
struct normalize_small_norm_params {
    data_types data_type;
    bool across_spatial;
    bool zero_input = false;
};

class normalize_small_norm : public testing::TestWithParam<normalize_small_norm_params> {};

TEST_P(normalize_small_norm, matches_reference) {
    const auto& p = GetParam();
    auto& engine = get_test_engine();
    const float eps = 1e-12f;
    const tensor shape{1, 4, 3, 2};  // b, f, x, y
    const size_t b = 1, f = 4, y = 2, x = 3;
    const size_t spatial = y * x;

    // Sum of squares of every normalized slice is below 1e-4 (5.8e-6..4.9e-5); values are normal f16 numbers.
    std::vector<float> input_vals(b * f * spatial);
    for (size_t i = 0; i < input_vals.size(); ++i) {
        const float v = p.zero_input ? 0.f : (i % 2 ? -1.f : 1.f) * static_cast<float>(i + 1) * 1e-4f;
        input_vals[i] = p.data_type == data_types::f16 ? static_cast<float>(ov::float16(v)) : v;
    }

    std::vector<float> expected(input_vals.size());
    if (p.across_spatial) {
        double sum = 0.0;
        for (auto v : input_vals)
            sum += static_cast<double>(v) * v;
        for (size_t i = 0; i < input_vals.size(); ++i)
            expected[i] = static_cast<float>(input_vals[i] / std::sqrt(sum + eps));
    } else {
        for (size_t s = 0; s < spatial; ++s) {
            double sum = 0.0;
            for (size_t c = 0; c < f; ++c)
                sum += static_cast<double>(input_vals[c * spatial + s]) * input_vals[c * spatial + s];
            for (size_t c = 0; c < f; ++c)
                expected[c * spatial + s] = static_cast<float>(input_vals[c * spatial + s] / std::sqrt(sum + eps));
        }
    }

    auto input = engine.allocate_memory({p.data_type, format::bfyx, shape});
    auto scale = engine.allocate_memory({p.data_type, format::bfyx, tensor{1}});
    if (p.data_type == data_types::f16) {
        std::vector<ov::float16> input_f16(input_vals.begin(), input_vals.end());
        set_values(input, input_f16);
        set_values(scale, {ov::float16(1.f)});
    } else {
        set_values(input, input_vals);
        set_values(scale, {1.f});
    }

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("scale", scale));
    topology.add(normalize("normalize", input_info("input"), "scale", p.across_spatial, eps));
    topology.add(reorder("output", input_info("normalize"), format::bfyx, data_types::f32));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("output").get_memory();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr.size(), expected.size());
    const float tolerance = p.data_type == data_types::f16 ? 2e-3f : 1e-5f;
    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_NEAR(expected[i], output_ptr[i], tolerance) << "at index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_normalize_small_norm,
                         normalize_small_norm,
                         testing::Values(normalize_small_norm_params{data_types::f32, false},
                                         normalize_small_norm_params{data_types::f32, true},
                                         normalize_small_norm_params{data_types::f16, false},
                                         normalize_small_norm_params{data_types::f16, true},
                                         normalize_small_norm_params{data_types::f32, false, true},
                                         normalize_small_norm_params{data_types::f32, true, true},
                                         normalize_small_norm_params{data_types::f16, false, true},
                                         normalize_small_norm_params{data_types::f16, true, true}));

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TYPED_TEST(normalize_basic, basic_cached) {
    this->execute(true);
}
#else
template <typename NormalizeInput>
struct normalize_basic_cached : public normalize_basic<NormalizeInput> {
};

using format_types_cached = testing::Types<normalize_input_types<format::bfyx, float, false>>;

TYPED_TEST_SUITE(normalize_basic_cached, format_types_cached);

TYPED_TEST(normalize_basic_cached, basic) {
    this->execute(true);
}
#endif
