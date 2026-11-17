// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/normalize.hpp>
#include <iostream>
#include <vector>

#include "primitive_inst.h"
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

namespace {
template <typename InputType>
void test_normalize_across_x(const tensor& shape, format::type input_format, bool caching) {
    auto& engine = get_test_engine();
    const auto data_type = ov::element::from<InputType>();
    const size_t width = shape.spatial[0];
    const size_t height = shape.spatial[1];
    const size_t features = shape.feature[0];
    auto input = engine.allocate_memory({data_type, format::bfyx, shape});
    auto scales = engine.allocate_memory({data_types::f32, format::bfyx, tensor{1, static_cast<int32_t>(features), 1, 1}});
    std::vector<InputType> values(input->count());
    std::vector<float> scale_values(features);
    for (size_t feature = 0; feature < features; ++feature) {
        scale_values[feature] = 1.0f + 0.25f * static_cast<float>(feature % 5);
    }
    for (size_t index = 0; index < values.size(); ++index) {
        const auto row = index / width;
        const float value = row == 0 ? 0.0f : row == 1 ? 1e-4f : static_cast<float>(static_cast<int>(index % 23) - 11);
        values[index] = static_cast<InputType>(value);
    }
    set_values(input, values);
    set_values(scales, scale_values);

    const float epsilon = std::numeric_limits<float>::epsilon();
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(data("scales", scales));
    topology.add(reorder("formatted_input", input_info("input"), input_format, data_type));
    topology.add(normalize("normalize", input_info("formatted_input"), "scales", false, epsilon, true));
    topology.add(reorder("output", input_info("normalize"), format::bfyx, data_type));
    auto network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), caching);
    network->set_input_data("input", input);
    ASSERT_EQ(network->get_primitive("normalize")->get_impl()->get_kernel_name(), "normalize_gpu_across_x");
    auto output = network->execute().at("output").get_memory();
    mem_lock<InputType, mem_lock_type::read> output_values(output, get_test_stream());
    const float tolerance = data_type == ov::element::f16 ? 0.002f : 2e-6f;
    for (size_t row = 0; row < values.size() / width; ++row) {
        double sum = epsilon;
        for (size_t column = 0; column < width; ++column) {
            const float value = static_cast<float>(values[row * width + column]);
            sum += static_cast<double>(value) * value;
        }
        const auto scale = scale_values[(row / height) % features];
        for (size_t column = 0; column < width; ++column) {
            const auto index = row * width + column;
            const float expected = static_cast<float>(values[index]) / std::sqrt(sum) * scale;
            ASSERT_NEAR(static_cast<float>(output_values[index]), expected, tolerance) << "index=" << index;
        }
    }
}
}  // namespace

TEST(normalize_across_x, fp32_onnx_shape) {
    test_normalize_across_x<float>(tensor{1, 512, 5120, 1}, format::bfyx, false);
}

TEST(normalize_across_x, fp16_onnx_shape) {
    test_normalize_across_x<ov::float16>(tensor{1, 512, 5120, 1}, format::bfyx, false);
}

TEST(normalize_across_x, layouts_and_tail) {
    for (auto input_format : {format::bfyx, format::byxf, format::yxfb, format::b_fs_yx_fsv16}) {
        test_normalize_across_x<float>(tensor{2, 3, 17, 2}, input_format, false);
        test_normalize_across_x<ov::float16>(tensor{2, 3, 17, 2}, input_format, false);
    }
}

TEST(normalize_across_x, single_element_rows) {
    test_normalize_across_x<float>(tensor{2, 3, 1, 2}, format::bfyx, false);
}

TEST(normalize_across_x, cached) {
    test_normalize_across_x<float>(tensor{2, 3, 17, 2}, format::bfyx, true);
}
