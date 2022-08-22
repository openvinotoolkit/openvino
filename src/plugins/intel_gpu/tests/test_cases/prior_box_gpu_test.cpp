// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/prior_box.hpp>
#include <random>
#include <vector>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

template <class OutputType, class InputType>
using prior_box_param = std::tuple<format,                       // Input and output format
                                   std::vector<InputType>,       // output_size
                                   std::vector<InputType>,       // image_size
                                   cldnn::prior_box_attributes,  // attributes
                                   std::vector<OutputType>>;     // expected values

template <class OutputType, class InputType>
class PriorBoxGPUTest : public ::testing::TestWithParam<prior_box_param<OutputType, InputType>> {
public:
    void SetUp() override {
        format fmt{format::bfyx};
        std::vector<InputType> output_size{};
        std::vector<InputType> image_size{};
        cldnn::prior_box_attributes attributes;
        std::vector<OutputType> expected_values;

        std::tie(fmt, output_size, image_size, attributes, expected_values) = this->GetParam();

        memory::ptr output_size_layout = engine_.allocate_memory({type_to_data_type<InputType>::value, fmt, tensor{2}});
        set_values<InputType>(output_size_layout, output_size);
        memory::ptr image_size_layout = engine_.allocate_memory({type_to_data_type<InputType>::value, fmt, tensor{2}});
        set_values<InputType>(image_size_layout, image_size);

        topology tp;
        tp.add(data("output_size", output_size_layout));
        tp.add(data("image_size", image_size_layout));

        std::vector<primitive_id> inputs{"output_size", "image_size"};
        auto prior_box = cldnn::prior_box("prior_box",
                                          inputs,
                                          static_cast<int32_t>(output_size[0]),
                                          static_cast<int32_t>(output_size[1]),
                                          static_cast<int32_t>(image_size[0]),
                                          static_cast<int32_t>(image_size[1]),
                                          attributes,
                                          type_to_data_type<OutputType>::value);
        tp.add(prior_box);

        network network(engine_, tp);

        auto outputs = network.execute();

        EXPECT_EQ(outputs.begin()->first, "prior_box");

        auto output = outputs.at("prior_box").get_memory();

        cldnn::mem_lock<OutputType> output_ptr(output, get_test_stream());

        ASSERT_EQ(output_ptr.size(), expected_values.size());
        for (size_t i = 0; i < output_ptr.size(); ++i) {
            EXPECT_TRUE(are_equal(expected_values[i], output_ptr[i], 2e-3));
        }
    }

protected:
    engine& engine_ = get_test_engine();

private:
};

std::vector<format> four_d_formats{format::bfyx,
                                   format::bfzyx,
                                   format::bfwzyx,
                                   format::bfyx,
                                   format::bfzyx,
                                   format::bfwzyx};

using prior_box_test_4d_float_int32 = PriorBoxGPUTest<float, int32_t>;
TEST_P(prior_box_test_4d_float_int32, prior_box_test_4d_float_int32) {}
INSTANTIATE_TEST_SUITE_P(
    prior_box_test_4d_float_int32,
    prior_box_test_4d_float_int32,
    testing::Combine(
        testing::ValuesIn(four_d_formats),
        testing::Values(std::vector<int32_t>{2, 2}),
        testing::Values(std::vector<int32_t>{10, 10}),
        testing::Values(
            cldnn::prior_box_attributes{{2.0f}, {5.0f}, {1.5f}, {}, {}, {}, false, false, 0.0f, 0.0f, {}, true, false}),
        testing::Values(std::vector<float>{
            0.15, 0.15, 0.35, 0.35, 0.127526, 0.16835, 0.372474, 0.33165, 0.0918861, 0.0918861, 0.408114, 0.408114,
            0.65, 0.15, 0.85, 0.35, 0.627526, 0.16835, 0.872474, 0.33165, 0.591886,  0.0918861, 0.908114, 0.408114,
            0.15, 0.65, 0.35, 0.85, 0.127526, 0.66835, 0.372474, 0.83165, 0.0918861, 0.591886,  0.408114, 0.908114,
            0.65, 0.65, 0.85, 0.85, 0.627526, 0.66835, 0.872474, 0.83165, 0.591886,  0.591886,  0.908114, 0.908114,
            0.1,  0.1,  0.1,  0.1,  0.1,      0.1,     0.1,      0.1,     0.1,       0.1,       0.1,      0.1,
            0.1,  0.1,  0.1,  0.1,  0.1,      0.1,     0.1,      0.1,     0.1,       0.1,       0.1,      0.1,
            0.1,  0.1,  0.1,  0.1,  0.1,      0.1,     0.1,      0.1,     0.1,       0.1,       0.1,      0.1,
            0.1,  0.1,  0.1,  0.1,  0.1,      0.1,     0.1,      0.1,     0.1,       0.1,       0.1,      0.1})));
}  // anonymous namespace
