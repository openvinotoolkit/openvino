// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "intel_gpu/primitives/roll.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace tests;

namespace {

template <typename T>
struct roll_test : testing::Test {
    static std::vector<T> GenInput(size_t size) {
        std::vector<T> result;
        for (size_t i = 1; i <= size; ++i) {
            result.push_back(i);
        }
        return result;
    }

    void TearDown() override {
        auto& engine = get_test_engine();

        const auto input_format = format::get_default_format(input_shape.size());
        const layout data_layout(type_to_data_type<T>::value, input_format, tensor{input_shape});
        const auto input_data = GenInput(data_layout.get_linear_size());
        auto input = engine.allocate_memory(data_layout);
        set_values(input, input_data);

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(roll("roll", "input", tensor{shift}));

        network network(engine, topology);
        network.set_input_data("input", input);
        const auto outputs = network.execute();

        EXPECT_EQ(outputs.size(), size_t(1));
        EXPECT_EQ(outputs.begin()->first, "roll");

        auto output = outputs.at("roll").get_memory();
        cldnn::mem_lock<T> output_ptr(output, get_test_stream());
        ASSERT_EQ(output_ptr.size(), expected_output.size());

        for (size_t i = 0; i < output_ptr.size(); ++i) {
            EXPECT_EQ(expected_output[i], output_ptr[i]);
        }
    }

    std::vector<int32_t> input_shape;
    std::vector<int32_t> shift;
    std::vector<T> expected_output;
};

using test_types = testing::Types<int8_t, uint8_t, int32_t, int64_t, float>;
TYPED_TEST_SUITE(roll_test, test_types);

TYPED_TEST(roll_test, example) {
    this->input_shape = {4, 3, 1, 1};
    this->shift = {1, 0, 0, 0};
    this->expected_output = {10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9};
}

TYPED_TEST(roll_test, bfyx) {
    this->input_shape = {2, 3, 1, 2};
    this->shift = {1, 2, 0, 5};
    this->expected_output = {10, 9, 12, 11, 8, 7, 4, 3, 6, 5, 2, 1};
}

TYPED_TEST(roll_test, bfzyx) {
    this->input_shape = {1, 1, 3, 3, 2};
    this->shift = {1, 2, 10, 23, 6};
    this->expected_output = {6, 4, 5, 9, 7, 8, 3, 1, 2, 15, 13, 14, 18, 16, 17, 12, 10, 11};
}

TYPED_TEST(roll_test, bfwzyx) {
    this->input_shape = {2, 1, 1, 3, 2, 3};
    this->shift = {0, 58, 27, 9, 99, 13};
    this->expected_output = {16, 17, 18, 13, 14, 15, 4,  5,  6,  1,  2,  3,  10, 11, 12, 7,  8,  9,
                             34, 35, 36, 31, 32, 33, 22, 23, 24, 19, 20, 21, 28, 29, 30, 25, 26, 27};
}

}  // namespace
