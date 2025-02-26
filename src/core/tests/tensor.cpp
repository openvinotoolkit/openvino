// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/test_tools.hpp"
#include "gtest/gtest.h"
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/runtime/tensor_util.hpp"
#include "openvino/op/constant.hpp"

using namespace std;

namespace ov {
namespace test {
TEST(tensor, tensor_names) {
    auto arg0 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = make_shared<ov::op::v0::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f0 = make_shared<Model>(relu, ParameterVector{arg0});

    ASSERT_EQ(arg0->get_output_tensor(0).get_names(), relu->get_input_tensor(0).get_names());
    ASSERT_EQ(arg0->get_output_tensor(0).get_names(), relu->input_value(0).get_tensor().get_names());
    ASSERT_EQ(f0->get_result()->get_input_tensor(0).get_names(), relu->get_output_tensor(0).get_names());
    ASSERT_EQ(f0->get_result()->input_value(0).get_tensor().get_names(), relu->get_output_tensor(0).get_names());
}

TEST(tensor, create_tensor_with_zero_dims_check_stride) {
    ov::Shape shape = {0, 0, 0, 0};
    auto tensor = ov::Tensor(element::f32, shape);
    EXPECT_EQ(!!tensor, true);
    auto stride = tensor.get_strides();
    EXPECT_EQ(stride.size(), shape.size());
    EXPECT_EQ(stride.back(), 0);
    EXPECT_EQ(tensor.is_continuous(), true);
}

TEST(tensor, get_byte_size_u2_less_than_min_storage_unit) {
    const auto tensor = Tensor(element::u2, Shape{3});
    EXPECT_EQ(tensor.get_byte_size(), 1);
}

TEST(tensor, get_byte_size_u2_even_div_by_storage_unit) {
    const auto tensor = Tensor(element::u2, Shape{16});
    EXPECT_EQ(tensor.get_byte_size(), 4);
}

TEST(tensor, get_byte_size_u2_not_even_div_by_storage_unit) {
    const auto tensor = Tensor(element::u2, Shape{17});
    EXPECT_EQ(tensor.get_byte_size(), 5);
}

TEST(tensor, get_byte_size_u3_less_than_min_storage_unit) {
    const auto tensor = Tensor(element::u3, Shape{3});
    EXPECT_EQ(tensor.get_byte_size(), 3);
}

TEST(tensor, get_byte_size_u3_even_div_by_storage_unit) {
    const auto tensor = Tensor(element::u3, Shape{16});
    EXPECT_EQ(tensor.get_byte_size(), 2 * 3);
}

TEST(tensor, get_byte_size_u3_not_even_div_by_storage_unit) {
    const auto tensor = Tensor(element::u3, Shape{17});
    EXPECT_EQ(tensor.get_byte_size(), 3 + 2 * 3);
}

TEST(tensor, get_byte_size_u6_less_than_min_storage_unit) {
    const auto tensor = Tensor(element::u6, Shape{3});
    EXPECT_EQ(tensor.get_byte_size(), 3);
}

TEST(tensor, get_byte_size_u6_even_div_by_storage_unit) {
    const auto tensor = Tensor(element::u6, Shape{16});
    EXPECT_EQ(tensor.get_byte_size(), 4 * 3);
}

TEST(tensor, get_byte_size_u6_not_even_div_by_storage_unit) {
    const auto tensor = Tensor(element::u6, Shape{17});
    EXPECT_EQ(tensor.get_byte_size(), 3 + 4 * 3);
}



TEST(tensor, mmaped_tensor_test) {
    {
        std::shared_ptr<ov::Node> constant;
        {
            auto init_tensor = ov::Tensor(ov::element::i32, ov::Shape({1, 1, 2, 2}));
            std::vector<int> data = {1, 2, 3, 4};
            std::memcpy(init_tensor.data(), data.data(), data.size() * sizeof(int));
    
            auto new_tensor = ov::create_mmaped_tensor(init_tensor, std::string{"tmp_file"});
            constant = std::make_shared<ov::op::v0::Constant>(new_tensor);
        }
        // mmaped object is live here and tmp_file exists
    }
    // mmaped object is destroyed here and tmp_file is deleted
}




}  // namespace test
}  // namespace ov
