// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/test_tools.hpp"
#include "gtest/gtest.h"
#include "openvino/core/model.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/runtime/tensor_util.hpp"

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

template <typename element_type>
class ParametredOffloadTensorTest : public ::testing::Test {
public:
    static constexpr ov::element::Type ov_type = ov::element::from<element_type>();

    void SetUp() override {
        shape = {10, 20, 30, 40};
        auto static_shape = shape.get_shape();
        initial_tensor = Tensor(ov_type, static_shape);
        std::vector<float> init_values(initial_tensor.get_size());
        ov::test::utils::fill_data_random(init_values.data(), initial_tensor.get_size(), 10, 0, 100);

        std::transform(init_values.begin(),
                       init_values.end(),
                       element::iterator<ov_type>(initial_tensor.data()),
                       [](float value) {
                           return static_cast<element_type>(value);
                       });

        file_name = ov::test::utils::generateTestFilePrefix();
    }

    void remove_file() {
        if (std::filesystem::exists(file_name))
            std::filesystem::remove(file_name);
    }

    ov::PartialShape shape;
    ov::Tensor initial_tensor;
    std::filesystem::path file_name;
};

TYPED_TEST_SUITE_P(ParametredOffloadTensorTest);

TYPED_TEST_P(ParametredOffloadTensorTest, save_tensor) {
    auto data_size = this->initial_tensor.get_byte_size();
    EXPECT_NO_THROW(save_tensor_data(this->initial_tensor, this->file_name));
    ASSERT_TRUE(std::filesystem::exists(this->file_name));
    {
        std::ifstream fin(this->file_name, std::ios::binary);
        std::vector<char> file_data(data_size);
        fin.read(reinterpret_cast<char*>(file_data.data()), data_size);
        EXPECT_EQ(0, memcmp(file_data.data(), this->initial_tensor.data(), data_size));
    }
    this->remove_file();
}

TYPED_TEST_P(ParametredOffloadTensorTest, read_tensor) {
    auto data_size = this->initial_tensor.get_byte_size();
    {
        std::ofstream fout(this->file_name, std::ios::binary);
        fout.write(reinterpret_cast<char*>(this->initial_tensor.data()), data_size);
    }
    ASSERT_TRUE(std::filesystem::exists(this->file_name));

    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(this->file_name, this->ov_type, this->shape, 0, true));
        EXPECT_EQ(0, memcmp(tensor.data(), this->initial_tensor.data(), data_size));
    }
    {
        ov::Tensor tensor(this->ov_type, this->shape.get_shape());
        EXPECT_NO_THROW(read_tensor_data(this->file_name, tensor, 0));
        EXPECT_EQ(0, memcmp(tensor.data(), this->initial_tensor.data(), data_size));
    }
    this->remove_file();
}

TYPED_TEST_P(ParametredOffloadTensorTest, create_mmaped_tensor) {
    auto data_size = this->initial_tensor.get_byte_size();
    {
        auto tensor = ov::create_mmaped_tensor(this->initial_tensor, this->file_name);
        ASSERT_TRUE(std::filesystem::exists(this->file_name));
        EXPECT_EQ(0, memcmp(tensor.data(), this->initial_tensor.data(), data_size));
    }
    ASSERT_FALSE(std::filesystem::exists(this->file_name));
}

REGISTER_TYPED_TEST_SUITE_P(ParametredOffloadTensorTest, save_tensor, read_tensor, create_mmaped_tensor);

using TypesToTest = ::testing::Types<
    float,
    double,
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
    ov::bfloat16,
    ov::float8_e4m3,
    ov::float8_e5m2,
    ov::float4_e2m1,
    ov::float8_e8m0>;

INSTANTIATE_TYPED_TEST_SUITE_P(OffloadTensorTest, ParametredOffloadTensorTest, TypesToTest);

TEST(OffloadTensorTest, string_tensor_throws) {
    ov::Tensor str_tensor(ov::element::string, ov::Shape{1});
    auto file_name = ov::test::utils::generateTestFilePrefix();
    {
        EXPECT_THROW(save_tensor_data(str_tensor, file_name), ov::Exception);
        ASSERT_FALSE(std::filesystem::exists(file_name));
    }
    {
        std::ofstream fout(file_name, std::ios::binary);
        fout << "Hello, world!";
        fout.close();
        ASSERT_TRUE(std::filesystem::exists(file_name));
        EXPECT_THROW(read_tensor_data(file_name, str_tensor), ov::Exception);
        std::filesystem::remove(file_name);
    }
    {
        EXPECT_THROW(std::ignore = ov::create_mmaped_tensor(str_tensor, file_name), ov::Exception);
        ASSERT_FALSE(std::filesystem::exists(file_name));
    }
}

class FunctionalOffloadTensorTest : public ::testing::Test {
public:
    void SetUp() override {
        initial_tensor = Tensor(ov_type, shape.get_shape());
        std::vector<float> init_values(initial_tensor.get_size());
        ov::test::utils::fill_data_random(init_values.data(), initial_tensor.get_size(), 10, 0, 100);
        std::memcpy(initial_tensor.data(), init_values.data(), initial_tensor.get_byte_size());

        file_name = ov::test::utils::generateTestFilePrefix();
    }

    void remove_file() {
        if (std::filesystem::exists(file_name))
            std::filesystem::remove(file_name);
    }

    ov::element::Type ov_type{ov::element::f32};
    ov::PartialShape shape{1, 2, 3, 4};
    ov::Tensor initial_tensor;
    std::string file_name;
};

TEST_F(FunctionalOffloadTensorTest, read_with_offset) {
    auto data_size = initial_tensor.get_byte_size();
    {
        float dummy = 0;
        std::ofstream fout(file_name, std::ios::binary);
        fout.write(reinterpret_cast<char*>(&dummy), sizeof(float));
        fout.write(reinterpret_cast<char*>(initial_tensor.data()), data_size);
    }
    ASSERT_TRUE(std::filesystem::exists(file_name));

    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(file_name, ov_type, shape, sizeof(float), true));
        EXPECT_EQ(0, memcmp(tensor.data(), initial_tensor.data(), data_size));
    }
    remove_file();
}

TEST_F(FunctionalOffloadTensorTest, read_without_mmap) {
    auto data_size = initial_tensor.get_byte_size();
    {
        std::ofstream fout(file_name, std::ios::binary);
        fout.write(reinterpret_cast<char*>(initial_tensor.data()), data_size);
    }
    ASSERT_TRUE(std::filesystem::exists(file_name));

    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(file_name, ov_type, shape, 0, false));
        EXPECT_EQ(0, memcmp(tensor.data(), initial_tensor.data(), data_size));
    }
    remove_file();
}

TEST_F(FunctionalOffloadTensorTest, read_without_mmap_and_with_offset) {
    auto data_size = initial_tensor.get_byte_size();
    {
        float dummy = 0;
        std::ofstream fout(file_name, std::ios::binary);
        fout.write(reinterpret_cast<char*>(&dummy), sizeof(float));
        fout.write(reinterpret_cast<char*>(initial_tensor.data()), data_size);
    }
    ASSERT_TRUE(std::filesystem::exists(file_name));

    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(file_name, ov_type, shape, sizeof(float), false));
        EXPECT_EQ(0, memcmp(tensor.data(), initial_tensor.data(), data_size));
    }
    remove_file();
}

TEST_F(FunctionalOffloadTensorTest, read_small_file) {
    auto data_size = initial_tensor.get_byte_size();
    {
        std::ofstream fout(file_name, std::ios::binary);
        fout.write(reinterpret_cast<char*>(initial_tensor.data()), data_size - 1);
    }
    ASSERT_TRUE(std::filesystem::exists(file_name));

    {
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape, 0, true), ov::Exception);
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape, 0, false), ov::Exception);
    }
    remove_file();
}

TEST_F(FunctionalOffloadTensorTest, read_too_big_offset) {
    auto data_size = initial_tensor.get_byte_size();
    {
        std::ofstream fout(file_name, std::ios::binary);
        fout.write(reinterpret_cast<char*>(initial_tensor.data()), data_size);
    }
    ASSERT_TRUE(std::filesystem::exists(file_name));

    {
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape, 1, true), ov::Exception);
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape, 1, false), ov::Exception);
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape, data_size, true), ov::Exception);
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape, data_size, false), ov::Exception);
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape, data_size + 1, true), ov::Exception);
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape, data_size + 1, false), ov::Exception);
    }
    remove_file();
}

TEST_F(FunctionalOffloadTensorTest, read_dynamic_shape) {
    auto data_size = initial_tensor.get_byte_size();
    {
        std::ofstream fout(file_name, std::ios::binary);
        fout.write(reinterpret_cast<char*>(initial_tensor.data()), data_size);
    }
    ASSERT_TRUE(std::filesystem::exists(file_name));

    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(file_name, ov_type, PartialShape{Dimension::dynamic()}, 0, true));
        EXPECT_NO_THROW(tensor = read_tensor_data(file_name, ov_type, PartialShape{Dimension::dynamic()}, 0, false));
        EXPECT_EQ(0, memcmp(tensor.data(), initial_tensor.data(), data_size));
    }
    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(file_name));
        EXPECT_EQ(0, memcmp(tensor.data(), initial_tensor.data(), data_size));
    }
    remove_file();
}

TEST_F(FunctionalOffloadTensorTest, read_1_dynamic_dimension) {
    auto data_size = initial_tensor.get_byte_size();
    {
        std::ofstream fout(file_name, std::ios::binary);
        fout.write(reinterpret_cast<char*>(initial_tensor.data()), data_size);
    }
    ASSERT_TRUE(std::filesystem::exists(file_name));

    {
        ov::Tensor tensor;
        auto shape_with_1_dynamic_dimension = shape;
        size_t dynamic_dimension_number = shape_with_1_dynamic_dimension.size() - 1;
        shape_with_1_dynamic_dimension[dynamic_dimension_number] = -1;
        EXPECT_NO_THROW(tensor = read_tensor_data(file_name, ov_type, shape_with_1_dynamic_dimension, 0, true));
        EXPECT_EQ(tensor.get_shape()[dynamic_dimension_number], initial_tensor.get_shape()[dynamic_dimension_number]);
        EXPECT_EQ(0, memcmp(tensor.data(), initial_tensor.data(), data_size));
    }
    remove_file();
}

TEST_F(FunctionalOffloadTensorTest, read_wrong_dynamic_shape) {
    auto data_size = initial_tensor.get_byte_size();
    {
        std::ofstream fout(file_name, std::ios::binary);
        fout.write(reinterpret_cast<char*>(initial_tensor.data()), data_size);
    }
    ASSERT_TRUE(std::filesystem::exists(file_name));

    {
        auto shape_with_1_dynamic_dimension = shape;
        shape_with_1_dynamic_dimension[shape_with_1_dynamic_dimension.size() - 1] = -1;
        shape_with_1_dynamic_dimension[shape_with_1_dynamic_dimension.size() - 2] = 100;
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape_with_1_dynamic_dimension, 0, true), ov::Exception);
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape_with_1_dynamic_dimension, 0, false), ov::Exception);
    }
    remove_file();
}

TEST_F(FunctionalOffloadTensorTest, read_type_doesnt_fit_file_size) {
    auto data_size = initial_tensor.get_byte_size();
    {
        std::ofstream fout(file_name, std::ios::binary);
        fout.write(reinterpret_cast<char*>(initial_tensor.data()), data_size - 1);
    }
    ASSERT_TRUE(std::filesystem::exists(file_name));

    {
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov::element::f32), ov::Exception);
    }
    remove_file();
}
}  // namespace test
}  // namespace ov
