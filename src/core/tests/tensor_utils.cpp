// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::test {
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

        ov::reference::convert(ov::element::iterator<ov::element::f32>(init_values.data()),
                               ov::element::iterator<ov_type>(initial_tensor.data()),
                               shape_size(static_shape));

        file_name = ov::test::utils::generateTestFilePrefix();
        data_size = this->initial_tensor.get_byte_size();
        {
            std::ofstream fout(this->file_name, std::ios::binary);
            fout.write(reinterpret_cast<char*>(this->initial_tensor.data()), data_size);
        }
        ASSERT_TRUE(std::filesystem::exists(this->file_name));
    }

    void TearDown() override {
        remove_file();
    }

    void remove_file() {
        if (std::filesystem::exists(file_name))
            std::filesystem::remove(file_name);
    }

    ov::PartialShape shape;
    ov::Tensor initial_tensor;
    std::filesystem::path file_name;
    size_t data_size;
};

TYPED_TEST_SUITE_P(ParametredOffloadTensorTest);

TYPED_TEST_P(ParametredOffloadTensorTest, read_tensor) {
    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(this->file_name, this->ov_type, this->shape, 0));
        EXPECT_EQ(0, memcmp(tensor.data(), this->initial_tensor.data(), this->data_size));
    }
    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(this->file_name, this->ov_type, this->shape, 0, false));
        EXPECT_EQ(0, memcmp(tensor.data(), this->initial_tensor.data(), this->data_size));
    }
}

REGISTER_TYPED_TEST_SUITE_P(ParametredOffloadTensorTest, read_tensor);

using TypesToTest = ::testing::Types<float,
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

class FunctionalOffloadTensorTest : public ::testing::Test {
public:
    void SetUp() override {
        auto elements_number = ov::shape_size(shape.get_shape());
        data_size = elements_number * sizeof(float);
        init_values.resize(elements_number);
        ov::test::utils::fill_data_random(init_values.data(), elements_number, 10, 0, 100);

        file_name = ov::test::utils::generateTestFilePrefix();
        {
            std::ofstream fout(file_name, std::ios::binary);
            fout.write(reinterpret_cast<char*>(init_values.data()), data_size);
        }
        ASSERT_TRUE(std::filesystem::exists(file_name));
    }

    void TearDown() override {
        remove_file();
    }

    void remove_file() {
        if (std::filesystem::exists(file_name))
            std::filesystem::remove(file_name);
    }

    ov::element::Type ov_type{ov::element::f32};
    ov::PartialShape shape{1, 2, 3, 4};
    size_t data_size;
    std::vector<float> init_values;
    std::string file_name;
};

TEST_F(FunctionalOffloadTensorTest, string_tensor_throws) {
    {
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov::element::string), ov::Exception);
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov::element::string, PartialShape::dynamic(1), 0, false),
                     ov::Exception);
    }
}

TEST_F(FunctionalOffloadTensorTest, read_with_offset) {
    {
        float dummy = 0;
        std::ofstream fout(file_name, std::ios::binary);
        fout.write(reinterpret_cast<char*>(&dummy), sizeof(float));
        fout.write(reinterpret_cast<char*>(init_values.data()), data_size);
    }
    ASSERT_TRUE(std::filesystem::exists(file_name));

    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(file_name, ov_type, shape, sizeof(float)));
        EXPECT_EQ(0, memcmp(tensor.data(), init_values.data(), data_size));
    }
    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(file_name, ov_type, shape, sizeof(float), false));
        EXPECT_EQ(0, memcmp(tensor.data(), init_values.data(), data_size));
    }
}

TEST_F(FunctionalOffloadTensorTest, read_small_file) {
    auto new_shape = shape;
    new_shape[0] = 10;
    {
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, new_shape, 0), ov::Exception);
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, new_shape, 0, false), ov::Exception);
    }
}

TEST_F(FunctionalOffloadTensorTest, read_too_big_offset) {
    {
        // offset + data_size > file_size
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape, 1), ov::Exception);
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape, 1, false), ov::Exception);
        // offset == file_size
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape, data_size), ov::Exception);
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape, data_size, false), ov::Exception);
        // offset > file_size
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape, data_size + 1), ov::Exception);
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape, data_size + 1, false), ov::Exception);
    }
}

TEST_F(FunctionalOffloadTensorTest, read_dynamic_shape) {
    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(file_name, ov_type, PartialShape::dynamic(1), 0));
        EXPECT_EQ(0, memcmp(tensor.data(), init_values.data(), data_size));
    }
    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(file_name, ov_type, PartialShape::dynamic(1), 0, false));
        EXPECT_EQ(0, memcmp(tensor.data(), init_values.data(), data_size));
    }
    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(file_name));
        EXPECT_EQ(0, memcmp(tensor.data(), init_values.data(), data_size));
    }
    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(file_name, ov::element::u8, PartialShape::dynamic(1), 0, false));
        EXPECT_EQ(0, memcmp(tensor.data(), init_values.data(), data_size));
    }
}

TEST_F(FunctionalOffloadTensorTest, read_1_dynamic_dimension) {
    auto shape_with_1_dynamic_dimension = shape;
    size_t dynamic_dimension_number = shape_with_1_dynamic_dimension.size() - 1;
    shape_with_1_dynamic_dimension[dynamic_dimension_number] = -1;
    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(file_name, ov_type, shape_with_1_dynamic_dimension, 0));
        EXPECT_EQ(tensor.get_shape()[dynamic_dimension_number], shape.get_shape()[dynamic_dimension_number]);
        EXPECT_EQ(0, memcmp(tensor.data(), init_values.data(), data_size));
    }
    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(file_name, ov_type, shape_with_1_dynamic_dimension, 0, false));
        EXPECT_EQ(tensor.get_shape()[dynamic_dimension_number], shape.get_shape()[dynamic_dimension_number]);
        EXPECT_EQ(0, memcmp(tensor.data(), init_values.data(), data_size));
    }
}

TEST_F(FunctionalOffloadTensorTest, read_wrong_dynamic_shape) {
    {
        auto shape_with_1_dynamic_dimension = shape;
        shape_with_1_dynamic_dimension[shape_with_1_dynamic_dimension.size() - 1] = -1;
        shape_with_1_dynamic_dimension[shape_with_1_dynamic_dimension.size() - 2] = 100;
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape_with_1_dynamic_dimension, 0),
                     ov::Exception);
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, shape_with_1_dynamic_dimension, 0, false),
                     ov::Exception);
    }
}

TEST_F(FunctionalOffloadTensorTest, read_type_doesnt_fit_file_size) {
    {
        std::ofstream fout(file_name, std::ios::binary);
        fout.write(reinterpret_cast<char*>(init_values.data()), data_size - 1);
    }
    ASSERT_TRUE(std::filesystem::exists(file_name));

    {
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov::element::f32), ov::Exception);
        EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov::element::f32, PartialShape::dynamic(1), 0, false),
                     ov::Exception);
    }
}

TEST_F(FunctionalOffloadTensorTest, read_null_shape) {
    auto new_shape = shape;
    // 1 dynamic dimension and 1 null dimension
    new_shape[0] = 0;
    new_shape[1] = Dimension::dynamic();
    { EXPECT_THROW(std::ignore = read_tensor_data(file_name, ov_type, new_shape, 0), ov::Exception); }
}
}  // namespace ov::test
