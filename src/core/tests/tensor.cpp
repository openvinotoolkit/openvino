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

template <typename TypeParam>
class OffloadedTensorTest : public ::testing::Test {
public:
    using path_type = typename std::tuple_element<0, TypeParam>::type;
    using element_type = typename std::tuple_element<1, TypeParam>::type;
    static constexpr ov::element::Type ov_type = ov::element::from<element_type>();

    void SetUp() override {
        shape = {10, 20, 30, 40};
        auto static_shape = shape.get_shape();
        initial_tensor = Tensor(ov_type, static_shape);
        std::vector<float> init_values(initial_tensor.get_size());
        ov::test::utils::fill_data_random(init_values.data(), initial_tensor.get_size(), 10, 0, 100);

        std::copy(init_values.begin(), init_values.end(), element::iterator<ov_type>(initial_tensor.data()));

        file_name_str = ov::test::utils::generateTestFilePrefix();
        file_name = path_type(file_name_str.begin(), file_name_str.end());
    }

    void remove_file() {
        if (std::filesystem::exists(file_name))
            std::filesystem::remove(file_name);
    }

    ov::PartialShape shape;
    ov::Tensor initial_tensor;
    path_type file_name;
    std::string file_name_str;
};

TYPED_TEST_SUITE_P(OffloadedTensorTest);

TYPED_TEST_P(OffloadedTensorTest, save_tensor) {
    auto data_size = this->initial_tensor.get_byte_size();
    EXPECT_NO_THROW(save_tensor_data(this->initial_tensor, this->file_name));
    ASSERT_TRUE(std::filesystem::exists(this->file_name));
    {
        std::ifstream fin(this->file_name_str, std::ios::binary);
        std::vector<char> file_data(data_size);
        fin.read(reinterpret_cast<char*>(file_data.data()), data_size);
        EXPECT_EQ(0, memcmp(file_data.data(), this->initial_tensor.data(), data_size));
    }
    this->remove_file();
}

TYPED_TEST_P(OffloadedTensorTest, read_tensor) {
    auto data_size = this->initial_tensor.get_byte_size();
    {
        std::ofstream fout(this->file_name_str, std::ios::binary);
        fout.write(reinterpret_cast<char*>(this->initial_tensor.data()), data_size);
    }
    ASSERT_TRUE(std::filesystem::exists(this->file_name));

    {
        ov::Tensor tensor;
        EXPECT_NO_THROW(tensor = read_tensor_data(this->file_name, this->ov_type, this->shape, 0, true));
        EXPECT_EQ(0, memcmp(tensor.data(), this->initial_tensor.data(), data_size));
    }
    this->remove_file();
}

TYPED_TEST_P(OffloadedTensorTest, create_mmaped_tensor) {
    auto data_size = this->initial_tensor.get_byte_size();
    {
        auto tensor = ov::create_mmaped_tensor(this->initial_tensor, this->file_name);
        ASSERT_TRUE(std::filesystem::exists(this->file_name));
        EXPECT_EQ(0, memcmp(tensor.data(), this->initial_tensor.data(), data_size));
    }
    ASSERT_FALSE(std::filesystem::exists(this->file_name));
}

REGISTER_TYPED_TEST_CASE_P(OffloadedTensorTest, save_tensor, read_tensor, create_mmaped_tensor);

using TypesToTest = ::testing::Types<
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
    std::tuple<std::wstring, float>,
    std::tuple<std::wstring, double>,
    std::tuple<std::wstring, int8_t>,
    std::tuple<std::wstring, int16_t>,
    std::tuple<std::wstring, int32_t>,
    std::tuple<std::wstring, int64_t>,
    std::tuple<std::wstring, uint8_t>,
    std::tuple<std::wstring, uint16_t>,
    std::tuple<std::wstring, uint32_t>,
    std::tuple<std::wstring, uint64_t>,
    std::tuple<std::wstring, ov::bfloat16>,
    std::tuple<std::wstring, ov::float8_e4m3>,
    std::tuple<std::wstring, ov::float8_e5m2>,
    std::tuple<std::wstring, ov::float4_e2m1>,
    std::tuple<std::wstring, ov::float8_e8m0>,
#endif
    std::tuple<std::string, float>,
    std::tuple<std::string, double>,
    std::tuple<std::string, int8_t>,
    std::tuple<std::string, int16_t>,
    std::tuple<std::string, int32_t>,
    std::tuple<std::string, int64_t>,
    std::tuple<std::string, uint8_t>,
    std::tuple<std::string, uint16_t>,
    std::tuple<std::string, uint32_t>,
    std::tuple<std::string, uint64_t>,
    std::tuple<std::string, ov::bfloat16>,
    std::tuple<std::string, ov::float8_e4m3>,
    std::tuple<std::string, ov::float8_e5m2>,
    std::tuple<std::string, ov::float4_e2m1>,
    std::tuple<std::string, ov::float8_e8m0>,
    std::tuple<std::filesystem::path, float>,
    std::tuple<std::filesystem::path, double>,
    std::tuple<std::filesystem::path, int8_t>,
    std::tuple<std::filesystem::path, int16_t>,
    std::tuple<std::filesystem::path, int32_t>,
    std::tuple<std::filesystem::path, int64_t>,
    std::tuple<std::filesystem::path, uint8_t>,
    std::tuple<std::filesystem::path, uint16_t>,
    std::tuple<std::filesystem::path, uint32_t>,
    std::tuple<std::filesystem::path, uint64_t>,
    std::tuple<std::filesystem::path, ov::bfloat16>,
    std::tuple<std::filesystem::path, ov::float8_e4m3>,
    std::tuple<std::filesystem::path, ov::float8_e5m2>,
    std::tuple<std::filesystem::path, ov::float4_e2m1>,
    std::tuple<std::filesystem::path, ov::float8_e8m0>
>;

INSTANTIATE_TYPED_TEST_SUITE_P(test_types, OffloadedTensorTest, TypesToTest);

}  // namespace test
}  // namespace ov
