// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>

#include "ngraph/function.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/serialize.hpp"
#include "util/test_common.hpp"

class SerializationConstantCompressionTest : public ov::test::TestsCommon {
protected:
    std::string test_name = GetTestName();
    std::string m_out_xml_path_1 = test_name + "1" + ".xml";
    std::string m_out_bin_path_1 = test_name + "1" + ".bin";

    void TearDown() override {
        std::remove(m_out_xml_path_1.c_str());
        std::remove(m_out_bin_path_1.c_str());
    }

    std::uintmax_t file_size(std::ifstream& f) {
        // get length of file:
        const auto pos_to_restore = f.tellg();
        f.seekg(0, f.end);
        std::uintmax_t length = f.tellg();
        f.seekg(pos_to_restore, f.beg);
        return length;
    }
};

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsI32) {
    constexpr int unique_const_count = 1;
    const ngraph::Shape shape{2, 2, 2};

    auto A = ov::op::v0::Constant::create(ngraph::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::op::v0::Constant::create(ngraph::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});

    auto ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B}, ngraph::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ngraph::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsI64) {
    constexpr int unique_const_count = 1;
    const ngraph::Shape shape{2, 2, 2};

    auto A = ov::op::v0::Constant::create(ngraph::element::i64, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::op::v0::Constant::create(ngraph::element::i64, shape, {1, 2, 3, 4, 5, 6, 7, 8});

    auto ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B}, ngraph::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ngraph::shape_size(shape) * sizeof(int64_t));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsFP16) {
    constexpr int unique_const_count = 1;
    const ngraph::Shape shape{2, 2, 2};

    auto A = ngraph::op::v0::Constant::create(ngraph::element::f16, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ngraph::op::v0::Constant::create(ngraph::element::f16, shape, {1, 2, 3, 4, 5, 6, 7, 8});

    auto ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B}, ngraph::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ngraph::shape_size(shape) * sizeof(ngraph::float16));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsFP32) {
    constexpr int unique_const_count = 1;
    const ngraph::Shape shape{2, 2, 2};

    auto A = ngraph::op::v0::Constant::create(ngraph::element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ngraph::op::v0::Constant::create(ngraph::element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});

    auto ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B}, ngraph::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ngraph::shape_size(shape) * sizeof(float));
}

TEST_F(SerializationConstantCompressionTest, NonIdenticalConstantsI64) {
    constexpr int unique_const_count = 2;
    const ngraph::Shape shape{2};

    // hash_combine returns the same hash for this two constants so we also check the content of arrays
    auto A = ngraph::op::v0::Constant::create(ngraph::element::i64, shape, {2, 2});
    auto B = ngraph::op::v0::Constant::create(ngraph::element::i64, shape, {0, 128});

    auto ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B}, ngraph::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ngraph::shape_size(shape) * sizeof(int64_t));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsTimesTwo) {
    constexpr int unique_const_count = 2;
    const ngraph::Shape shape{2, 2, 2};

    auto A = ngraph::op::v0::Constant::create(ngraph::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ngraph::op::v0::Constant::create(ngraph::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto C = ngraph::op::v0::Constant::create(ngraph::element::i32, shape, {0, 3, 1, 2, 5, 6, 25, 3});
    auto D = ngraph::op::v0::Constant::create(ngraph::element::i32, shape, {0, 3, 1, 2, 5, 6, 25, 3});

    auto ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B, C, D}, ngraph::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ngraph::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsTimesTwoMultipleOccurences) {
    constexpr int unique_const_count = 2;
    const ngraph::Shape shape{2, 2, 2};

    auto A = ngraph::op::v0::Constant::create(ngraph::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ngraph::op::v0::Constant::create(ngraph::element::i32, shape, {0, 3, 1, 2, 5, 6, 25, 3});
    auto C = ngraph::op::v0::Constant::create(ngraph::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto D = ngraph::op::v0::Constant::create(ngraph::element::i32, shape, {0, 3, 1, 2, 5, 6, 25, 3});
    auto E = ngraph::op::v0::Constant::create(ngraph::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto F = ngraph::op::v0::Constant::create(ngraph::element::i32, shape, {0, 3, 1, 2, 5, 6, 25, 3});

    auto ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B, C, D, E, F}, ngraph::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ngraph::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializationConstantCompressionTest, NonIdenticalConstants) {
    constexpr int unique_const_count = 2;
    const ngraph::Shape shape{2, 2, 2};

    auto A = ngraph::op::v0::Constant::create(ngraph::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ngraph::op::v0::Constant::create(ngraph::element::i32, shape, {2, 2, 3, 4, 5, 6, 7, 8});

    auto ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B}, ngraph::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ngraph::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsDifferentTypesI32I64) {
    constexpr int unique_const_count = 1;
    const ngraph::Shape shape{2, 2, 2};

    auto A = ngraph::op::v0::Constant::create(ngraph::element::i32, shape, {1, 0, 2, 0, 3, 0, 4, 0});
    auto B = ngraph::op::v0::Constant::create(ngraph::element::i64, ngraph::Shape({1, 2, 2}), {1, 2, 3, 4});

    auto ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B}, ngraph::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ngraph::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsDifferentTypesI32I8) {
    constexpr int unique_const_count = 1;
    const ngraph::Shape shape{1, 1, 2};

    auto A = ngraph::op::v0::Constant::create(ngraph::element::i32, shape, {1, 2});
    auto B = ngraph::op::v0::Constant::create(ngraph::element::i8, ngraph::Shape({1, 2, 4}), {1, 0, 0, 0, 2, 0, 0, 0});

    auto ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B}, ngraph::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ngraph::shape_size(shape) * sizeof(int32_t));
}
