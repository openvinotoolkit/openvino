// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>

#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/serialize.hpp"
#include "util/test_common.hpp"

class SerializatioConstantCompressionTest : public ov::test::TestsCommon {
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

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsI32) {
    constexpr int unique_const_count = 1;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});

    auto ngraph_a = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ov::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsI64) {
    constexpr int unique_const_count = 1;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::i64, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::i64, shape, {1, 2, 3, 4, 5, 6, 7, 8});

    auto ngraph_a = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ov::shape_size(shape) * sizeof(int64_t));
}

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsFP16) {
    constexpr int unique_const_count = 1;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::f16, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::f16, shape, {1, 2, 3, 4, 5, 6, 7, 8});

    auto ngraph_a = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ov::shape_size(shape) * sizeof(ov::float16));
}

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsFP32) {
    constexpr int unique_const_count = 1;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});

    auto ngraph_a = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ov::shape_size(shape) * sizeof(float));
}

TEST_F(SerializatioConstantCompressionTest, NonIdenticalConstantsI64) {
    constexpr int unique_const_count = 2;
    const ov::Shape shape{2};

    // hash_combine returns the same hash for this two constants so we also check the content of arrays
    auto A = ov::opset8::Constant::create(ov::element::i64, shape, {2, 2});
    auto B = ov::opset8::Constant::create(ov::element::i64, shape, {0, 128});

    auto ngraph_a = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ov::shape_size(shape) * sizeof(int64_t));
}

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsTimesTwo) {
    constexpr int unique_const_count = 2;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto C = ov::opset8::Constant::create(ov::element::i32, shape, {0, 3, 1, 2, 5, 6, 25, 3});
    auto D = ov::opset8::Constant::create(ov::element::i32, shape, {0, 3, 1, 2, 5, 6, 25, 3});

    auto ngraph_a = std::make_shared<ov::Model>(ov::NodeVector{A, B, C, D}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ov::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsTimesTwoMultipleOccurences) {
    constexpr int unique_const_count = 2;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::i32, shape, {0, 3, 1, 2, 5, 6, 25, 3});
    auto C = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto D = ov::opset8::Constant::create(ov::element::i32, shape, {0, 3, 1, 2, 5, 6, 25, 3});
    auto E = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto F = ov::opset8::Constant::create(ov::element::i32, shape, {0, 3, 1, 2, 5, 6, 25, 3});

    auto ngraph_a = std::make_shared<ov::Model>(ov::NodeVector{A, B, C, D, E, F}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ov::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializatioConstantCompressionTest, NonIdenticalConstants) {
    constexpr int unique_const_count = 2;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::i32, shape, {2, 2, 3, 4, 5, 6, 7, 8});

    auto ngraph_a = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ov::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsDifferentTypesI32I64) {
    constexpr int unique_const_count = 1;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::i32, shape, {1, 0, 2, 0, 3, 0, 4, 0});
    auto B = ov::opset8::Constant::create(ov::element::i64, ov::Shape({1, 2, 2}), {1, 2, 3, 4});

    auto ngraph_a = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ov::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsDifferentTypesI32I8) {
    constexpr int unique_const_count = 1;
    const ov::Shape shape{1, 1, 2};

    auto A = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2});
    auto B = ov::opset8::Constant::create(ov::element::i8, ov::Shape({1, 2, 4}), {1, 0, 0, 0, 2, 0, 0, 0});

    auto ngraph_a = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_TRUE(file_size(bin_1) == unique_const_count * ov::shape_size(shape) * sizeof(int32_t));
}
