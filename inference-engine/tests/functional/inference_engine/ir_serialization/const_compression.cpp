// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ie_core.hpp"
#include "gtest/gtest.h"

#include <ngraph/function.hpp>
#include <transformations/serialize.hpp>

#ifndef IR_SERIALIZATION_MODELS_PATH // should be already defined by cmake
#define IR_SERIALIZATION_MODELS_PATH ""
#endif

class SerializatioConstantCompressionTest : public ::testing::Test {
protected:
    std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::string m_out_xml_path_1 = test_name + "1" + ".xml";
    std::string m_out_bin_path_1 = test_name + "1" + ".bin";

    void TearDown() override {
        std::remove(m_out_xml_path_1.c_str());
        std::remove(m_out_bin_path_1.c_str());
    }

    std::uintmax_t file_size(std::ifstream &f) {
        // get length of file:
        f.seekg(0, f.end);
        std::uintmax_t length = f.tellg();
        f.seekg(0, f.beg);
        return length;
    }
};

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsI32) {
    std::shared_ptr<ngraph::Function> ngraph_a;
    const int unique_const_count = 1;
    ngraph::Shape shape{2, 2, 2};

    auto A = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {1, 2, 3, 4, 5, 6, 7, 8});

    ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B},
        ngraph::ParameterVector{});

    ngraph::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);

    int shape_element_count = 1;
    std::for_each(shape.begin(), shape.end(), [&](const int &n) {shape_element_count *= n;});
    ASSERT_TRUE(file_size(bin_1) == unique_const_count * shape_element_count * sizeof(int));
}

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsI64) {
    std::shared_ptr<ngraph::Function> ngraph_a;
    const int unique_const_count = 1;
    ngraph::Shape shape{2, 2, 2};

    auto A = ngraph::op::Constant::create(ngraph::element::i64, shape,
        {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ngraph::op::Constant::create(ngraph::element::i64, shape,
        {1, 2, 3, 4, 5, 6, 7, 8});

    ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B},
        ngraph::ParameterVector{});

    ngraph::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);

    int shape_element_count = 1;
    std::for_each(shape.begin(), shape.end(), [&](const int &n) {shape_element_count *= n;});
    ASSERT_TRUE(file_size(bin_1) == unique_const_count * shape_element_count * sizeof(int64_t));
}

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsFP16) {
    std::shared_ptr<ngraph::Function> ngraph_a;
    const int unique_const_count = 1;
    ngraph::Shape shape{2, 2, 2};

    auto A = ngraph::op::Constant::create(ngraph::element::f16, shape,
        {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ngraph::op::Constant::create(ngraph::element::f16, shape,
        {1, 2, 3, 4, 5, 6, 7, 8});

    ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B},
        ngraph::ParameterVector{});

    ngraph::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);

    int shape_element_count = 1;
    std::for_each(shape.begin(), shape.end(), [&](const int &n) {shape_element_count *= n;});
    ASSERT_TRUE(file_size(bin_1) == unique_const_count * shape_element_count * sizeof(ngraph::float16));
}

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsFP32) {
    std::shared_ptr<ngraph::Function> ngraph_a;
    const int unique_const_count = 1;
    ngraph::Shape shape{2, 2, 2};

    auto A = ngraph::op::Constant::create(ngraph::element::f32, shape,
        {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ngraph::op::Constant::create(ngraph::element::f32, shape,
        {1, 2, 3, 4, 5, 6, 7, 8});

    ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B},
        ngraph::ParameterVector{});

    ngraph::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);

    int shape_element_count = 1;
    std::for_each(shape.begin(), shape.end(), [&](const int &n) {shape_element_count *= n;});
    ASSERT_TRUE(file_size(bin_1) == unique_const_count * shape_element_count * sizeof(float));
}

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsTimesTwo) {
    std::shared_ptr<ngraph::Function> ngraph_a;
    const int unique_const_count = 2;
    ngraph::Shape shape{2, 2, 2};

    auto A = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {1, 2, 3, 4, 5, 6, 7, 8});
    auto C = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {0, 3, 1, 2, 5, 6, 25, 3});
    auto D = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {0, 3, 1, 2, 5, 6, 25, 3});

    ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B, C, D},
        ngraph::ParameterVector{});

    ngraph::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);

    int shape_element_count = 1;
    std::for_each(shape.begin(), shape.end(), [&](const int &n) {shape_element_count *= n;});
    ASSERT_TRUE(file_size(bin_1) == unique_const_count * shape_element_count * sizeof(int));
}

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsTimesTwoMultipleOccurences) {
    std::shared_ptr<ngraph::Function> ngraph_a;
    const int unique_const_count = 2;
    ngraph::Shape shape{2, 2, 2};

    auto A = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {0, 3, 1, 2, 5, 6, 25, 3});
    auto C = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {1, 2, 3, 4, 5, 6, 7, 8});
    auto D = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {0, 3, 1, 2, 5, 6, 25, 3});
    auto E = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {1, 2, 3, 4, 5, 6, 7, 8});
    auto F = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {0, 3, 1, 2, 5, 6, 25, 3});

    ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B, C, D, E, F},
        ngraph::ParameterVector{});

    ngraph::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);

    int shape_element_count = 1;
    std::for_each(shape.begin(), shape.end(), [&](const int &n) {shape_element_count *= n;});
    ASSERT_TRUE(file_size(bin_1) == unique_const_count * shape_element_count * sizeof(int));
}

TEST_F(SerializatioConstantCompressionTest, NonIdenticalConstants) {
    std::shared_ptr<ngraph::Function> ngraph_a;
    const int unique_const_count = 2;
    ngraph::Shape shape{2, 2, 2};

    auto A = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {2, 2, 3, 4, 5, 6, 7, 8});

    ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B},
        ngraph::ParameterVector{});

    ngraph::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);

    int shape_element_count = 1;
    std::for_each(shape.begin(), shape.end(), [&](const int &n) {shape_element_count *= n;});
    ASSERT_TRUE(file_size(bin_1) == unique_const_count * shape_element_count * sizeof(int));
}

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsDifferentTypesI32I64) {
    std::shared_ptr<ngraph::Function> ngraph_a;
    const int unique_const_count = 1;
    ngraph::Shape shape{2, 2, 2};

    auto A = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {1, 0, 2, 0, 3, 0, 4, 0});
    auto B = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape({1, 2, 2}),
        {1, 2, 3, 4});

    ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B},
        ngraph::ParameterVector{});

    ngraph::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);

    int shape_element_count = 1;
    std::for_each(shape.begin(), shape.end(), [&](const int &n) {shape_element_count *= n;});
    ASSERT_TRUE(file_size(bin_1) == unique_const_count * shape_element_count * sizeof(int));
}

TEST_F(SerializatioConstantCompressionTest, IdenticalConstantsDifferentTypesI32I8) {
    std::shared_ptr<ngraph::Function> ngraph_a;
    const int unique_const_count = 1;

    ngraph::Shape shape{2, 2, 2};
    auto A = ngraph::op::Constant::create(ngraph::element::i32, shape,
        {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ngraph::op::Constant::create(ngraph::element::i8, ngraph::Shape({2, 4, 4}),
        {1, 0, 0, 0,
         2, 0, 0, 0,
         3, 0, 0, 0,
         4, 0, 0, 0,
         5, 0, 0, 0,
         6, 0, 0, 0,
         7, 0, 0, 0,
         8, 0, 0, 0});

    ngraph_a = std::make_shared<ngraph::Function>(ngraph::NodeVector{A, B},
        ngraph::ParameterVector{});

    ngraph::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_function(ngraph_a);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);

    int shape_element_count = 1;
    std::for_each(shape.begin(), shape.end(), [&](const int &n) {shape_element_count *= n;});
    ASSERT_TRUE(file_size(bin_1) == unique_const_count * shape_element_count * sizeof(int));
}
