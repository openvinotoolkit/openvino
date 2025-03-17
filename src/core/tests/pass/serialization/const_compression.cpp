// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/core.hpp"
#include "transformations/common_optimizations/compress_float_constants.hpp"

class SerializationConstantCompressionTest : public ov::test::TestsCommon {
protected:
    std::string m_out_xml_path_1;
    std::string m_out_bin_path_1;

    void SetUp() override {
        std::string filePrefix = ov::test::utils::generateTestFilePrefix();
        m_out_xml_path_1 = filePrefix + ".xml";
        m_out_bin_path_1 = filePrefix + ".bin";
    }

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
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * ov::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsI64) {
    constexpr int unique_const_count = 1;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::i64, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::i64, shape, {1, 2, 3, 4, 5, 6, 7, 8});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * ov::shape_size(shape) * sizeof(int64_t));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsFP32_COMPRESSED_TO_F16) {
    constexpr int unique_const_count = 1;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});
    ov::pass::CompressFloatConstants(/*postponed=*/true).run_on_model(model);
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * ov::shape_size(shape) * sizeof(ov::float16));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsFP16) {
    constexpr int unique_const_count = 1;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::f16, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::f16, shape, {1, 2, 3, 4, 5, 6, 7, 8});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * ov::shape_size(shape) * sizeof(ov::float16));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsFP32) {
    constexpr int unique_const_count = 1;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * ov::shape_size(shape) * sizeof(float));
}

TEST_F(SerializationConstantCompressionTest, NonIdenticalConstantsI64) {
    constexpr int unique_const_count = 2;
    const ov::Shape shape{2};

    // hash_combine returns the same hash for this two constants so we also check the content of arrays
    auto A = ov::opset8::Constant::create(ov::element::i64, shape, {2, 2});
    auto B = ov::opset8::Constant::create(ov::element::i64, shape, {0, 128});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * ov::shape_size(shape) * sizeof(int64_t));
}

TEST_F(SerializationConstantCompressionTest, NonIdenticalConstantsI64_CHECK_MULTIMAP) {
    constexpr int unique_const_count = 2;
    const ov::Shape shape{2};

    // hash_combine returns the same hash for this two constants so we also check the content of arrays
    auto A = ov::opset8::Constant::create(ov::element::i64, shape, {2, 2});
    auto B = ov::opset8::Constant::create(ov::element::i64, shape, {0, 128});
    auto C = ov::opset8::Constant::create(ov::element::i64, shape, {2, 2});
    auto D = ov::opset8::Constant::create(ov::element::i64, shape, {0, 128});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{A, B, C, D}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * ov::shape_size(shape) * sizeof(int64_t));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsTimesTwo) {
    constexpr int unique_const_count = 2;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto C = ov::opset8::Constant::create(ov::element::i32, shape, {0, 3, 1, 2, 5, 6, 25, 3});
    auto D = ov::opset8::Constant::create(ov::element::i32, shape, {0, 3, 1, 2, 5, 6, 25, 3});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{A, B, C, D}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * ov::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsTimesTwo_FP32_COMPRESSED_TO_FP16) {
    constexpr int unique_const_count = 2;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto C = ov::opset8::Constant::create(ov::element::f32, shape, {0, 3, 1, 2, 5, 6, 25, 3});
    auto D = ov::opset8::Constant::create(ov::element::f32, shape, {0, 3, 1, 2, 5, 6, 25, 3});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{A, B, C, D}, ov::ParameterVector{});
    ov::pass::CompressFloatConstants(/*postponed=*/true).run_on_model(model);
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * ov::shape_size(shape) * sizeof(ov::float16));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsTimesTwoMultipleOccurrences) {
    constexpr int unique_const_count = 2;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::i32, shape, {0, 3, 1, 2, 5, 6, 25, 3});
    auto C = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto D = ov::opset8::Constant::create(ov::element::i32, shape, {0, 3, 1, 2, 5, 6, 25, 3});
    auto E = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto F = ov::opset8::Constant::create(ov::element::i32, shape, {0, 3, 1, 2, 5, 6, 25, 3});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{A, B, C, D, E, F}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * ov::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsTimesTwoMultipleOccurrences_FP32_COMPRESSED_TO_FP16) {
    constexpr int unique_const_count = 2;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::f32, shape, {0, 3, 1, 2, 5, 6, 25, 3});
    auto C = ov::opset8::Constant::create(ov::element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto D = ov::opset8::Constant::create(ov::element::f32, shape, {0, 3, 1, 2, 5, 6, 25, 3});
    auto E = ov::opset8::Constant::create(ov::element::f32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto F = ov::opset8::Constant::create(ov::element::f32, shape, {0, 3, 1, 2, 5, 6, 25, 3});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{A, B, C, D, E, F}, ov::ParameterVector{});
    ov::pass::CompressFloatConstants(/*postponed=*/true).run_on_model(model);
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * ov::shape_size(shape) * sizeof(ov::float16));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsTimesTwoMultipleOccurrences_FP64_COMPRESSED_TO_FP16) {
    constexpr int unique_const_count = 2;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::f64, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::f64, shape, {0, 3, 1, 2, 5, 6, 25, 3});
    auto C = ov::opset8::Constant::create(ov::element::f64, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto D = ov::opset8::Constant::create(ov::element::f64, shape, {0, 3, 1, 2, 5, 6, 25, 3});
    auto E = ov::opset8::Constant::create(ov::element::f64, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto F = ov::opset8::Constant::create(ov::element::f64, shape, {0, 3, 1, 2, 5, 6, 25, 3});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{A, B, C, D, E, F}, ov::ParameterVector{});
    ov::pass::CompressFloatConstants(/*postponed=*/true).run_on_model(model);
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * ov::shape_size(shape) * sizeof(ov::float16));
}

TEST_F(SerializationConstantCompressionTest, NonIdenticalConstants) {
    constexpr int unique_const_count = 2;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2, 3, 4, 5, 6, 7, 8});
    auto B = ov::opset8::Constant::create(ov::element::i32, shape, {2, 2, 3, 4, 5, 6, 7, 8});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * ov::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsDifferentTypesI32I64) {
    constexpr int unique_const_count = 1;
    const ov::Shape shape{2, 2, 2};

    auto A = ov::opset8::Constant::create(ov::element::i32, shape, {1, 0, 2, 0, 3, 0, 4, 0});
    auto B = ov::opset8::Constant::create(ov::element::i64, ov::Shape({1, 2, 2}), {1, 2, 3, 4});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * ov::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializationConstantCompressionTest, IdenticalConstantsDifferentTypesI32I8) {
    constexpr int unique_const_count = 1;
    const ov::Shape shape{1, 1, 2};

    auto A = ov::opset8::Constant::create(ov::element::i32, shape, {1, 2});
    auto B = ov::opset8::Constant::create(ov::element::i8, ov::Shape({1, 2, 4}), {1, 0, 0, 0, 2, 0, 0, 0});

    auto model = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * ov::shape_size(shape) * sizeof(int32_t));
}

TEST_F(SerializationConstantCompressionTest, EmptyConstants) {
    constexpr int unique_const_count = 1;
    auto A = ov::opset8::Constant::create(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{});
    auto B = ov::opset8::Constant::create(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{});

    auto model_initial = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model_initial);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * sizeof(int8_t));

    ov::Core core;
    auto model_imported = core.read_model(m_out_xml_path_1, m_out_bin_path_1);

    bool success;
    std::string message;
    std::tie(success, message) = compare_functions(model_initial, model_imported, true, true, false, true, true);
    ASSERT_TRUE(success) << message;
}

TEST_F(SerializationConstantCompressionTest, EmptyAndNotEmptyConstantSameValues) {
    constexpr int unique_const_count = 1;
    auto A = ov::opset8::Constant::create(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{});
    auto B = ov::opset8::Constant::create(ov::element::i8, ov::Shape{1}, std::vector<int8_t>{0});

    auto model_initial = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model_initial);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * sizeof(int8_t));

    ov::Core core;
    auto model_imported = core.read_model(m_out_xml_path_1, m_out_bin_path_1);

    bool success;
    std::string message;
    std::tie(success, message) = compare_functions(model_initial, model_imported, true, true, false, true, true);
    ASSERT_TRUE(success) << message;
}

TEST_F(SerializationConstantCompressionTest, EmptyAndNotEmptyConstantsDifferentValues) {
    constexpr int unique_const_count = 2;
    auto A = ov::opset8::Constant::create(ov::element::i32, ov::Shape{0}, std::vector<int32_t>{});
    auto B = ov::opset8::Constant::create(ov::element::i8, ov::Shape{1}, std::vector<int8_t>{1});

    auto model_initial = std::make_shared<ov::Model>(ov::NodeVector{A, B}, ov::ParameterVector{});

    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(model_initial);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::binary);

    ASSERT_EQ(file_size(bin_1), unique_const_count * sizeof(int8_t));

    ov::Core core;
    auto model_imported = core.read_model(m_out_xml_path_1, m_out_bin_path_1);

    bool success;
    std::string message;
    std::tie(success, message) = compare_functions(model_initial, model_imported, true, true, false, true, true);
    ASSERT_TRUE(success) << message;
}