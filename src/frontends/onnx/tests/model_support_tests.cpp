// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>

#include "common_test_utils/file_utils.hpp"
#include "core/graph_iterator_proto.hpp"
#include "onnx_common/onnx_model_validator.hpp"
#include "openvino/runtime/core.hpp"

namespace {
std::string model_path(const char* model) {
    std::string path = TEST_ONNX_MODELS_DIRNAME;
    path += "support_test/";
    path += model;
    return ov::test::utils::getModelFromTestModelZoo(path);
}
}  // namespace

TEST(ONNXModelValidator, configuration_field_before_graph_common) {
    // Field 26's length-delimited tag is the two-byte varint d2 01.
    std::istringstream model(std::string("\xd2\x01\x00\x08\x09\x3a\x00", 7), std::ios::binary);
    EXPECT_TRUE(ov::frontend::onnx::common::is_valid_model(model));
}

TEST(ONNXModelValidator, configuration_field_before_graph_iterator) {
    std::istringstream model(std::string("\xd2\x01\x00\x08\x09\x3a\x00", 7), std::ios::binary);
    EXPECT_TRUE(ov::frontend::onnx::is_valid_model(model));
}

TEST(ONNXModelValidator, truncated_multibyte_tag) {
    std::istringstream common_model(std::string("\x08\x09\xd2", 3), std::ios::binary);
    std::istringstream iterator_model(std::string("\x08\x09\xd2", 3), std::ios::binary);
    EXPECT_FALSE(ov::frontend::onnx::common::is_valid_model(common_model));
    EXPECT_FALSE(ov::frontend::onnx::is_valid_model(iterator_model));
}

TEST(ONNXReader_ModelSupported, basic_model) {
    // this model is a basic ONNX model taken from OpenVINO's unit test (add_abc.onnx)
    // it contains the minimum number of fields required to accept this file as a valid model
    EXPECT_NO_THROW(ov::Core{}.read_model(model_path("supported/basic.onnx")));
}

TEST(ONNXReader_ModelSupported, basic_reverse_fields_order) {
    // this model contains the same fields as basic.onnx but serialized in reverse order
    EXPECT_NO_THROW(ov::Core{}.read_model(model_path("supported/basic_reverse_fields_order.onnx")));
}

TEST(ONNXReader_ModelSupported, more_fields) {
    // this model contains some optional fields (producer_name and doc_string) but 5 fields in total
    EXPECT_NO_THROW(ov::Core{}.read_model(model_path("supported/more_fields.onnx")));
}

TEST(ONNXReader_ModelSupported, varint_on_two_bytes) {
    // the docstring's payload length is encoded as varint using 2 bytes which should be parsed correctly
    EXPECT_NO_THROW(ov::Core{}.read_model(model_path("supported/varint_on_two_bytes.onnx")));
}

TEST(ONNXReader_ModelSupported, scrambled_keys) {
    // same as the prototxt_basic but with a different order of keys
    EXPECT_NO_THROW(ov::Core{}.read_model(model_path("supported/scrambled_keys.onnx")));
}

TEST(ONNXReader_ModelUnsupported, no_graph_field) {
    // this model contains only 2 fields (it doesn't contain a graph in particular)
    EXPECT_THROW(ov::Core{}.read_model(model_path("unsupported/no_graph_field.onnx")), ov::Exception);
}

TEST(ONNXReader_ModelUnsupported, incorrect_onnx_field) {
    // in this model the second field's key is F8 (field number 31) which is doesn't exist in ONNX
    // this  test will have to be changed if the number of fields in onnx.proto
    // (ModelProto message definition) ever reaches 31 or more
    EXPECT_THROW(ov::Core{}.read_model(model_path("unsupported/incorrect_onnx_field.onnx")), ov::Exception);
}

TEST(ONNXReader_ModelUnsupported, unknown_wire_type) {
    // in this model the graph key contains wire type 7 encoded in it - this value is incorrect
    EXPECT_THROW(ov::Core{}.read_model(model_path("unsupported/unknown_wire_type.onnx")), ov::Exception);
}

TEST(ONNXReader_ModelUnsupported, duplicate_fields) {
    // the model contains the IR_VERSION field twice - this is not correct
    EXPECT_THROW(ov::Core{}.read_model(model_path("unsupported/duplicate_onnx_fields.onnx")), ov::Exception);
}
