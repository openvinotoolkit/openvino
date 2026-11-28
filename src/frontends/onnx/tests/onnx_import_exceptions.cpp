// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <exception>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "core/tensor.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

TEST(onnx_importer, exception_throws_Exception) {
    EXPECT_THROW(convert_model("depth_to_space_bad_blocksize.onnx"), Exception);
}

TEST(onnx_importer, exception_msg_Exception) {
    try {
        convert_model("depth_to_space_bad_blocksize.onnx");
        // Should have thrown, so fail if it didn't
        FAIL() << "ONNX Importer did not detected incorrect model!";
    } catch (const Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("must be a multiple of divisor"));
    } catch (...) {
        FAIL() << "The ONNX model importer failed for unexpected reason";
    }
}

TEST(onnx_importer, exception_msg_onnx_node_validation_failure) {
    try {
        convert_model("instance_norm_bad_scale_type.onnx");
        // Should have thrown, so fail if it didn't
        FAIL() << "ONNX Importer did not detected incorrect model!";
    } catch (const ::ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("While validating ONNX node '<Node(InstanceNormalization)"));
    }
    // On MacOS after we re-throw ov::Exception exception, we couldn't catch it as is,
    // thus below workaround.
    catch (const std::exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("While validating ONNX node '<Node(InstanceNormalization)"));
    } catch (...) {
        FAIL() << "The ONNX model importer failed for unexpected reason";
    }
}

// This test aims to check for wrapping all std::exception not deriving from Exception.
// This test should throw a std error because of attempt to access shape from dynamic tensor.
TEST(onnx_importer, exception_msg_std_err_wrapped) {
    try {
        convert_model("eye_like_wrong_shape.onnx");
        // Should have thrown, so fail if it didn't
        FAIL() << "ONNX Importer did not detected incorrect model!";
    } catch (const std::exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("While validating ONNX node '<Node(EyeLike): y"));
    } catch (...) {
        FAIL() << "The ONNX model importer failed for unexpected reason";
    }
}

TEST(onnx_importer, exception_msg_onnx_reduce_wrong_type_v1) {
    try {
        convert_model("reduce_wrong_type_v1.onnx");
        // Should have thrown, so fail if it didn't
        FAIL() << "ONNX Importer did not detected incorrect model!";
    } catch (const ::ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Unsupported input type bf16"));
    }
    // On MacOS after we re-throw ov::Exception exception, we couldn't catch it as is,
    // thus below workaround.
    catch (const std::exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Unsupported input type bf16"));
    } catch (...) {
        FAIL() << "The ONNX model importer failed for unexpected reason";
    }
}

TEST(onnx_importer, exception_msg_onnx_reduce_wrong_type_v2) {
    try {
        convert_model("reduce_wrong_type_v2.onnx");
        // Should have thrown, so fail if it didn't
        FAIL() << "ONNX Importer did not detected incorrect model!";
    } catch (const ::ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Unsupported input type i8"));
    }
    // On MacOS after we re-throw ov::Exception exception, we couldn't catch it as is,
    // thus below workaround.
    catch (const std::exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Unsupported input type i8"));
    } catch (...) {
        FAIL() << "The ONNX model importer failed for unexpected reason";
    }
}

TEST(onnx_importer, exception_msg_onnx_reduce_wrong_type_v3) {
    try {
        convert_model("reduce_wrong_type_v3.onnx");
        // Should have thrown, so fail if it didn't
        FAIL() << "ONNX Importer did not detected incorrect model!";
    } catch (const ::ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Unsupported input type boolean"));
    }
    // On MacOS after we re-throw ov::Exception exception, we couldn't catch it as is,
    // thus below workaround.
    catch (const std::exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("Unsupported input type boolean"));
    } catch (...) {
        FAIL() << "The ONNX model importer failed for unexpected reason";
    }
}

TEST(onnx_importer, no_exception_onnx_reduce_wrong_type_v4) {
    EXPECT_NO_THROW(convert_model("reduce_wrong_type_v4.onnx"));
}

TEST(onnx_importer, exception_msg_softmax_axis_out_of_range) {
    try {
        convert_model("softmax_invalid_axis_opset11.onnx");
        // Should have thrown, so fail if it didn't
        FAIL() << "ONNX Importer did not detected incorrect model!";
    } catch (const ::ov::Exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("out of tensor range"));
    }
    // On MacOS after we re-throw ov::Exception exception, we couldn't catch it as is,
    // thus below workaround.
    catch (const std::exception& e) {
        EXPECT_HAS_SUBSTRING(e.what(), std::string("out of tensor range"));
    } catch (...) {
        FAIL() << "The ONNX model importer failed for unexpected reason";
    }
}

TEST(onnx_importer, exception_msg_sparse_constant_int32_indices) {
    OV_EXPECT_THROW_HAS_SUBSTRING(convert_model("constant_sparse_tensor_int32_indices.onnx"),
                                  ov::Exception,
                                  "Sparse tensor indices must be of type INT64");
}

TEST(onnx_importer, exception_msg_sparse_constant_uint8_raw_indices) {
    OV_EXPECT_THROW_HAS_SUBSTRING(convert_model("constant_sparse_tensor_uint8_raw_indices.onnx"),
                                  ov::Exception,
                                  "Sparse tensor indices must be of type INT64");
}

TEST(onnx_importer, raw_data_is_read_with_the_declared_element_size) {
    const std::string raw_data(16, static_cast<char>(1));
    OV_EXPECT_THROW_HAS_SUBSTRING(
        ov::frontend::onnx::detail::__get_raw_data<int64_t>(raw_data, ONNX_NAMESPACE::TensorProto_DataType_UINT8),
        ov::Exception,
        "Raw data of type UINT8 cannot be read as 8-byte elements");

    const auto values =
        ov::frontend::onnx::detail::__get_raw_data<int32_t>(raw_data, ONNX_NAMESPACE::TensorProto_DataType_INT32);
    EXPECT_EQ(values, std::vector<int32_t>(4, 0x01010101));
}
