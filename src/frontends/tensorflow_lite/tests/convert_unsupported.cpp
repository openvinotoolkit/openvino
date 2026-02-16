// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "convert_model.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend;

TEST(FrontEndConvertModelTest, test_zerolen) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    OV_ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_LITE_FE));
    ASSERT_NE(frontEnd, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME) +
                                                             string("bad_header/zerolen.tflite"));
    OV_ASSERT_NO_THROW(inputModel = frontEnd->load(model_filename));
    ASSERT_NE(inputModel, nullptr);
    shared_ptr<ov::Model> model;
    ASSERT_THROW(model = frontEnd->convert(inputModel), std::exception);
}

TEST(FrontEndConvertModelTest, test_wrong_len) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    OV_ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_LITE_FE));
    ASSERT_NE(frontEnd, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME) +
                                                             string("bad_header/wrong_len_3.tflite"));
    OV_ASSERT_NO_THROW(inputModel = frontEnd->load(model_filename));
    ASSERT_NE(inputModel, nullptr);
    shared_ptr<ov::Model> model;
    ASSERT_THROW(model = frontEnd->convert(inputModel), std::exception);
}

TEST(FrontEndConvertModelTest, test_wrong_pos) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    OV_ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_LITE_FE));
    ASSERT_NE(frontEnd, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME) +
                                                             string("bad_header/wrong_pos.tflite"));
    OV_ASSERT_NO_THROW(inputModel = frontEnd->load(model_filename));
    ASSERT_NE(inputModel, nullptr);
    shared_ptr<ov::Model> model;
    ASSERT_THROW(model = frontEnd->convert(inputModel), std::exception);
}

// Sparse tensor with out-of-bounds index value
// Index 999 in a dimension of size 100 would cause heap OOB write without bounds checking
TEST(FrontEndConvertModelTest, test_sparse_oob_index) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    OV_ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_LITE_FE));
    ASSERT_NE(frontEnd, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME) +
                                                             string("sparse_oob/sparse_oob_index.tflite"));
    ASSERT_THROW(
        {
            inputModel = frontEnd->load(model_filename);
            if (inputModel) {
                frontEnd->convert(inputModel);
            }
        },
        std::exception);
}

// Sparse tensor with negative index value
TEST(FrontEndConvertModelTest, test_sparse_negative_index) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    OV_ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_LITE_FE));
    ASSERT_NE(frontEnd, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME) +
                                                             string("sparse_oob/sparse_negative_index.tflite"));
    ASSERT_THROW(
        {
            inputModel = frontEnd->load(model_filename);
            if (inputModel) {
                frontEnd->convert(inputModel);
            }
        },
        std::exception);
}

// Sparse tensor with non-monotonic segments
TEST(FrontEndConvertModelTest, test_sparse_non_monotonic_segments) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    OV_ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_LITE_FE));
    ASSERT_NE(frontEnd, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME) +
                                                             string("sparse_oob/sparse_non_monotonic_segments.tflite"));
    ASSERT_THROW(
        {
            inputModel = frontEnd->load(model_filename);
            if (inputModel) {
                frontEnd->convert(inputModel);
            }
        },
        std::exception);
}

// Sparse tensor with shape dimensions that cause integer overflow in size computation
TEST(FrontEndConvertModelTest, test_sparse_overflow_shape) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    OV_ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_LITE_FE));
    ASSERT_NE(frontEnd, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_LITE_MODELS_DIRNAME) +
                                                             string("sparse_oob/sparse_overflow_shape.tflite"));
    ASSERT_THROW(
        {
            inputModel = frontEnd->load(model_filename);
            if (inputModel) {
                frontEnd->convert(inputModel);
            }
        },
        std::exception);
}
