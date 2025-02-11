// Copyright (C) 2018-2025 Intel Corporation
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
