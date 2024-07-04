// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/manager.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "paddle_utils.hpp"
#include "utils.hpp"

using namespace ov::frontend;

TEST(FrontEndConvertModelTest, throw_in_conversion) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    OV_ASSERT_NO_THROW(frontEnd = fem.load_by_framework(PADDLE_FE));
    ASSERT_NE(frontEnd, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(
        std::string(TEST_PADDLE_MODELS_DIRNAME) + std::string("throw_in_conversion/throw_in_conversion.pdmodel"));
    OV_ASSERT_NO_THROW(inputModel = frontEnd->load(model_filename));
    ASSERT_NE(inputModel, nullptr);
    std::shared_ptr<ov::Model> model;
    ASSERT_THROW(model = frontEnd->convert(inputModel), OpConversionFailure);
}

TEST(FrontEndConvertModelTest, unsupported_version) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    OV_ASSERT_NO_THROW(frontEnd = fem.load_by_framework(PADDLE_FE));
    ASSERT_NE(frontEnd, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(std::string(TEST_PADDLE_MODELS_DIRNAME) +
                                                             std::string("lower_version/lower_version.pdmodel"));

    ASSERT_THROW(inputModel = frontEnd->load(model_filename), GeneralFailure);
}
