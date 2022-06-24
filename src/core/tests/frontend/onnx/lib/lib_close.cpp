// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <fstream>
#include <ngraph/file_util.hpp>
#include <openvino/frontend/manager.hpp>

#include "utils.hpp"

using namespace ngraph;
using namespace testing;
using namespace ov::frontend;

class LibCloseTest : public Test {
protected:
    const std::string frontend{"onnx"};
    const std::string exp_name{"Y"};  //!< Tensor name in model used for tests.
    std::string model_path;           //!< Model filename path.

    void SetUp() override {
        model_path = file_util::path_join(std::string(TEST_ONNX_MODELS_DIRNAME),
                                          std::string("external_data/external_data.onnx"));
    }
};

/**
 * \brief Delete InputModel object as last.
 *
 * Frontend library must close after object deletion, otherwise segfault can occur.
 */
TEST_F(LibCloseTest, testModelIsLasDeletedObject) {
    InputModel::Ptr model;
    {
        auto fem = std::make_shared<ov::frontend::FrontEndManager>();
        auto fe = fem->load_by_framework(frontend);
        model = fe->load(model_path);
    }
    ASSERT_NE(model, nullptr);
}

/**
 * \brief Delete Place object as last.
 *
 * Frontend library must close after object deletion, otherwise segfault can occur.
 */
TEST_F(LibCloseTest, testPlaceIsLastDeletedObject) {
    Place::Ptr place;
    {
        auto fem = std::make_shared<ov::frontend::FrontEndManager>();
        auto fe = fem->load_by_framework(frontend);
        auto model = fe->load(model_path);
        place = model->get_place_by_tensor_name(exp_name);
    }
    ASSERT_NE(place, nullptr);
    EXPECT_EQ(place->get_names().at(0), exp_name);
}
