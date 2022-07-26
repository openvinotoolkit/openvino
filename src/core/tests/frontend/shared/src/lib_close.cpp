// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lib_close.hpp"

#include "openvino/frontend/manager.hpp"
#include "openvino/util/file_util.hpp"

using namespace testing;
using namespace ov::util;
using namespace ov::frontend;

void FrontendLibCloseTest::SetUp() {
    std::tie(frontend, model_path, exp_name) = GetParam();
}

std::string FrontendLibCloseTest::get_test_case_name(const TestParamInfo<FrontendLibCloseParams>& obj) {
    return std::get<0>(obj.param);
}

/**
 * \brief Delete InputModel object as last.
 *
 * Frontend library must close after object deletion, otherwise segfault can occur.
 */
TEST_P(FrontendLibCloseTest, testModelIsLasDeletedObject) {
    InputModel::Ptr model;
    {
        auto fem = std::make_shared<ov::frontend::FrontEndManager>();
        auto fe = fem->load_by_framework(frontend);
        model = fe->load(model_path);
    }
    ASSERT_NE(model, nullptr);
}

/** \brief Frontend library must close after object deletion, otherwise segfault can occur. */
TEST_P(FrontendLibCloseTest, testPlaceIsLastDeletedObject) {
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

/** \brief Delete place which is created from other place instance. */
TEST_P(FrontendLibCloseTest, testPlaceFromPlaceIsLastDeletedObject) {
    Place::Ptr port_place;
    {
        auto fem = std::make_shared<ov::frontend::FrontEndManager>();
        auto fe = fem->load_by_framework(frontend);
        auto model = fe->load(model_path);
        auto tensor_place = model->get_place_by_tensor_name(exp_name);
        port_place = tensor_place->get_producing_port();
    }
    ASSERT_NE(port_place, nullptr);
    ASSERT_EQ(port_place->get_producing_port(), nullptr);
}

/** \brief Delete vector of places object as last one. */
TEST_P(FrontendLibCloseTest, testGetVectorOfPlaces) {
    std::vector<Place::Ptr> inputs;
    {
        auto fem = std::make_shared<ov::frontend::FrontEndManager>();
        auto fe = fem->load_by_framework(frontend);
        auto model = fe->load(model_path);
        inputs = model->get_inputs();
    }

    ASSERT_FALSE(inputs.empty());
}
