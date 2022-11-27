// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/manager.hpp>

#include "test_common.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::frontend;

TEST(FrontEndConvertModelTest, test_undefined_input_shape) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_FE));
    ASSERT_NE(frontEnd, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(string(TEST_TENSORFLOW_MODELS_DIRNAME) +
                                                             string("undefined_input_shape/undefined_input_shape.pb"));
    ASSERT_NO_THROW(inputModel = frontEnd->load(model_filename));
    ASSERT_NE(inputModel, nullptr);
    shared_ptr<Model> model;
    ASSERT_NO_THROW(model = frontEnd->convert(inputModel));
    ASSERT_NE(model, nullptr);

    for (auto& node : model->get_ordered_ops()) {
        if (node->get_friendly_name() == "x") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
        } else if (node->get_friendly_name() == "y") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(PartialShape{2, 3}));
        } else if (node->get_friendly_name() == "z") {
            ASSERT_TRUE(node->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
        }
    }
}
