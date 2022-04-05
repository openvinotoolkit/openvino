// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/exception.hpp>
#include <openvino/frontend/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "tf_utils.hpp"
#include "utils.hpp"

using namespace ngraph;
using namespace ov::frontend;

TEST(FrontEndConvertModelTest, test_unsupported_op) {
    FrontEndManager fem;
    FrontEnd::Ptr frontEnd;
    InputModel::Ptr inputModel;
    ASSERT_NO_THROW(frontEnd = fem.load_by_framework(TF_FE));
    ASSERT_NE(frontEnd, nullptr);
    auto model_filename = FrontEndTestUtils::make_model_path(std::string(TEST_TENSORFLOW_MODELS_DIRNAME) +
                                                             std::string("relu_unsupported/relu_unsupported.pb"));
    ASSERT_NO_THROW(inputModel = frontEnd->load(model_filename));
    ASSERT_NE(inputModel, nullptr);
    std::shared_ptr<ngraph::Function> function;
    ASSERT_THROW(function = frontEnd->convert(inputModel), OpConversionFailure);
    ASSERT_EQ(function, nullptr);
    ASSERT_NO_THROW(function = frontEnd->decode(inputModel));
    ASSERT_THROW(frontEnd->convert(function), OpConversionFailure);
    ASSERT_NO_THROW(function = frontEnd->convert_partially(inputModel));
    ASSERT_THROW(frontEnd->convert(function), OpConversionFailure);

    for (auto& node : function->get_ordered_ops()) {
        if (node->get_friendly_name() == "relu_0") {
            function->replace_node(node, std::make_shared<opset6::Relu>(node->input(0).get_source_output()));
        }
    }
    ASSERT_NO_THROW(frontEnd->convert(function));
}
