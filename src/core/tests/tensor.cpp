// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/test_tools.hpp"
#include "gtest/gtest.h"
#include "ngraph/function.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "ngraph/pass/manager.hpp"
#include "tensor_conversion_util.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

TEST(tensor, tensor_names) {
    auto arg0 = make_shared<opset6::Parameter>(element::f32, Shape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = make_shared<opset6::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f0 = make_shared<Function>(relu, ParameterVector{arg0});

    ASSERT_EQ(arg0->get_output_tensor(0).get_names(), relu->get_input_tensor(0).get_names());
    ASSERT_EQ(arg0->get_output_tensor(0).get_names(), relu->input_value(0).get_tensor().get_names());
    ASSERT_EQ(f0->get_result()->get_input_tensor(0).get_names(), relu->get_output_tensor(0).get_names());
    ASSERT_EQ(f0->get_result()->input_value(0).get_tensor().get_names(), relu->get_output_tensor(0).get_names());
}

TEST(tensor, wrap_tensor_with_unspecified_type) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element::undefined, ov::PartialShape{});
    auto tensor = ov::util::wrap_tensor(param->output(0));
    // !tensor means that the tensor is not initialized
    EXPECT_EQ(!tensor, true);
}

TEST(tensor, wrap_tensor_with_unspecified_type_from_host_tensor) {
    auto host_tensor = std::make_shared<ngraph::HostTensor>(element::undefined, ov::PartialShape{});
    auto tensor = ov::util::wrap_tensor(host_tensor);
    // !tensor means that the tensor is not initialized
    EXPECT_EQ(!tensor, true);
}
