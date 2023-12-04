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
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "tensor_conversion_util.hpp"

using namespace std;
using namespace ov;

TEST(tensor, tensor_names) {
    auto arg0 = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = make_shared<ov::op::v0::Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f0 = make_shared<Model>(relu, ParameterVector{arg0});

    ASSERT_EQ(arg0->get_output_tensor(0).get_names(), relu->get_input_tensor(0).get_names());
    ASSERT_EQ(arg0->get_output_tensor(0).get_names(), relu->input_value(0).get_tensor().get_names());
    ASSERT_EQ(f0->get_result()->get_input_tensor(0).get_names(), relu->get_output_tensor(0).get_names());
    ASSERT_EQ(f0->get_result()->input_value(0).get_tensor().get_names(), relu->get_output_tensor(0).get_names());
}

TEST(tensor, wrap_tensor_with_unspecified_type) {
    auto param = std::make_shared<ov::op::v0::Parameter>(element::undefined, ov::PartialShape{});
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto tensor = ov::util::wrap_tensor(param->output(0));
    OPENVINO_SUPPRESS_DEPRECATED_END
    // !tensor means that the tensor is not initialized
    EXPECT_EQ(!tensor, true);
}

TEST(tensor, wrap_tensor_with_unspecified_type_from_host_tensor) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto host_tensor = std::make_shared<ngraph::runtime::HostTensor>(element::undefined, ov::PartialShape{});
    auto tensor = ov::util::wrap_tensor(host_tensor);
    OPENVINO_SUPPRESS_DEPRECATED_END
    // !tensor means that the tensor is not initialized
    EXPECT_EQ(!tensor, true);
}

TEST(tensor, create_tensor_with_zero_dims_check_stride) {
    ov::Shape shape = {0, 0, 0, 0};
    auto tensor = ov::Tensor(element::f32, shape);
    EXPECT_EQ(!!tensor, true);
    auto stride = tensor.get_strides();
    EXPECT_EQ(stride.size(), shape.size());
    EXPECT_EQ(stride.back(), 0);
    EXPECT_EQ(tensor.is_continuous(), true);
}
