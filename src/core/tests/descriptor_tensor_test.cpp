// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/test_tools.hpp"
#include "gmock/gmock.h"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"

namespace ov::test {
using testing::UnorderedElementsAre;

using op::v0::Parameter, op::v0::Relu, op::v0::Result;

using DescriptorTensorTest = ::testing::Test;

TEST_F(DescriptorTensorTest, tensor_names) {
    auto arg0 = std::make_shared<Parameter>(element::f32, Shape{1});
    arg0->set_friendly_name("data");
    arg0->get_output_tensor(0).set_names({"input"});

    auto relu = std::make_shared<Relu>(arg0);
    relu->set_friendly_name("relu");
    relu->get_output_tensor(0).set_names({"relu_t", "identity"});
    auto f0 = std::make_shared<Model>(relu, ParameterVector{arg0});

    EXPECT_EQ(arg0->get_output_tensor(0).get_names(), relu->get_input_tensor(0).get_names());
    EXPECT_EQ(arg0->get_output_tensor(0).get_names(), relu->input_value(0).get_tensor().get_names());
    EXPECT_EQ(f0->get_result()->get_input_tensor(0).get_names(), relu->get_output_tensor(0).get_names());
    EXPECT_EQ(f0->get_result()->input_value(0).get_tensor().get_names(), relu->get_output_tensor(0).get_names());
}

}  // namespace ov::test
