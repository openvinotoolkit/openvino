// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/descriptor_tensor.hpp"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/test_tools.hpp"
#include "gmock/gmock.h"
#include "openvino/core/model.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"

namespace ov::test {
using testing::UnorderedElementsAre, testing::_;

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

TEST_F(DescriptorTensorTest, update_names_on_shared_tensor_from_result_output) {
    const auto data = std::make_shared<Parameter>(element::f32, Shape{1});
    const auto relu = std::make_shared<Relu>(data);
    const auto result = std::make_shared<Result>(relu);

    auto& relu_tensor = relu->get_output_tensor(0);
    auto& result_tensor = result->get_output_tensor(0);

    result_tensor.set_names({"result"});
    EXPECT_THAT(relu_tensor.get_names(), UnorderedElementsAre("result"));

    result_tensor.add_names({"my_output"});
    EXPECT_THAT(relu_tensor.get_names(), UnorderedElementsAre("result", "my_output"));
}

TEST_F(DescriptorTensorTest, update_names_on_shared_tensor_from_result_input) {
    const auto data = std::make_shared<Parameter>(element::f32, Shape{1});
    const auto relu = std::make_shared<Relu>(data);
    const auto result = std::make_shared<Result>(relu);

    auto& relu_tensor = relu->get_output_tensor(0);
    auto& result_tensor = result->get_output_tensor(0);

    relu_tensor.set_names({"relu"});
    EXPECT_THAT(relu_tensor.get_names(), UnorderedElementsAre("relu"));
    EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre("relu"));

    relu_tensor.add_names({"identity"});
    EXPECT_THAT(relu_tensor.get_names(), UnorderedElementsAre("relu", "identity"));
    EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre("relu", "identity"));
}

TEST_F(DescriptorTensorTest, update_names_on_shared_tensor_with_names_from_result_input) {
    const auto data = std::make_shared<Parameter>(element::f32, Shape{1});
    const auto relu = std::make_shared<Relu>(data);
    const auto result = std::make_shared<Result>(relu);

    auto& relu_tensor = relu->get_output_tensor(0);
    auto& result_tensor = result->get_output_tensor(0);
    result_tensor.set_names({"result"});

    {
        relu_tensor.set_names({"relu"});
        EXPECT_THAT(relu_tensor.get_names(), UnorderedElementsAre("relu", "result"));
        EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre("result"));
    }
    {
        relu_tensor.add_names({"identity"});
        EXPECT_THAT(relu_tensor.get_names(), UnorderedElementsAre("relu", "identity", "result"));
        EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre("result"));
    }
}

TEST_F(DescriptorTensorTest, set_names_on_shared_tensor_from_temporary_results) {
    const auto data = std::make_shared<Parameter>(element::f32, Shape{1});
    const auto relu = std::make_shared<Relu>(data);
    const auto relu_name = "relu:0";
    auto& relu_tensor = relu->get_output_tensor(0);
    relu_tensor.set_names({relu_name});

    {
        const auto result_name = "result1";
        const auto result = std::make_shared<Result>(relu);
        auto& result_tensor = result->get_output_tensor(0);

        result_tensor.set_names({result_name});
        EXPECT_THAT(relu_tensor.get_names(), UnorderedElementsAre(relu_name, result_name));
        EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre(result_name));
    }
    {
        const auto result_name = "result2";
        const auto result = std::make_shared<Result>(relu);
        auto& result_tensor = result->get_output_tensor(0);

        EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre(relu_name));

        result_tensor.set_names({result_name});
        EXPECT_THAT(relu_tensor.get_names(), UnorderedElementsAre(relu_name, result_name));
        EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre(result_name));
    }

    EXPECT_THAT(relu_tensor.get_names(), UnorderedElementsAre(relu_name));
}

TEST_F(DescriptorTensorTest, set_names_on_shared_tensor_multiple_results) {
    const auto data = std::make_shared<Parameter>(element::f32, Shape{1});
    const auto relu = std::make_shared<Relu>(data);
    const auto relu_name = "relu:0";
    auto& relu_tensor = relu->get_output_tensor(0);
    relu_tensor.set_names({relu_name});

    {
        const auto result_name = "result1";
        const auto result = std::make_shared<Result>(relu);
        auto& result_tensor = result->get_output_tensor(0);

        result_tensor.set_names({result_name});
        EXPECT_THAT(relu_tensor.get_names(), UnorderedElementsAre(relu_name, result_name));
        EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre(result_name));
        {
            const auto new_result_name = "result2";
            const auto new_result = std::make_shared<Result>(result);
            auto& new_result_tensor = new_result->get_output_tensor(0);
            EXPECT_THAT(new_result_tensor.get_names(), UnorderedElementsAre(relu_name, result_name));

            new_result_tensor.set_names({new_result_name});
            EXPECT_THAT(relu_tensor.get_names(), UnorderedElementsAre(relu_name, new_result_name, result_name));
            EXPECT_THAT(new_result_tensor.get_names(), UnorderedElementsAre(new_result_name));
        }
    }

    EXPECT_THAT(relu_tensor.get_names(), UnorderedElementsAre(relu_name));
}

TEST_F(DescriptorTensorTest, replace_shared_tensor_for_result) {
    const auto data = std::make_shared<Parameter>(element::f32, Shape{1});
    const auto relu1 = std::make_shared<Relu>(data);
    const auto relu2 = std::make_shared<Relu>(data);
    const auto relu1_name = "relu1:0";
    const auto relu2_name = "relu2:0";
    auto& relu1_tensor = relu1->get_output_tensor(0);
    auto& relu2_tensor = relu2->get_output_tensor(0);
    relu1_tensor.set_names({relu1_name});
    relu2_tensor.set_names({relu2_name});

    const auto result_name = "result1";
    const auto result = std::make_shared<Result>(relu1);
    auto& result_tensor = result->get_output_tensor(0);
    result_tensor.set_names({result_name});

    const auto opt_result = std::make_shared<Result>(relu1);
    const auto opt_result_name = "opt_result";
    auto& opt_result_tensor = opt_result->get_output_tensor(0);
    opt_result_tensor.set_names({opt_result_name});

    EXPECT_THAT(relu1_tensor.get_names(), UnorderedElementsAre(relu1_name, result_name, opt_result_name));
    EXPECT_THAT(relu2_tensor.get_names(), UnorderedElementsAre(relu2_name));
    EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre(result_name));
    EXPECT_THAT(opt_result_tensor.get_names(), UnorderedElementsAre(opt_result_name));

    result->input(0).replace_source_output(relu2);
    result->validate_and_infer_types();

    EXPECT_THAT(relu1_tensor.get_names(), UnorderedElementsAre(relu1_name, opt_result_name));
    EXPECT_THAT(relu2_tensor.get_names(), UnorderedElementsAre(relu2_name, result_name));
    EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre(result_name));
    EXPECT_THAT(opt_result_tensor.get_names(), UnorderedElementsAre(opt_result_name));
}

TEST_F(DescriptorTensorTest, replace_shared_tensor_for_result_as_graph_branch) {
    const auto data = std::make_shared<Parameter>(element::f32, Shape{1});
    const auto relu1 = std::make_shared<Relu>(data);
    const auto relu2 = std::make_shared<Relu>(relu1);
    const auto relu1_name = "relu1:0";
    const auto relu2_name = "relu2:0";
    auto& relu1_tensor = relu1->get_output_tensor(0);
    auto& relu2_tensor = relu2->get_output_tensor(0);
    relu1_tensor.set_names({relu1_name});
    relu2_tensor.set_names({relu2_name});
    const auto result_name = "result1";
    const auto branch_result = std::make_shared<Result>(relu1);
    auto& result_tensor = branch_result->get_output_tensor(0);
    result_tensor.set_names({result_name});

    {
        const auto opt_result = std::make_shared<Result>(relu1);
        auto& opt_result_tensor = opt_result->get_output_tensor(0);

        EXPECT_THAT(relu1_tensor.get_names(), UnorderedElementsAre(relu1_name, result_name));
        EXPECT_THAT(relu2_tensor.get_names(), UnorderedElementsAre(relu2_name));
        EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre(result_name));
        EXPECT_THAT(opt_result_tensor.get_names(), UnorderedElementsAre(relu1_name, result_name));
    }

    EXPECT_THAT(relu1_tensor.get_names(), UnorderedElementsAre(relu1_name, result_name));
    EXPECT_THAT(relu2_tensor.get_names(), UnorderedElementsAre(relu2_name));
    EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre(result_name));
}

TEST_F(DescriptorTensorTest, update_names_on_parameter_tensor_from_result_output) {
    const auto data = std::make_shared<Parameter>(element::f32, Shape{1});
    const auto result = std::make_shared<Result>(data);

    auto& data_tensor = data->get_output_tensor(0);
    auto& result_tensor = result->get_output_tensor(0);

    result_tensor.set_names({"result"});
    EXPECT_THAT(data_tensor.get_names(), UnorderedElementsAre("result"));

    result_tensor.add_names({"my_output"});
    EXPECT_THAT(data_tensor.get_names(), UnorderedElementsAre("result", "my_output"));
}

TEST_F(DescriptorTensorTest, update_names_on_paramter_tensor_from_result_input) {
    const auto data = std::make_shared<Parameter>(element::f32, Shape{1});
    const auto data_name = "data";
    const auto result = std::make_shared<Result>(data, true);

    auto& data_tensor = data->get_output_tensor(0);
    auto& result_tensor = result->get_output_tensor(0);

    data_tensor.set_names({data_name});
    EXPECT_THAT(data_tensor.get_names(), UnorderedElementsAre(data_name));
    EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre(data_name));

    data_tensor.add_names({"identity"});
    EXPECT_THAT(data_tensor.get_names(), UnorderedElementsAre(data_name, "identity"));
    EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre(data_name, "identity"));
}

TEST_F(DescriptorTensorTest, set_names_on_parameter_tensor_from_temporary_results) {
    const auto data = std::make_shared<Parameter>(element::f32, Shape{1});
    const auto data_name = "data:0";
    auto& data_tensor = data->get_output_tensor(0);
    data_tensor.set_names({data_name});

    {
        const auto result_name = "result1";
        const auto result = std::make_shared<Result>(data, true);
        auto& result_tensor = result->get_output_tensor(0);

        result_tensor.set_names({result_name});
        EXPECT_THAT(data_tensor.get_names(), UnorderedElementsAre(data_name, result_name));
        EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre(result_name));
    }
    {
        const auto result_name = "result2";
        const auto result = std::make_shared<Result>(data, true);
        auto& result_tensor = result->get_output_tensor(0);

        EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre(data_name));

        result_tensor.set_names({result_name});
        EXPECT_THAT(data_tensor.get_names(), UnorderedElementsAre(data_name, result_name));
        EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre(result_name));
    }

    EXPECT_THAT(data_tensor.get_names(), UnorderedElementsAre(data_name));
}

TEST_F(DescriptorTensorTest, replace_parameter_tensor_in_result_node) {
    const auto data1 = std::make_shared<Parameter>(element::f32, Shape{1});
    const auto data2 = std::make_shared<Parameter>(element::f32, Shape{1});
    const auto data1_name = "param_1";
    const auto data2_name = "param_2";
    auto& data1_tensor = data1->get_output_tensor(0);
    auto& data2_tensor = data2->get_output_tensor(0);
    data1_tensor.set_names({data1_name});
    data2_tensor.set_names({data2_name});

    const auto result_name = "result1";
    const auto result = std::make_shared<Result>(data1);
    auto& result_tensor = result->get_output_tensor(0);
    result_tensor.set_names({result_name});

    const auto opt_result = std::make_shared<Result>(data1);
    const auto opt_result_name = "opt_result";
    auto& opt_result_tensor = opt_result->get_output_tensor(0);
    opt_result_tensor.set_names({opt_result_name});

    EXPECT_THAT(data1_tensor.get_names(), UnorderedElementsAre(data1_name, result_name, opt_result_name));
    EXPECT_THAT(data2_tensor.get_names(), UnorderedElementsAre(data2_name));
    EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre(result_name));
    EXPECT_THAT(opt_result_tensor.get_names(), UnorderedElementsAre(opt_result_name));

    result->input(0).replace_source_output(data2);
    result->validate_and_infer_types();

    EXPECT_THAT(data1_tensor.get_names(), UnorderedElementsAre(data1_name, opt_result_name));
    EXPECT_THAT(data2_tensor.get_names(), UnorderedElementsAre(data2_name, result_name));
    EXPECT_THAT(result_tensor.get_names(), UnorderedElementsAre(result_name));
    EXPECT_THAT(opt_result_tensor.get_names(), UnorderedElementsAre(opt_result_name));
}

TEST_F(DescriptorTensorTest, set_result_tensor_name_same_as_paramater_name) {
    GTEST_SKIP() << "Enable test when names validation added and test issues resolved";

    const auto data = std::make_shared<Parameter>(element::f32, Shape{1});
    const auto result = std::make_shared<Result>(data);

    auto& data_tensor = data->get_output_tensor(0);
    auto& result_tensor = result->get_output_tensor(0);

    data_tensor.set_names({"data", "my_output"});
    result_tensor.set_names({"result"});
    EXPECT_THAT(data_tensor.get_names(), UnorderedElementsAre("result", "data", "my_output"));

    OV_EXPECT_THROW(result_tensor.add_names({"my_output"}), ov::AssertFailure, _);
}
}  // namespace ov::test
