// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"
#include "matchers/subgraph/read_value_assign.hpp"

class Model_4 {
protected:
    std::shared_ptr<ov::Model> model;

public:
    Model_4() {
        //      Param_0                                       Param_1
        //         |                                             |
        //      ReadVal_0                                        |
        //         |                                           Relu_0
        //       Abs_0                                           |
        //         |                                             |
        //      Assign_0                                      ReadVal_1
        //         +----------------------+----------------------+
        //                             Concat_0
        //                                |
        //                              Split_0
        //                                |
        //                    +-------------------------+
        //         Param_2    |                         |
        //           +--------|                         |
        //  Param_3         Add_0                    ReadVal_3
        //     |              |                         |           Param_4
        //  ReadVal_2         |                         |--------------+
        //     +--------------|                        Add_1
        //                 Multiply_0                   |
        //                    |                      Assign_3
        //                 Assign_2                     |
        //                    |                       Relu_2
        //                  Relu_1                      |
        //                    |                       Assign_1
        //                 Result_0                     |
        //                                           Result_1
        //
        auto param_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});
        param_0->set_friendly_name("param_0");

        auto variable_info_0 = ov::op::util::VariableInfo{ov::PartialShape{1, 80}, ov::element::f32, "id_0"};
        auto variable_0 = std::make_shared<ov::op::util::Variable>(variable_info_0);
        auto readVal_0 = std::make_shared<ov::op::v6::ReadValue>(param_0, variable_0);
        readVal_0->set_friendly_name("readVal_0");

        auto abs_0 = std::make_shared<ov::op::v0::Abs>(readVal_0);
        abs_0->set_friendly_name("abs_0");

        auto assign_0 = std::make_shared<ov::op::v6::Assign>(abs_0, variable_0);
        assign_0->set_friendly_name("assign_0");

        auto param_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});
        param_1->set_friendly_name("param_1");

        auto relu_0 = std::make_shared<ov::op::v0::Relu>(param_1);
        relu_0->set_friendly_name("relu_0");

        auto variable_info_1 = ov::op::util::VariableInfo{ov::PartialShape{1, 80}, ov::element::f32, "id_1"};
        auto variable_1 = std::make_shared<ov::op::util::Variable>(variable_info_1);
        auto readVal_1 = std::make_shared<ov::op::v6::ReadValue>(relu_0, variable_1);
        readVal_1->set_friendly_name("readVal_1");

        auto concat_0 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{assign_0, readVal_1}, 1);
        concat_0->set_friendly_name("concat_0");

        auto axis_split = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>({1}));
        auto split_0 = std::make_shared<ov::op::v1::Split>(concat_0, axis_split, 2);
        split_0->set_friendly_name("split_0");

        auto param_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});
        param_2->set_friendly_name("param_2");
        auto add_0 = std::make_shared<ov::op::v1::Add>(split_0->output(0), param_2);
        add_0->set_friendly_name("add_0");

        auto param_3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});
        param_3->set_friendly_name("param_3");
        auto variable_info_2 = ov::op::util::VariableInfo{ov::PartialShape{1, 80}, ov::element::f32, "id_2"};
        auto variable_2 = std::make_shared<ov::op::util::Variable>(variable_info_2);
        auto readVal_2 = std::make_shared<ov::op::v6::ReadValue>(param_3, variable_2);
        readVal_2->set_friendly_name("readVal_2");

        auto multiply_0 = std::make_shared<ov::op::v1::Multiply>(add_0, readVal_2);
        multiply_0->set_friendly_name("multiply_0");

        auto assign_2 = std::make_shared<ov::op::v6::Assign>(multiply_0, variable_2);
        assign_2->set_friendly_name("assign_2");

        auto relu_1 = std::make_shared<ov::op::v0::Relu>(assign_2);
        relu_1->set_friendly_name("relu_1");

        auto result_0 = std::make_shared<ov::op::v0::Result>(relu_1);
        result_0->set_friendly_name("result_0");

        auto variable_info_3 = ov::op::util::VariableInfo{ov::PartialShape{1, 80}, ov::element::f32, "id_3"};
        auto variable_3 = std::make_shared<ov::op::util::Variable>(variable_info_3);
        auto readVal_3 = std::make_shared<ov::op::v6::ReadValue>(split_0->output(1), variable_3);
        readVal_3->set_friendly_name("readVal_3");

        auto param_4 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});
        param_4->set_friendly_name("param_4");
        auto add_1 = std::make_shared<ov::op::v1::Add>(param_4, readVal_3);
        add_1->set_friendly_name("add_1");

        auto assign_3 = std::make_shared<ov::op::v6::Assign>(add_1, variable_3);
        assign_3->set_friendly_name("assign_3");

        auto relu_2 = std::make_shared<ov::op::v0::Relu>(assign_3);
        relu_2->set_friendly_name("relu_2");

        auto assign_1 = std::make_shared<ov::op::v6::Assign>(relu_2, variable_1);
        assign_1->set_friendly_name("assign_1");

        auto result_1 = std::make_shared<ov::op::v0::Result>(assign_1);
        result_1->set_friendly_name("result_1");

        model = std::make_shared<ov::Model>(ov::ResultVector{result_0, result_1},
                                    ov::ParameterVector{param_0, param_1, param_2, param_3, param_4});
    }

    std::shared_ptr<ov::Model> get() {
        return model;
    }

    std::vector<std::shared_ptr<ov::Model>> get_ref_models() {
        std::vector<std::shared_ptr<ov::Model>> ref_models;
        {
            auto param_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});

            auto variable_info_0 = ov::op::util::VariableInfo{ov::PartialShape{1, 80}, ov::element::f32, "id_0"};
            auto variable_0 = std::make_shared<ov::op::util::Variable>(variable_info_0);
            auto readVal_0 = std::make_shared<ov::op::v6::ReadValue>(param_0, variable_0);

            auto abs_0 = std::make_shared<ov::op::v0::Abs>(readVal_0);

            auto assign_0 = std::make_shared<ov::op::v6::Assign>(abs_0, variable_0);

            auto result_1 = std::make_shared<ov::op::v0::Result>(assign_0);

            auto ref_model = std::make_shared<ov::Model>(ov::ResultVector{result_1},
                                                        ov::ParameterVector{param_0});
            ref_models.push_back(ref_model);
        }
        {
            auto param_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});
            auto variable_info_1 = ov::op::util::VariableInfo{ov::PartialShape{1, 80}, ov::element::f32, "id_1"};
            auto variable_1 = std::make_shared<ov::op::util::Variable>(variable_info_1);
            auto readVal_1 = std::make_shared<ov::op::v6::ReadValue>(param_1, variable_1);

            auto param_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});
            auto concat_0 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{param_2, readVal_1}, 1);

            auto axis_split = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>({1}));
            auto split_0 = std::make_shared<ov::op::v1::Split>(concat_0, axis_split, 2);

            auto variable_info_3 = ov::op::util::VariableInfo{ov::PartialShape{1, 80}, ov::element::f32, "id_3"};
            auto variable_3 = std::make_shared<ov::op::util::Variable>(variable_info_3);
            auto readVal_3 = std::make_shared<ov::op::v6::ReadValue>(split_0->output(1), variable_3);

            auto param_4 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});
            auto add_1 = std::make_shared<ov::op::v1::Add>(param_4, readVal_3);

            auto assign_3 = std::make_shared<ov::op::v6::Assign>(add_1, variable_3);

            auto relu_2 = std::make_shared<ov::op::v0::Relu>(assign_3);

            auto assign_1 = std::make_shared<ov::op::v6::Assign>(relu_2, variable_1);

            auto result_0 = std::make_shared<ov::op::v0::Result>(split_0->output(0));
            auto result_1 = std::make_shared<ov::op::v0::Result>(assign_1);

            auto ref_model = std::make_shared<ov::Model>(ov::ResultVector{result_0, result_1},
                                                         ov::ParameterVector{param_1, param_2, param_4});
            ref_models.push_back(ref_model);
        }
        {
            auto param_3 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});
            auto variable_info_2 = ov::op::util::VariableInfo{ov::PartialShape{1, 80}, ov::element::f32, "id_2"};
            auto variable_2 = std::make_shared<ov::op::util::Variable>(variable_info_2);
            auto readVal_2 = std::make_shared<ov::op::v6::ReadValue>(param_3, variable_2);
            readVal_2->set_friendly_name("readVal_2");

            auto param_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});
            auto multiply_0 = std::make_shared<ov::op::v1::Multiply>(param_0, readVal_2);
            multiply_0->set_friendly_name("multiply_0");

            auto assign_2 = std::make_shared<ov::op::v6::Assign>(multiply_0, variable_2);
            assign_2->set_friendly_name("assign_2");

            auto result_0 = std::make_shared<ov::op::v0::Result>(assign_2);

            auto ref_model = std::make_shared<ov::Model>(ov::ResultVector{result_0},
                                                        ov::ParameterVector{param_0, param_3});
            ref_models.push_back(ref_model);
        }
        {
            auto param_4 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});
            auto variable_info_3 = ov::op::util::VariableInfo{ov::PartialShape{1, 80}, ov::element::f32, "id_3"};
            auto variable_3 = std::make_shared<ov::op::util::Variable>(variable_info_3);
            auto readVal_3 = std::make_shared<ov::op::v6::ReadValue>(param_4, variable_3);

            auto param_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});
            auto add_1 = std::make_shared<ov::op::v1::Add>(param_0, readVal_3);

            auto assign_3 = std::make_shared<ov::op::v6::Assign>(add_1, variable_3);

            auto result_1 = std::make_shared<ov::op::v0::Result>(assign_3);

            auto ref_model = std::make_shared<ov::Model>(ov::ResultVector{result_1},
                                                        ov::ParameterVector{param_0, param_4});
            ref_models.push_back(ref_model);
        }

        return ref_models;
    }
};
