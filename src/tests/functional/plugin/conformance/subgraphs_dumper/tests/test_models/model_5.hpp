// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/add.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"
#include "matchers/subgraph/read_value_assign.hpp"

class Model_5 {
protected:
    std::shared_ptr<ov::Model> model;

public:
    Model_5() {
        //                        param_0
        //                           |
        //                       readVal_0
        //                           |
        //                         relu_1
        //                           |
        //                        assign_0
        //                           |              param_1
        //                           |-----------------+
        //                         add_2
        //                           |
        //                        assign_1
        //                           |
        //                        result_0


        auto param_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});
        param_0->set_friendly_name("param_0");

        auto variable_info = ov::op::util::VariableInfo{ov::PartialShape{1, 80}, ov::element::f32, "id_0"};
        auto variable = std::make_shared<ov::op::util::Variable>(variable_info);
        auto readVal_0 = std::make_shared<ov::op::v6::ReadValue>(param_0, variable);
        readVal_0->set_friendly_name("readVal_0");

        auto relu_0 = std::make_shared<ov::op::v0::Relu>(readVal_0);
        relu_0->set_friendly_name("relu_0");

        auto assign_3 = std::make_shared<ov::op::v6::Assign>(relu_0, variable);
        assign_3->set_friendly_name("assign_3");

        auto param_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});
        param_1->set_friendly_name("param_1");

        auto add_1 = std::make_shared<ov::op::v1::Add>(param_1, assign_3);
        add_1->set_friendly_name("add_1");

        auto variable_info_1 = ov::op::util::VariableInfo{ov::PartialShape{1, 80}, ov::element::f32, "id_1"};
        auto variable_1 = std::make_shared<ov::op::util::Variable>(variable_info_1);
        auto assign_1 = std::make_shared<ov::op::v6::Assign>(add_1, variable_1);
        assign_1->set_friendly_name("assign_1");

        auto result_0 = std::make_shared<ov::op::v0::Result>(assign_1);
        result_0->set_friendly_name("result_0");

        model = std::make_shared<ov::Model>(ov::ResultVector{result_0},
                                    ov::ParameterVector{param_0, param_1});
    }

    std::shared_ptr<ov::Model> get() {
        return model;
    }

    std::vector<std::shared_ptr<ov::Model>> get_ref_models() {
        std::vector<std::shared_ptr<ov::Model>> ref_models;

        auto param_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 80});

        auto variable_info = ov::op::util::VariableInfo{ov::PartialShape{1, 80}, ov::element::f32, "id_0"};
        auto variable = std::make_shared<ov::op::util::Variable>(variable_info);
        auto readVal = std::make_shared<ov::op::v6::ReadValue>(param_0, variable);

        auto relu_0 = std::make_shared<ov::op::v0::Relu>(readVal);

        auto assign = std::make_shared<ov::op::v6::Assign>(relu_0, variable);

        auto result_0 = std::make_shared<ov::op::v0::Result>(assign);

        auto ref_model = std::make_shared<ov::Model>(ov::ResultVector{result_0},
                                                     ov::ParameterVector{param_0});
        ref_models.push_back(ref_model);

        return ref_models;
    }
};
