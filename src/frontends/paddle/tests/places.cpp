// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <frontend/shared/include/utils.hpp>
#include <fstream>
#include <iostream>
#include <openvino/frontend/manager.hpp>

#include "gtest/gtest.h"
#include "paddle_utils.hpp"

using namespace ov::frontend;

const std::string model_file = FrontEndTestUtils::make_model_path(std::string(TEST_PADDLE_MODELS_DIRNAME) +
                                                                  "place_test_model/place_test_model.pdmodel");
const std::string vars_name_file =
    FrontEndTestUtils::make_model_path(std::string(TEST_PADDLE_MODELS_DIRNAME) + "place_test_model/vars_name.txt");
const std::string outputs_name_file =
    FrontEndTestUtils::make_model_path(std::string(TEST_PADDLE_MODELS_DIRNAME) + "place_test_model/outputs_name.txt");

class Paddle_Places : public ::testing::Test {
protected:
    void SetUp() override {
        std::fstream name_file;
        name_file.open(vars_name_file, std::ios::in);
        if (name_file.is_open()) {
            std::string name;
            while (std::getline(name_file, name))
                tensor_names.push_back(name);
            name_file.close();
        } else
            FRONT_END_THROW("Can not open " + vars_name_file);

        std::fstream output_file;
        output_file.open(outputs_name_file, std::ios::in);
        if (output_file.is_open()) {
            std::string name;
            while (std::getline(output_file, name))
                output_names.push_back(name);
            output_file.close();
        } else
            FRONT_END_THROW("Can not open " + outputs_name_file);
    }

    std::vector<std::string> tensor_names;
    std::vector<std::string> output_names;
};

TEST_F(Paddle_Places, check_tensor_names) {
    auto fem = FrontEndManager();
    FrontEnd::Ptr frontend;
    OV_ASSERT_NO_THROW(frontend = fem.load_by_framework(PADDLE_FE));
    InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = frontend->load(FrontEndTestUtils::make_model_path(model_file)));

    for (const auto& tensor_name : tensor_names) {
        auto place = input_model->get_place_by_tensor_name(tensor_name);
        EXPECT_NE(place, nullptr);
    }
}

TEST_F(Paddle_Places, check_input_outputs) {
    auto fem = FrontEndManager();
    FrontEnd::Ptr frontend;
    OV_ASSERT_NO_THROW(frontend = fem.load_by_framework(PADDLE_FE));
    InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = frontend->load(FrontEndTestUtils::make_model_path(model_file)));

    auto inputs = input_model->get_inputs();
    auto outputs = input_model->get_outputs();

    EXPECT_EQ(inputs.size(), 1);
    EXPECT_EQ(outputs.size(), 6);

    auto tensor_place = input_model->get_place_by_tensor_name("x");
    tensor_place->is_equal(inputs[0]);

    for (const auto& name : output_names) {
        const auto output_place = input_model->get_place_by_tensor_name(name);
        auto it = std::find_if(outputs.begin(), outputs.end(), [&output_place](const Place::Ptr& place) {
            return output_place->is_equal(place);
        });
        EXPECT_NE(it, outputs.end());
    }
}

// all existed in the model ops have "Out" port
TEST_F(Paddle_Places, check_out_port_of_all_ops) {
    auto fem = FrontEndManager();
    FrontEnd::Ptr frontend;
    OV_ASSERT_NO_THROW(frontend = fem.load_by_framework(PADDLE_FE));
    InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = frontend->load(FrontEndTestUtils::make_model_path(model_file)));

    for (const auto& tensor_name : tensor_names) {
        auto place = input_model->get_place_by_tensor_name(tensor_name);
        EXPECT_NE(place, nullptr);

        auto producing_op = place->get_producing_operation();
        EXPECT_NE(producing_op, nullptr);
        auto out_port_by_name = producing_op->get_output_port("Out");
        EXPECT_NE(out_port_by_name, nullptr);
        auto out_port_by_name_idx = producing_op->get_output_port("Out", 0);
        EXPECT_NE(out_port_by_name_idx, nullptr);

        EXPECT_TRUE(out_port_by_name->is_equal(out_port_by_name_idx));
    }
}

TEST_F(Paddle_Places, check_in_out_ports_of_model_outputs) {
    auto fem = FrontEndManager();
    FrontEnd::Ptr frontend;
    OV_ASSERT_NO_THROW(frontend = fem.load_by_framework(PADDLE_FE));
    InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = frontend->load(FrontEndTestUtils::make_model_path(model_file)));

    auto outputs = input_model->get_outputs();
    for (const auto& output : outputs) {
        auto producing_op = output->get_producing_operation();
        EXPECT_NE(producing_op, nullptr);

        auto out_port = producing_op->get_output_port();
        EXPECT_NE(out_port, nullptr);
        auto out_port_by_name = producing_op->get_output_port("Out");
        EXPECT_NE(out_port_by_name, nullptr);
        auto out_port_by_name_idx = producing_op->get_output_port("Out", 0);
        EXPECT_NE(out_port_by_name_idx, nullptr);

        EXPECT_TRUE(out_port->is_equal(out_port_by_name));
        EXPECT_TRUE(out_port->is_equal(out_port_by_name_idx));

        auto in_port = producing_op->get_input_port();
        EXPECT_NE(in_port, nullptr);
        auto in_port_by_name = producing_op->get_input_port("X");
        EXPECT_NE(in_port_by_name, nullptr);
        auto in_port_by_name_idx = producing_op->get_input_port("X", 0);
        EXPECT_NE(in_port_by_name_idx, nullptr);

        EXPECT_TRUE(in_port->is_equal(in_port_by_name));
        EXPECT_TRUE(in_port->is_equal(in_port_by_name_idx));
    }
}

TEST_F(Paddle_Places, check_source_target_tensors_of_model_outputs) {
    auto fem = FrontEndManager();
    FrontEnd::Ptr frontend;
    OV_ASSERT_NO_THROW(frontend = fem.load_by_framework(PADDLE_FE));
    InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = frontend->load(FrontEndTestUtils::make_model_path(model_file)));

    auto outputs = input_model->get_outputs();
    for (const auto& output : outputs) {
        auto producing_op = output->get_producing_operation();
        EXPECT_NE(producing_op, nullptr);

        auto out = producing_op->get_target_tensor();
        EXPECT_NE(out, nullptr);
        auto out_by_name = producing_op->get_target_tensor("Out");
        EXPECT_NE(out_by_name, nullptr);
        auto out_by_name_idx = producing_op->get_target_tensor("Out", 0);
        EXPECT_NE(out_by_name_idx, nullptr);

        EXPECT_TRUE(out->is_equal(out_by_name));
        EXPECT_TRUE(out->is_equal(out_by_name_idx));

        auto in = producing_op->get_source_tensor();
        EXPECT_NE(in, nullptr);
        auto in_by_name = producing_op->get_source_tensor("X");
        EXPECT_NE(in_by_name, nullptr);
        auto in_by_name_idx = producing_op->get_source_tensor("X", 0);
        EXPECT_NE(in_by_name_idx, nullptr);

        EXPECT_TRUE(in->is_equal(in_by_name));
        EXPECT_TRUE(in->is_equal(in_by_name_idx));
    }
}

TEST_F(Paddle_Places, check_producing_consuming_ops_of_model_outputs) {
    auto fem = FrontEndManager();
    FrontEnd::Ptr frontend;
    OV_ASSERT_NO_THROW(frontend = fem.load_by_framework(PADDLE_FE));
    InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = frontend->load(FrontEndTestUtils::make_model_path(model_file)));

    auto outputs = input_model->get_outputs();
    for (const auto& output : outputs) {
        auto op = output->get_producing_operation();
        EXPECT_NE(op, nullptr);

        auto out = op->get_consuming_operations();
        EXPECT_EQ(out.size(), 1);
        auto out_by_name = op->get_consuming_operations("Out");
        EXPECT_EQ(out_by_name.size(), 1);
        auto out_by_name_idx = op->get_consuming_operations("Out", 0);
        EXPECT_EQ(out_by_name_idx.size(), 1);

        EXPECT_TRUE(out[0]->is_equal(out_by_name[0]));
        EXPECT_TRUE(out[0]->is_equal(out_by_name_idx[0]));

        auto in = op->get_producing_operation();
        EXPECT_NE(in, nullptr);
        auto in_by_name = op->get_producing_operation("X");
        EXPECT_NE(in_by_name, nullptr);
        auto in_by_name_idx = op->get_producing_operation("X", 0);
        EXPECT_NE(in_by_name_idx, nullptr);

        EXPECT_TRUE(in->is_equal(in_by_name));
        EXPECT_TRUE(in->is_equal(in_by_name_idx));
    }
}

// check data flow [ output port -> tensor -> input port ]
TEST_F(Paddle_Places, check_data_flow) {
    auto fem = FrontEndManager();
    FrontEnd::Ptr frontend;
    OV_ASSERT_NO_THROW(frontend = fem.load_by_framework(PADDLE_FE));
    InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = frontend->load(FrontEndTestUtils::make_model_path(model_file)));

    for (const auto& tensor_name : tensor_names) {
        auto tensor_place = input_model->get_place_by_tensor_name(tensor_name);
        EXPECT_NE(tensor_place, nullptr);

        auto out_port = tensor_place->get_producing_port();
        auto in_ports = tensor_place->get_consuming_ports();
        EXPECT_TRUE(tensor_place->is_equal_data(out_port));
        EXPECT_TRUE(out_port->is_equal_data(tensor_place));
        EXPECT_FALSE(out_port->is_equal(tensor_place));

        auto source_tensor = out_port->get_target_tensor();
        EXPECT_TRUE(source_tensor->is_equal(tensor_place));
        for (const auto& in_port : in_ports) {
            EXPECT_TRUE(out_port->is_equal_data(in_port));
            EXPECT_TRUE(in_port->is_equal_data(out_port));

            EXPECT_TRUE(in_port->is_equal_data(tensor_place));
            EXPECT_TRUE(tensor_place->is_equal_data(in_port));

            EXPECT_FALSE(in_port->is_equal(out_port));
            EXPECT_FALSE(in_port->is_equal(tensor_place));

            EXPECT_TRUE(out_port->is_equal(in_port->get_producing_port()));
            EXPECT_TRUE(tensor_place->is_equal(in_port->get_source_tensor()));
        }
    }
}

// check [ tensor -> input_port
//                -> input_port_2
//                -> input_port_N]
// input_port, input_port_2, ... input_port_N are equal data
TEST_F(Paddle_Places, check_tensor_to_multiple_ports) {
    auto fem = FrontEndManager();
    FrontEnd::Ptr frontend;
    OV_ASSERT_NO_THROW(frontend = fem.load_by_framework(PADDLE_FE));
    InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = frontend->load(FrontEndTestUtils::make_model_path(model_file)));

    for (const auto& tensor_name : tensor_names) {
        auto tensor_place = input_model->get_place_by_tensor_name(tensor_name);
        auto inputs_to = tensor_place->get_consuming_ports();
        for (size_t idx = 0; idx < inputs_to.size(); ++idx) {
            for (size_t idx_2 = 0; idx_2 < inputs_to.size(); ++idx_2) {
                EXPECT_TRUE(inputs_to[idx]->is_equal_data(inputs_to[idx_2]));
                EXPECT_TRUE(inputs_to[idx_2]->is_equal_data(inputs_to[idx]));

                if (idx == idx_2) {
                    EXPECT_TRUE(inputs_to[idx]->is_equal(inputs_to[idx_2]));
                } else {
                    EXPECT_FALSE(inputs_to[idx]->is_equal(inputs_to[idx_2]));
                }
            }
        }
    }
}

// consuming ops should be equal for tensor place and producing output port
TEST_F(Paddle_Places, check_consuming_ops) {
    auto fem = FrontEndManager();
    FrontEnd::Ptr frontend;
    OV_ASSERT_NO_THROW(frontend = fem.load_by_framework(PADDLE_FE));
    InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = frontend->load(FrontEndTestUtils::make_model_path(model_file)));

    for (const auto& tensor_name : tensor_names) {
        auto tensor_place = input_model->get_place_by_tensor_name(tensor_name);
        EXPECT_NE(tensor_place, nullptr);

        auto consuming_ops_for_tensor = tensor_place->get_consuming_operations();
        auto out_port = tensor_place->get_producing_port();
        auto consuming_ops_for_out_port = out_port->get_consuming_operations();

        bool is_permutation = std::is_permutation(consuming_ops_for_out_port.begin(),
                                                  consuming_ops_for_out_port.end(),
                                                  consuming_ops_for_tensor.begin(),
                                                  [](const Place::Ptr& place1, const Place::Ptr& place2) {
                                                      return place1->is_equal(place2);
                                                  });

        EXPECT_TRUE(is_permutation);

        auto consuming_ports_for_tensor = tensor_place->get_consuming_ports();
        std::vector<Place::Ptr> consuming_ops_for_in_ports;
        for (const auto& port : consuming_ports_for_tensor) {
            EXPECT_EQ(port->get_consuming_operations().size(), 1);
            consuming_ops_for_in_ports.push_back(port->get_consuming_operations()[0]);
        }

        is_permutation = std::is_permutation(consuming_ops_for_in_ports.begin(),
                                             consuming_ops_for_in_ports.end(),
                                             consuming_ops_for_tensor.begin(),
                                             [](const Place::Ptr& place1, const Place::Ptr& place2) {
                                                 return place1->is_equal(place2);
                                             });
        EXPECT_TRUE(is_permutation);
    }
}

TEST_F(Paddle_Places, check_consuming_ops_2) {
    auto fem = FrontEndManager();
    FrontEnd::Ptr frontend;
    OV_ASSERT_NO_THROW(frontend = fem.load_by_framework(PADDLE_FE));
    InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = frontend->load(FrontEndTestUtils::make_model_path(model_file)));

    auto it = find(tensor_names.begin(), tensor_names.end(), "lstm_0.tmp_2");
    EXPECT_NE(it, tensor_names.end());

    auto tensor_place = input_model->get_place_by_tensor_name(*it);
    auto consuming_ports = tensor_place->get_consuming_ports();
    auto consuming_ops = tensor_place->get_consuming_operations();
    EXPECT_EQ(consuming_ports.size(), 4);
    EXPECT_EQ(consuming_ops.size(), 4);

    for (const auto& consuming_port : consuming_ports) {
        auto port_consuming_ops = consuming_port->get_consuming_operations();
        EXPECT_EQ(port_consuming_ops.size(), 1);

        auto in_port = port_consuming_ops[0]->get_input_port();
        auto in_port_by_name = port_consuming_ops[0]->get_input_port("X");
        auto in_port_by_name_and_idx = port_consuming_ops[0]->get_input_port("X", 0);

        EXPECT_TRUE(consuming_port->is_equal(in_port) && consuming_port->is_equal(in_port_by_name) &&
                    consuming_port->is_equal(in_port_by_name_and_idx));

        auto op =
            std::find_if(consuming_ops.begin(), consuming_ops.end(), [&port_consuming_ops](const Place::Ptr& place) {
                return place->is_equal(port_consuming_ops[0]);
            });
        EXPECT_NE(op, consuming_ops.end());

        const auto source_tensor = port_consuming_ops[0]->get_source_tensor();
        EXPECT_TRUE(source_tensor->is_equal(tensor_place));
        EXPECT_TRUE(source_tensor->is_equal(consuming_port->get_source_tensor()));
    }
}

TEST_F(Paddle_Places, check_producing_ops) {
    auto fem = FrontEndManager();
    FrontEnd::Ptr frontend;
    OV_ASSERT_NO_THROW(frontend = fem.load_by_framework(PADDLE_FE));
    InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = frontend->load(FrontEndTestUtils::make_model_path(model_file)));

    for (const auto& tensor_name : tensor_names) {
        auto tensor_place = input_model->get_place_by_tensor_name(tensor_name);
        EXPECT_NE(tensor_place, nullptr);

        auto producing_op = tensor_place->get_producing_operation();
        auto consuming_ports = tensor_place->get_consuming_ports();
        auto producing_port = tensor_place->get_producing_port();

        EXPECT_TRUE(producing_op->is_equal(producing_port->get_producing_operation()));
        for (const auto& consuming_port : consuming_ports) {
            EXPECT_TRUE(producing_op->is_equal(consuming_port->get_producing_operation()));
        }
    }
}

TEST_F(Paddle_Places, check_input_output_ports_dy_idx) {
    auto fem = FrontEndManager();
    FrontEnd::Ptr frontend;
    OV_ASSERT_NO_THROW(frontend = fem.load_by_framework(PADDLE_FE));
    InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = frontend->load(FrontEndTestUtils::make_model_path(model_file)));

    for (const auto& tensor_name : output_names) {
        auto tensor_place = input_model->get_place_by_tensor_name(tensor_name);
        EXPECT_NE(tensor_place, nullptr);

        auto op = tensor_place->get_producing_operation();
        auto input_port = op->get_input_port(0);
        EXPECT_NE(input_port, nullptr);
        auto out_port = op->get_output_port(0);
        EXPECT_NE(out_port, nullptr);
    }
}

TEST_F(Paddle_Places, check_ops_tensors_by_idx) {
    auto fem = FrontEndManager();
    FrontEnd::Ptr frontend;
    OV_ASSERT_NO_THROW(frontend = fem.load_by_framework(PADDLE_FE));
    InputModel::Ptr input_model;
    OV_ASSERT_NO_THROW(input_model = frontend->load(FrontEndTestUtils::make_model_path(model_file)));

    for (const auto& tensor_name : output_names) {
        auto tensor_place = input_model->get_place_by_tensor_name(tensor_name);
        EXPECT_NE(tensor_place, nullptr);

        auto op = tensor_place->get_producing_operation();
        auto prod_op = op->get_producing_operation(0);
        EXPECT_NE(prod_op, nullptr);

        auto target_tensor = op->get_target_tensor(0);
        EXPECT_EQ(tensor_place, target_tensor);

        auto source_tensor = op->get_source_tensor(0);
        EXPECT_NE(source_tensor, nullptr);

        auto consum_op = op->get_consuming_operations(0);
        EXPECT_EQ(consum_op.size(), 1);
    }
}
