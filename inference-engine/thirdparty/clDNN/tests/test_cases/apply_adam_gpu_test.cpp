/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/apply_adam.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"
#include <api/CPP/reorder.hpp>
#include <api/CPP/data.hpp>
#include <api/CPP/activation.hpp>
#include <api/CPP/mutable_data.hpp>

using namespace cldnn;
using namespace tests;

TEST(apply_adam_gpu, basic_in2x2x3x2_bfyx) {
    // Test creates topology with two apply adam primitives (t = [0, 1]) with the same output variable which is updated.

    engine engine;

    auto input_grad = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto var = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto m = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto v = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto beta1_power = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto beta2_power = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    float input_grad_f = 100.f;
    float var_f = 3.f;
    float m_f = 50.f;
    float v_f = 121.f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float beta1_power_f = beta1;
    float beta2_power_f = beta2;
    float lr = 0.001f;
    float epsilon = 0.0001f;

    topology topology;
    topology.add(input_layout("input", input_grad.get_layout()));
    topology.add(mutable_data("m", m));
    topology.add(mutable_data("v", v));
    topology.add(data("beta1_power_t1", beta1_power));
    topology.add(data("beta2_power_t1", beta2_power));
    topology.add(apply_adam("apply_adam", "input", "m", "v", "beta1_power_t1", "beta2_power_t1", lr, beta1, beta2, epsilon));
    topology.add(activation("relu", "input", activation_linear, { 4.f, 0.f }));
    topology.add(activation("beta1_power_t2", "beta1_power_t1", activation_linear, { beta1, 0.f }));
    topology.add(activation("beta2_power_t2", "beta2_power_t1", activation_linear, { beta2, 0.f }));
    topology.add(apply_adam("apply_adam2", "relu", "m", "v", "beta1_power_t2", "beta2_power_t2", lr, beta1, beta2, epsilon));
    topology.add(mutable_data("var", { "apply_adam", "apply_adam2" }, var));

    set_values(input_grad, {
        input_grad_f
    });

    set_values(m, { m_f });
    set_values(v, { v_f });
    set_values(beta1_power, { beta1_power_f });
    set_values(beta2_power, { beta2_power_f });
    set_values(var, { var_f });

    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    network network(engine, topology, bo);

    network.set_input_data("input", input_grad);

    auto outputs = network.execute();

    auto output = outputs.at("var").get_memory();
    auto output_ptr = output.pointer<float>();
    auto m_ptr = m.pointer<float>();
    auto v_ptr = v.pointer<float>();

    float lr_t1 = lr * sqrt(1 - beta2_power_f) / (1 - beta1_power_f);
    float m_t1 = beta1 * m_f + (1 - beta1) * input_grad_f;
    float v_t1 = beta2 * v_f + (1 - beta2) * input_grad_f * input_grad_f;
    float result_t1 = var_f - lr_t1 * m_t1 / (sqrt(v_t1) + epsilon);

    beta1_power_f *= beta1;
    beta2_power_f *= beta2;
    float input_grad2_f = input_grad_f * 4;
    float lr_t2 = lr * sqrt(1 - beta2_power_f) / (1 - beta1_power_f);
    float m_t2 = beta1 * m_t1 + (1 - beta1) * input_grad2_f;
    float v_t2 = beta2 * v_t1 + (1 - beta2) * input_grad2_f * input_grad2_f;
    float result_t2 = result_t1 - lr_t2 * m_t2 / (sqrt(v_t2) + epsilon);

    EXPECT_NEAR(m_t2, m_ptr[0], 1e-03F);
    EXPECT_NEAR(v_t2, v_ptr[0], 1e-03F);
    EXPECT_NEAR(result_t2, output_ptr[0], 1e-03F);
}