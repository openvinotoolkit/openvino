// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/assign.hpp>
#include <intel_gpu/primitives/read_value.hpp>

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

template<typename T>
struct VariableParams {
    cldnn::layout layout;
    std::vector<T> values;
};

template<typename T>
struct variable_test : public ::testing::TestWithParam<VariableParams<T>> {
    void test(bool is_caching_test) {
        const VariableParams<T> param = testing::TestWithParam<VariableParams<T>>::GetParam();

        auto& engine = get_test_engine();

        const auto variable_layout = param.layout;
        const auto input_data = engine.allocate_memory(variable_layout);
        set_values(input_data, param.values);

        topology topology;
        topology.add(input_layout("input", input_data->get_layout()));
        topology.add(read_value{"read_value", { input_info("input") }, "v0", { variable_layout }});
        topology.add(eltwise{"sum", { input_info("input"), input_info("read_value") }, eltwise_mode::sum, {}, variable_layout.data_type});
        topology.add(assign{"assign", { input_info("sum") }, "v0", variable_layout});

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
        auto variable = std::make_shared<VariableState>(VariableStateInfo{"v0", variable_layout}, context, network->get_shape_predictor());
        network->set_variable("v0", variable);
        network->set_input_data("input", input_data);

        constexpr size_t number_of_inferences = 5;
        for (size_t inference = 1; inference <= number_of_inferences; ++inference) {
            const auto outputs = network->execute();
            const auto output = outputs.at("assign").get_memory();
            const cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());
            const auto output_count = output_ptr.size();
            ASSERT_EQ(output_count, param.values.size()) << "inference " << inference;

            for (size_t i = 0; i < output_count; ++i) {
                if (ov::element::Type(output->get_layout().data_type).is_real()) {
                    ASSERT_FLOAT_EQ(output_ptr[i], (inference + 1) * param.values[i]) << "inference " << inference;
                } else {
                    ASSERT_EQ(output_ptr[i], (inference + 1) * param.values[i]) << "inference " << inference;
                }
            }
        }
    }
};

using variable_test_i32 = variable_test<int32_t>;
using variable_test_i64 = variable_test<int64_t>;
using variable_test_f32 = variable_test<float>;

TEST_P(variable_test_i32, variable_i32) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

TEST_P(variable_test_i64, variable_i64) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

TEST_P(variable_test_f32, variable_f32) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

INSTANTIATE_TEST_SUITE_P(
        basic,
        variable_test_i32,
        ::testing::Values(
                VariableParams<int32_t>{ {data_types::i32, format::bfyx, tensor{1}}, {333666} },
                VariableParams<int32_t>{ {data_types::i32, format::bfyx, tensor{1, 1, 1, 3}}, {444, 555, 666} },
                VariableParams<int32_t>{ {data_types::i32, format::bfzyx, tensor{1, 2, 3, 2}},
                                {1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1} }
        )
);

INSTANTIATE_TEST_SUITE_P(
        basic,
        variable_test_i64,
        ::testing::Values(
                VariableParams<int64_t>{ {data_types::i64, format::bfyx, tensor{1}}, {333666L} },
                VariableParams<int64_t>{ {data_types::i64, format::bfyx, tensor{1, 1, 1, 3}}, {444L, 555L, 666L} },
                VariableParams<int64_t>{ {data_types::i64, format::bfzyx, tensor{1, 2, 3, 2}},
                                         {1L, 2L, 3L, 4L, 5L, 6L, 6L, 5L, 4L, 3L, 2L, 1L} }
        )
);

INSTANTIATE_TEST_SUITE_P(
        basic,
        variable_test_f32,
        ::testing::Values(
                VariableParams<float>{ {data_types::f32, format::bfyx, tensor{1}}, {333666.f} },
                VariableParams<float>{ {data_types::f32, format::bfyx, tensor{1, 1, 1, 3}}, {44.4f, 55.5f, 66.6f} },
                VariableParams<float>{ {data_types::f32, format::bfzyx, tensor{1, 2, 3, 2}},
                                         {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f} }
        )
);

template <typename T>
void test_exception_on_wrong_layout(bool is_caching_test) {
    auto& engine = get_test_engine();

    const layout variable_layout{data_types::i32, format::bfyx, tensor{1}};
    const auto input_data = engine.allocate_memory(variable_layout);
    set_values(input_data, {333666});

    auto wrong_layout = variable_layout;
    wrong_layout.data_type = data_types::f32;
    const auto wrong_input_data = engine.allocate_memory(wrong_layout);
    set_values(input_data, {333.666f});

    topology topology;
    topology.add(input_layout("input", input_data->get_layout()));
    topology.add(read_value{"read_value", { input_info("input") }, "v0", { variable_layout }});
    topology.add(input_layout("wrong_input", wrong_input_data->get_layout()));
    topology.add(assign{"assign", { input_info("wrong_input") }, "v0", wrong_layout});

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto variable = std::make_shared<VariableState>(VariableStateInfo{"v0", variable_layout}, context, network->get_shape_predictor());
    network->set_variable("v0", variable);
    network->set_input_data("input", input_data);
    network->set_input_data("wrong_input", wrong_input_data);

    bool layout_mismatch_exception = false;
    try {
        network->execute();
    } catch(std::exception& exc) {
        const std::string error = exc.what();
        layout_mismatch_exception = error.find("Layout mismatch") != std::string::npos;
    }
    ASSERT_TRUE(layout_mismatch_exception);
}

TEST(variable_test_common, exception_on_wrong_layout) {
    test_exception_on_wrong_layout<float>(false);
}

template <typename T>
void test_different_output_data_type(bool is_caching_test) {
    auto& engine = get_test_engine();

    const layout in_layout{{ 1 }, data_types::f32, format::bfyx};
    const auto input_data = engine.allocate_memory(in_layout);
    std::vector<float> inputs = { 70.0f };
    set_values(input_data, inputs);

    const layout variable_layout{{ 1 }, data_types::f16, format::bfyx};

    topology topology;
    topology.add(input_layout("input", input_data->get_layout()));
    topology.add(assign("assign", { input_info("input") }, "v0", variable_layout));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto variable = std::make_shared<VariableState>(VariableStateInfo{"v0", variable_layout}, context, network->get_shape_predictor());
    network->set_variable("v0", variable);
    network->set_input_data("input", input_data);

    const auto outputs = network->execute();
    const auto output = outputs.at("assign").get_memory();
    const cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < output_ptr.size(); ++i) {
         ASSERT_EQ(half_to_float(output_ptr[i]), inputs[i]);
    }
}

TEST(variable_test_common, different_output_data_type) {
    test_different_output_data_type<int16_t>(false);
}

template <typename T>
void test_variables_are_preserved_across_inferences(bool is_caching_test) {
    auto& engine = get_test_engine();

    const layout variable_layout{data_types::i32, format::bfyx, tensor{1}};

    const auto input_1 = engine.allocate_memory(variable_layout);
    constexpr auto value_1 = 222;
    set_values(input_1, {value_1});

    const auto input_2 = engine.allocate_memory(variable_layout);
    constexpr auto value_2 = 555;
    set_values(input_2, {value_2});

    const auto dummy1 = engine.allocate_memory(variable_layout);
    set_values(dummy1, {11});
    const auto dummy2 = engine.allocate_memory(variable_layout);
    set_values(dummy2, {22});

    topology topology;
    topology.add(input_layout("input_1", input_1->get_layout()));
    topology.add(assign{"assign_1", { input_info("input_1") }, "v1", variable_layout});

    topology.add(input_layout("input_2", input_2->get_layout()));
    topology.add(assign{"assign_2", { input_info("input_2") }, "v2", variable_layout});

    topology.add(data("dummy1", dummy1));
    topology.add(read_value{"read_value_1", { input_info("dummy1") }, "v1", { variable_layout }});
    topology.add(read_value{"read_value_2", { input_info("dummy1") }, "v2", { variable_layout }});

    topology.add(eltwise{"sum", { input_info("read_value_1"), input_info("read_value_2") }, eltwise_mode::sum, {}, variable_layout.data_type});
    topology.add(assign{"assign_result", { input_info("sum") }, "v_result", variable_layout});

    topology.add(data("dummy2", dummy2));
    topology.add(read_value{"read_result", { input_info("dummy2") }, "v_result", { variable_layout }});

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    auto context = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto variable1 = std::make_shared<VariableState>(VariableStateInfo{"v1", variable_layout}, context, network->get_shape_predictor());
    auto variable2 = std::make_shared<VariableState>(VariableStateInfo{"v2", variable_layout}, context, network->get_shape_predictor());
    auto variable3 = std::make_shared<VariableState>(VariableStateInfo{"v_result", variable_layout}, context, network->get_shape_predictor());
    network->set_variable("v1", variable1);
    network->set_variable("v2", variable2);
    network->set_variable("v_result", variable3);
    network->set_input_data("input_1", input_1);
    network->set_input_data("input_2", input_2);

    // set variables with assign on 1st inference, read with read_values on 2nd one
    network->execute();
    const auto outputs = network->execute();
    const auto output = outputs.at("read_result").get_memory();
    const cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());
    ASSERT_EQ(output_ptr[0], value_1 + value_2);
}

TEST(variable_test_common, variables_are_preserved_across_inferences) {
    test_variables_are_preserved_across_inferences<int>(false);
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_P(variable_test_i32, variable_i32_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(variable_test_i64, variable_i64_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(variable_test_f32, variable_f32_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST(variable_test_common, exception_on_wrong_layout_cached) {
    test_exception_on_wrong_layout<float>(true);
}

TEST(variable_test_common, different_output_data_type_cached) {
    test_different_output_data_type<int16_t>(true);
}
#endif
TEST(variable_test_common, variables_are_preserved_across_inferences_cached) {
    test_variables_are_preserved_across_inferences<int>(true);
}
