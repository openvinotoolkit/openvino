// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/primitives/permute.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "openvino/core/symbol.hpp"
#include "random_generator.hpp"
#include "test_utils.h"
#include "condition_inst.h"
#include "program_helpers.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/concatenation.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/pooling.hpp>
#include <intel_gpu/primitives/condition.hpp>
#include <intel_gpu/primitives/softmax.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

using namespace cldnn;
using namespace ::tests;

namespace {
template <class T>
bool is_output_equal(const cldnn::memory::ptr mem, const std::vector<T>& ref) {
    if (mem->count() != ref.size())
        return false;
    cldnn::mem_lock<T> ptr(mem, get_test_stream());
    for (size_t i = 0; i < mem->get_layout().count(); i++) {
        if (!are_equal(ptr[i], ref[i])) return false;
    }
    return true;
}


topology generate_simple_branch (bool branch_true_false, const primitive_id& id, const primitive_id& input_id, const data_types dt = data_types::f32)
{
    topology branch;
    if (branch_true_false) {
        branch.add(
            input_layout(input_id, { dt, format::bfyx,{ 1, 1, 4, 1 } }),
            pooling(id + "_when_true", input_id, cldnn::pooling_mode::max, { 1, 2 }, { 1, 2 })
        );
    } else {
        branch.add(
            input_layout(input_id, { dt, format::bfyx,{ 1, 1, 4, 1 } }),
            pooling(id + "_when_false", input_id, cldnn::pooling_mode::average, { 1, 2 }, { 1, 2 })
        );
    }
    return branch;
}

// Body returning its parameter, which is bound to outer_input_id: the program skips it, and the condition passes the
// buffer of that input through as its output.
condition::branch make_identity_branch(engine& engine,
                                       const ExecutionConfig& config,
                                       const layout& data_layout,
                                       const primitive_id& outer_input_id,
                                       const std::string& suffix) {
    condition::branch branch;
    topology body;
    body.add(input_layout("body_input" + suffix, data_layout),
             reorder("body_result" + suffix, input_info("body_input" + suffix), data_layout));
    branch.inner_program = program::build_program(engine, body, config, false, false, true);
    branch.input_map.insert({outer_input_id, "body_input" + suffix});
    branch.output_map.insert({0, "body_result" + suffix});
    return branch;
}
}  // namespace

template < typename DataType>
struct condition_data_types {
    using type = DataType;
};

template <typename ConditionDataType>
class condition_gpu_basic_test : public ::testing::Test {
public:

    using input_type = typename ConditionDataType::type;
    std::vector<input_type> convert_data(std::vector<int> in_vec) {
        const size_t vec_size = in_vec.size();
        std::vector<input_type> converted_data_vec(vec_size);
        for (size_t i = 0; i < vec_size; i++) {
            converted_data_vec[i] = (input_type)in_vec[i];
        }
        return converted_data_vec;
    }

    void run_test(bool is_caching_test = false) {
        auto& engine = get_test_engine();

        auto dat_dt = static_cast<ov::element::Type>(ov::element::from<typename ConditionDataType::type>());

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        auto input = engine.allocate_memory({ dat_dt, format::bfyx,{ 1, 1, 4, 1 } });
        auto predicate = engine.allocate_memory({ data_types::u8, format::bfyx,{ 1, 1, 1, 1 } });
        auto scale_mem = engine.allocate_memory({ dat_dt, format::bfyx,{ 1, 1, 1, 1 } });

        primitive_id input_id           = "input";
        primitive_id pred_id            = "predicate";
        primitive_id branch_input_id    = "branch_input";
        primitive_id cond_id            = "condi";
        primitive_id scale_data_id      = "scale_data";
        primitive_id output_id          = "output";

        condition::branch branch_true;
        {
            cldnn::topology branch_true_topology   = generate_simple_branch(true,  cond_id, branch_input_id, dat_dt);
            branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
            branch_true.input_map.insert({input_id, branch_input_id});
            branch_true.output_map.insert({0, "condi_when_true"});
        }
        condition::branch branch_false;
        {
            cldnn::topology branch_false_topology  = generate_simple_branch(false, cond_id, branch_input_id, dat_dt);
            branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
            branch_false.input_map.insert({input_id, branch_input_id});
            branch_false.output_map.insert({0, "condi_when_false"});
        }

        cldnn::topology topology;
        topology.add(
            input_layout(input_id, input->get_layout())
        );
        topology.add(
            input_layout(pred_id, predicate->get_layout())
        );
        topology.add(
            input_layout(scale_data_id, scale_mem->get_layout())
        );
        topology.add(
            condition(cond_id, {input_info(pred_id), input_info(input_id)}, branch_true, branch_false)
        );
        topology.add(
            eltwise(output_id, { input_info(cond_id), input_info(scale_data_id) }, eltwise_mode::prod)
        );

        network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        set_values(input, convert_data({ 1, 2, 3, 4 }));
        set_values(scale_mem, convert_data({ 10 }));
        net->set_input_data(input_id, input);
        net->set_input_data(scale_data_id, scale_mem);

        decltype(net->execute()) out;

        //WHEN TRUE
        set_values<int8_t>(predicate, { 1 });
        net->set_input_data(pred_id, predicate);
        out = net->execute();
        auto out_data_true = out.at(output_id).get_memory();
        ASSERT_TRUE(is_output_equal(out_data_true, convert_data({ 20, 40 })));

        //WHEN FALSE
        set_values<int8_t>(predicate, {0});
        net->set_input_data(pred_id, predicate);
        out = net->execute();
        auto out_data_false = out.at(output_id).get_memory();
        ASSERT_TRUE(is_output_equal(out_data_false, convert_data({ 15, 35 })));
    }
};

using test_data_types = testing::Types<condition_data_types<ov::float16>,
                                    condition_data_types<float>>;

TYPED_TEST_SUITE(condition_gpu_basic_test, test_data_types);

TYPED_TEST(condition_gpu_basic_test, simple_basic_test) {
    this->run_test();
}

TYPED_TEST(condition_gpu_basic_test, simple_basic_test_cached) {
    this->run_test(true);
}

class condition_gpu_tests: public ::testing::Test {
public:
    void test_basic_range_equal_comp(bool is_caching_test) {
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        auto input0 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });
        auto input1 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });

        auto predicate = engine.allocate_memory({ data_types::u8, format::bfyx,{ 1, 1, 1, 1 } });

        primitive_id condi_id = "condi";
        primitive_id branch_input_id = "branch_input";
        primitive_id concat_id = "concat";

        cldnn::topology topology;
        topology.add(
            input_layout("input0", input0->get_layout())
        );
        topology.add(
            input_layout("input1", input1->get_layout())
        );
        topology.add(
            input_layout("predicate", predicate->get_layout())
        );
        topology.add(
            concatenation("concat", { input_info("input0"), input_info("input1") }, 3)
        );

        condition::branch branch_true;
        {
            cldnn::topology branch_true_topology  = generate_simple_branch(true,  condi_id, branch_input_id);
            branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
            branch_true.input_map.insert({concat_id, branch_input_id});
            branch_true.output_map.insert({0, "condi_when_true"});
        }
        condition::branch branch_false;
        {
            cldnn::topology branch_false_topology = generate_simple_branch(false, condi_id, branch_input_id);
            branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
            branch_false.input_map.insert({concat_id, branch_input_id});
            branch_false.output_map.insert({0, "condi_when_false"});
        }

        topology.add(
            condition("condi", {input_info("predicate"), input_info("concat")}, branch_true, branch_false)
        );

        std::vector<float> input0_data = {
            1, 2
        };
        std::vector<float> input1_data = {
            3, 4
        };
        std::vector<uint8_t> predicate_data_true = {
            1
        };
        std::vector<float> pooling_when_true_data = {
            2, 4
        };
        std::vector<uint8_t> predicate_data_false = {
            0
        };
        std::vector<float> pooling_when_false_data = {
            1.5, 3.5
        };

        set_values(input0, input0_data);
        set_values(input1, input1_data);
        network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        net->set_input_data("input0", input0);
        net->set_input_data("input1", input1);

        decltype(net->execute()) outputs;

        //CHECK TRUE
        set_values(predicate, predicate_data_true);
        net->set_input_data("predicate", predicate);
        outputs = net->execute();

        auto out_data_true = outputs.at("condi").get_memory();
        ASSERT_TRUE(is_output_equal(out_data_true, pooling_when_true_data));

        //CHECK FALSE
        set_values(predicate, predicate_data_false);
        net->set_input_data("predicate", predicate);
        outputs = net->execute();

        auto out_data_false = outputs.at("condi").get_memory();
        ASSERT_TRUE(is_output_equal(out_data_false, pooling_when_false_data));
    }

    void test_dynamic_shapes(bool is_caching_test) {
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        const int64_t d1 = 2;
        const int64_t d2 = 4;
        layout input_lay = {{-1, d1, -1, d2}, data_types::f32, format::bfyx};

        auto predicate = engine.allocate_memory({{ 1 }, data_types::u8, format::bfyx });

        const primitive_id condition_id = "condition";
        const primitive_id condition_id_true = condition_id + "_when_true";
        const primitive_id condition_id_false = condition_id + "_when_false";
        const primitive_id branch_input_id = "branch_input";
        const primitive_id model_input = "input";
        const primitive_id predicate_input = "predicate";
        const primitive_id tranpose = "transpose";

        cldnn::topology topology;
        topology.add(input_layout(model_input, input_lay));
        topology.add(input_layout(predicate_input, predicate->get_layout()));
        topology.add(permute(tranpose, model_input, {1, 0, 2, 3}));
        const float shift = 4.f;

        auto generate_simple_branch = [&](bool branch_true_false, const primitive_id& input_id, const data_types dt) {
            auto mem = engine.allocate_memory(layout{{d1, 1, 1, d2}, dt, format::bfyx});
            {
                cldnn::mem_lock<float> l(mem, get_test_stream());
                for (size_t i = 0; i < mem->count(); i++) {
                    l.data()[i] = shift;
                }
            }

            primitive_id const_id = "const_input";
            eltwise_mode mode = branch_true_false ? eltwise_mode::sum : eltwise_mode::sub;
            auto id = branch_true_false ? condition_id_true : condition_id_false;
            cldnn::topology branch_topology(input_layout(input_id, { {d1, -1, -1, d2}, dt, format::bfyx }),
                                            data(const_id, mem),
                                            eltwise(id, {input_id, const_id}, mode)
            );
            condition::branch branch;
            branch.inner_program = program::build_program(engine, branch_topology, config, false, false, true);
            branch.input_map.insert({tranpose, branch_input_id});
            branch.output_map.insert({0, id});

            return branch;
        };

        condition::branch branch_true = generate_simple_branch(true, branch_input_id, data_types::f32);
        condition::branch branch_false = generate_simple_branch(false, branch_input_id, data_types::f32);

        topology.add(condition(condition_id, { input_info(predicate_input), tranpose }, branch_true, branch_false));

        tests::random_generator rg(GET_SUITE_NAME);
        std::vector<uint8_t> predicate_data_true = { 1 };
        std::vector<uint8_t> predicate_data_false = { 0 };

        network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        auto check_output = [](const cldnn::memory::ptr mem, const std::vector<float>& ref, ov::Shape expected_shape) {
            ASSERT_EQ(mem->get_layout().get_shape(), expected_shape);
            ASSERT_EQ(mem->count(), ref.size());
            cldnn::mem_lock<float> ptr(mem, get_test_stream());
            for (size_t i = 0; i < mem->get_layout().count(); i++) {
                ASSERT_EQ(ptr[i], ref[i]) << "i = " << i;
            }
        };

        for (size_t i = 0; i < 10; i++) {
            layout l = {{1, d1, 1 + static_cast<int64_t>(i), d2}, data_types::f32, format::bfyx};
            std::vector<float> input_data = rg.generate_random_1d<float>(l.count(), -10, 10);
            auto mem = engine.allocate_memory(l);
            std::vector<float> expected_result_when_true = input_data;
            std::vector<float> expected_result_when_false = input_data;

            for (size_t i = 0; i < input_data.size(); i++) {
                expected_result_when_true[i] += shift;
                expected_result_when_false[i] -= shift;
            }

            set_values(mem, input_data);
            set_values(predicate, predicate_data_true);
            net->set_input_data(model_input, mem);
            net->set_input_data(predicate_input, predicate);
            auto outputs = net->execute();
            check_output(outputs.at(condition_id).get_memory(), expected_result_when_true, {d1, 1, 1+i, d2});

            set_values(predicate, predicate_data_false);
            net->set_input_data(model_input, mem);
            net->set_input_data(predicate_input, predicate);
            outputs = net->execute();
            check_output(outputs.at(condition_id).get_memory(), expected_result_when_false, {d1, 1, 1+i, d2});
        }
    }

    // This case will check the layout of condition in these conditions.
    // - it re-allocated at primitive_inst::realloc_if_needed().
    // - it can be skip subgraph.
    void test_dynamic_shapes_skip_condition(bool is_caching_test) {
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        const int64_t d1 = 2;
        const int64_t d2 = 4;
        layout input_lay = {{-1, d1, -1, d2}, data_types::f32, format::bfyx};

        auto predicate = engine.allocate_memory({{ 1 }, data_types::u8, format::bfyx });

        const primitive_id condition_id = "condition";
        const primitive_id condition_id_true = condition_id + "_when_true";
        const primitive_id condition_id_false = condition_id + "_when_false";
        const primitive_id branch_input_id = "branch_input";
        const primitive_id model_input = "input";
        const primitive_id predicate_input = "predicate";
        const primitive_id tranpose = "transpose";

        cldnn::topology topology;
        topology.add(input_layout(model_input, input_lay));
        topology.add(input_layout(predicate_input, predicate->get_layout()));
        topology.add(permute(tranpose, model_input, {1, 0, 2, 3}));

        auto generate_simple_branch = [&](bool branch_true_false, const primitive_id& input_id, const data_types dt) {
            auto id = branch_true_false ? condition_id_true : condition_id_false;
            cldnn::topology branch_topology(input_layout(input_id, { {d1, -1, -1, d2}, dt, format::bfyx }),
                                            reorder(id, input_info(input_id), { {d1, -1, -1, d2}, dt, format::bfyx })
            );
            condition::branch branch;
            branch.inner_program = program::build_program(engine, branch_topology, config, false, false, true);
            branch.input_map.insert({tranpose, branch_input_id});
            branch.output_map.insert({0, id});

            return branch;
        };

        condition::branch branch_true = generate_simple_branch(true, branch_input_id, data_types::f32);
        condition::branch branch_false = generate_simple_branch(false, branch_input_id, data_types::f32);

        topology.add(condition(condition_id, { input_info(predicate_input), tranpose }, branch_true, branch_false));

        tests::random_generator rg(GET_SUITE_NAME);
        std::vector<uint8_t> predicate_data_true = { 1 };
        std::vector<uint8_t> predicate_data_false = { 0 };

        network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        for (int i = 0; i < 10; i++) {
            layout l = {{1, d1, 1 + static_cast<int64_t>(i), d2}, data_types::f32, format::bfyx};
            std::vector<float> input_data = rg.generate_random_1d<float>(l.count(), -10, 10);
            auto mem = engine.allocate_memory(l);

            set_values(mem, input_data);
            set_values(predicate, predicate_data_true);
            net->set_input_data(model_input, mem);
            net->set_input_data(predicate_input, predicate);
            auto outputs = net->execute();

            auto cond_layout = outputs.at(condition_id).get_layout();
            ASSERT_TRUE(cond_layout.get_dim(2) == (i + 1));
        }
    }

    void test_basic_stacked_ifs(bool is_caching_test) {
        /*
            <prims...>
            <if>
            <...>
            <end_if>
            <...>
            <if>
            <...>
            <end_if>
            <prims...>
        */
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
        auto predicate = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
        auto predicate2 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

        primitive_id input_id           = "input";
        primitive_id pred_id            = "predicate";
        primitive_id predicate2_id      = "predicate2";
        primitive_id branch_input_id    = "branch_input";
        primitive_id cond_id            = "condi";
        primitive_id cond2_id           = "condi2";
        primitive_id scale_data_id      = "scale_data";
        primitive_id output_id          = "output";

        topology condi_1_true = generate_simple_branch(true, cond_id, branch_input_id);
        topology condi_1_false = generate_simple_branch(false, cond_id, branch_input_id);
        topology condi_2_true;
        condi_2_true.add(
            input_layout(branch_input_id, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } }),
            activation("activ_when_true", input_info(branch_input_id), activation_func::log2)
        );
        topology condi_2_false;
        condi_2_false.add(
            input_layout(branch_input_id, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } }),
            activation("activ_when_false", input_info(branch_input_id), activation_func::relu)
        );

        condition::branch branch_condi_1_true;
        branch_condi_1_true.inner_program = program::build_program(engine, condi_1_true, config, false, false, true);
        branch_condi_1_true.input_map.insert({input_id, branch_input_id});
        branch_condi_1_true.output_map.insert({0, "condi_when_true"});

        condition::branch branch_condi_1_false;
        branch_condi_1_false.inner_program = program::build_program(engine, condi_1_false, config, false, false, true);
        branch_condi_1_false.input_map.insert({input_id, branch_input_id});
        branch_condi_1_false.output_map.insert({0, "condi_when_false"});

        condition::branch branch_condi_2_true;
        branch_condi_2_true.inner_program = program::build_program(engine, condi_2_true, config, false, false, true);
        branch_condi_2_true.input_map.insert({cond_id, branch_input_id});
        branch_condi_2_true.output_map.insert({0, "activ_when_true"});

        condition::branch branch_condi_2_false;
        branch_condi_2_false.inner_program = program::build_program(engine, condi_2_false, config, false, false, true);
        branch_condi_2_false.input_map.insert({cond_id, branch_input_id});
        branch_condi_2_false.output_map.insert({0, "activ_when_false"});

        topology topology;
        topology.add(
            input_layout(input_id, input->get_layout())
        );
        topology.add(
            input_layout(pred_id, predicate->get_layout())
        );
        topology.add(
            condition(cond_id, { input_info(pred_id), input_info(input_id) }, branch_condi_1_true, branch_condi_1_false)
        );
        topology.add(
            input_layout(predicate2_id, predicate2->get_layout())
        );
        topology.add(
            condition(cond2_id, { input_info(predicate2_id), input_info(cond_id) }, branch_condi_2_true, branch_condi_2_false)
        );

        std::vector<float> input_data = {
            1, 2, 3, 4
        };
        std::vector<uint8_t> predicate_data = {
            1
        };
        std::vector<uint8_t> predicate_2_data = {
            0
        };
        set_values(input, input_data);
        set_values(predicate, predicate_data);
        set_values(predicate2, predicate_2_data);

        network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        net->set_input_data(input_id, input);
        net->set_input_data(pred_id, predicate);
        net->set_input_data(predicate2_id, predicate2);
        auto outputs = net->execute();

        std::vector<float> ref_data = {
            2.0f, 4.0f
        };
        auto out_data = outputs.at(cond2_id).get_memory();
        ASSERT_TRUE(is_output_equal(out_data, ref_data));
    }

    void test_basic_nested_ifs(bool is_caching_test) {
        /*
        <prims...>
        <if 0>
        <...>
        <if 1>
        <...>
        <end_if 1>
        <...>
        <end_if 0>
        <prims...>
        */
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
        auto predicate = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
        auto predicate2 = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
        auto scale_5_mem = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
        set_values(scale_5_mem, { 5.0f });
        auto scale_10_mem = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
        set_values(scale_10_mem, { 10.0f });

        condition::branch nested_true;
        {
            cldnn::topology nested_true_topology;
            nested_true_topology.add(
                input_layout("branch_input1", { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } }),
                data("scale_5_data", scale_5_mem),
                eltwise("scale_5", { input_info("branch_input1"), input_info("scale_5_data") }, eltwise_mode::prod)
            );
            nested_true.inner_program = program::build_program(engine, nested_true_topology, config, false, false, true);
            nested_true.input_map.insert({"pooling_when_true", "branch_input1"});
            nested_true.output_map.insert({0, "scale_5"});
        }
        condition::branch nested_false;
        {
            cldnn::topology nested_false_topology;
            nested_false_topology.add(
                input_layout("branch_input2", { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } }),
                data("scale_10_data", scale_10_mem),
                eltwise("scale_10", { input_info("branch_input2"), input_info("scale_10_data") }, eltwise_mode::prod)
            );
            nested_false.inner_program = program::build_program(engine, nested_false_topology, config, false, false, true);
            nested_false.input_map.insert({"pooling_when_true", "branch_input2"});
            nested_false.output_map.insert({0, "scale_10"});
        }

        condition::branch branch_true;
        {
            cldnn::topology branch_true_topology;
            branch_true_topology.add(
                input_layout("branch_input3", { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
                pooling("pooling_when_true", input_info("branch_input3"), cldnn::pooling_mode::max, { 1, 2 }, { 1, 2 }),
                input_layout("predicate2", predicate2->get_layout()),
                condition( "condi_nested", {input_info("predicate2"), input_info("pooling_when_true")}, nested_true, nested_false)
            );
            branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
            branch_true.input_map.insert({"input", "branch_input3"});
            branch_true.input_map.insert({"predicate2", "predicate2"});
            branch_true.output_map.insert({0, "condi_nested"});
        }

        condition::branch branch_false;
        {
            cldnn::topology branch_false_topology;
            branch_false_topology.add(
                input_layout("branch_input4", { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
                pooling("pooling_when_false", input_info("branch_input4"), cldnn::pooling_mode::average, { 1, 2 }, { 1, 2 })
            );
            branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
            branch_false.input_map.insert({"input", "branch_input4"});
            branch_false.output_map.insert({0, "pooling_when_false"});
        }

        cldnn::topology topology;
        topology.add(
            input_layout("input", input->get_layout())
        );

        topology.add(
            input_layout("predicate", predicate->get_layout()),
            input_layout("predicate2", predicate2->get_layout())
        );

        topology.add(
            condition("condi", {input_info("predicate"), input_info("predicate2"), input_info("input")}, branch_true, branch_false)
        );

        std::vector<float> input_data = {
            1.0f, 2.0f, 3.0f, 4.0f
        };
        std::vector<float> predicate_data = {
            1.0f
        };
        std::vector<float> predicate_2_data = {
            2.0f
        };
        set_values(input, input_data);
        set_values(predicate, predicate_data);
        set_values(predicate2, predicate_2_data);

        network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        net->set_input_data("input", input);
        net->set_input_data("predicate", predicate);
        net->set_input_data("predicate2", predicate2);
        auto outputs = net->execute();

        auto out_data = outputs.at("condi").get_memory();
        ASSERT_TRUE(is_output_equal(out_data, std::vector<float>({ 10.0f, 20.0f })));
    }

    void test_negative_predicate_wrong_layout(bool is_caching_test) {
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
        auto predicate = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 5, 1 } });

        primitive_id input_id           = "input";
        primitive_id pred_id            = "predicate";
        primitive_id branch_input_id    = "branch_input";
        primitive_id cond_id            = "condi";

        condition::branch branch_true;
        {
            cldnn::topology branch_true_topology   = generate_simple_branch(true,  cond_id, branch_input_id, data_types::f32);
            branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
            branch_true.input_map.insert({input_id, branch_input_id});
            branch_true.output_map.insert({0, "condi_when_true"});
        }
        condition::branch branch_false;
        {
            cldnn::topology branch_false_topology  = generate_simple_branch(false, cond_id, branch_input_id, data_types::f32);
            branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
            branch_false.input_map.insert({input_id, branch_input_id});
            branch_false.output_map.insert({0, "condi_when_false"});
        }

        topology topology;
        topology.add(
            input_layout(input_id, input->get_layout())
        );
        topology.add(
            input_layout(pred_id, predicate->get_layout())
        );
        topology.add(
            condition(cond_id, {input_info(pred_id), input_info(input_id)}, branch_true, branch_false)
        );

        EXPECT_ANY_THROW(network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test););
    }

    void test_negative_not_same_layouts(bool is_caching_test) {
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
        auto predicate = engine.allocate_memory({ data_types::u8, format::bfyx,{ 1, 1, 1, 1 } });

        primitive_id input_id           = "input";
        primitive_id pred_id            = "predicate";
        primitive_id branch_input_id    = "branch_input";
        primitive_id cond_id            = "condi";

        condition::branch branch_true;
        {
            primitive_id pool_id = "pooling_when_true";
            topology branch_true_topology;
            branch_true_topology.add(
                input_layout(branch_input_id, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
                pooling(pool_id, input_info(branch_input_id), cldnn::pooling_mode::max, { 1, 2 }, { 1, 2 })
            );
            branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
            branch_true.input_map.insert({input_id, branch_input_id});
            branch_true.output_map.insert({0, pool_id});
        }

        condition::branch branch_false;
        {
            primitive_id pool_id = "pooling_when_false";
            topology branch_false_topology;
            branch_false_topology.add(
                input_layout(branch_input_id, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
                pooling(pool_id, input_info(branch_input_id), cldnn::pooling_mode::max, { 1, 4 }, { 1, 4 })
            );
            branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
            branch_false.input_map.insert({input_id, branch_input_id});
            branch_false.output_map.insert({0, pool_id});
        }


        topology topology;
        topology.add(
            input_layout(input_id, input->get_layout())
        );
        topology.add(
            input_layout(pred_id, predicate->get_layout())
        );
        topology.add(
            condition(cond_id, {input_info(pred_id), input_info(input_id)}, branch_true, branch_false)
        );

        EXPECT_ANY_THROW(network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test););
    }

    void test_negative_same_names_within_different_networks(bool is_caching_test) {
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
        auto predicate = engine.allocate_memory({ data_types::u8, format::bfyx,{ 1, 1, 1, 1 } });

        primitive_id input_id           = "input";
        primitive_id pred_id         = "predicate";
        primitive_id branch_input_id    = "branch_input";
        primitive_id cond_id            = "condi";
        primitive_id duplicated_id      = "pooling_check_name";

        condition::branch branch_true;
        {
            topology branch_true_topology;
            branch_true_topology.add(
                input_layout(branch_input_id, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
                pooling(duplicated_id, input_info(branch_input_id), cldnn::pooling_mode::max, { 2, 1 }, { 2, 1 })
            );
            branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
            branch_true.input_map.insert({input_id, branch_input_id});
            branch_true.output_map.insert({0, duplicated_id});
        }

        condition::branch branch_false;
        {
            topology branch_false_topology;
            branch_false_topology.add(
                input_layout(branch_input_id, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
                pooling("pooling_when_false", input_info(branch_input_id), cldnn::pooling_mode::max, { 2, 1 }, { 2, 1 })
            );
            branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
            branch_false.input_map.insert({input_id, branch_input_id});
            branch_false.output_map.insert({0, "pooling_when_false"});
        }

        topology topology;
        topology.add(
            input_layout(input_id, input->get_layout())
        );
        topology.add(
            input_layout(pred_id, predicate->get_layout())
        );
        topology.add(
            condition(cond_id, {input_info(pred_id), input_info(input_id)}, branch_true, branch_false)
        );
        topology.add(
            pooling(duplicated_id, input_info(cond_id), cldnn::pooling_mode::max, { 2, 1 }, { 2, 1 })
        );

        EXPECT_NO_THROW(network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test););
    }

    void test_empty_body(bool is_caching_test) {
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        auto input_mem = engine.allocate_memory({ data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
        auto predicate_mem = engine.allocate_memory({ data_types::u8, format::bfyx,{ 1, 1, 1, 1 } });

        primitive_id input_id1           = "input1";
        primitive_id input_id2           = "input2";
        primitive_id pred_id             = "predicate";
        primitive_id branch_input_id1    = "branch_input1";
        primitive_id branch_input_id2    = "branch_input2";
        primitive_id cond_id             = "condi";

        condition::branch branch_true;
        {
            topology branch_true_topology;
            branch_true_topology.add(
                input_layout(branch_input_id1, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
                input_layout(branch_input_id2, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } }),
                eltwise("eltwise", { input_info(branch_input_id1), input_info(branch_input_id2) }, eltwise_mode::sum)
            );
            branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
            branch_true.input_map.insert({input_id1, branch_input_id1});
            branch_true.input_map.insert({input_id2, branch_input_id2});
            branch_true.output_map.insert({0, "eltwise"});
        }

        condition::branch branch_false;
        {
            topology branch_false_topology;
            branch_false_topology.add(
                input_layout(branch_input_id2, { data_types::f32, format::bfyx, { 1, 1, 4, 1 } }),
                reorder("result", input_info(branch_input_id2), {data_types::f32, format::bfyx, {1, 1, 4, 1}})
            );
            branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
            branch_false.input_map.insert({input_id2, branch_input_id2});
            branch_false.output_map.insert({0, "result"});
        }

        topology topology;
        topology.add(input_layout(input_id1, input_mem->get_layout()));
        topology.add(input_layout(input_id2, input_mem->get_layout()));
        topology.add(input_layout(pred_id, predicate_mem->get_layout()));
        topology.add(condition(cond_id, {input_info(pred_id), input_info(input_id1), input_info(input_id2)}, branch_true, branch_false)
        );

        network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        ASSERT_TRUE(net->get_primitive(cond_id)->get_node().as<condition>().get_branch_false().inner_program->can_be_optimized());
        ASSERT_FALSE(net->get_primitive(cond_id)->get_node().as<condition>().get_branch_true().inner_program->can_be_optimized());
    }

    // A skipped (identity) body passes the condition input buffer through as the condition output. "scale" runs while
    // the output of "cond" is still to be read by "sum", so the memory pool must not give it the buffer of "abs".
    // With nested, the bodies of "cond" run and return a nested condition with identity bodies, which passes the body
    // parameter, i.e. the buffer of "abs", through in the same way.
    void test_skipped_body_keeps_input_buffer(bool is_caching_test, bool nested = false) {
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        const layout data_layout = { data_types::f32, format::bfyx, { 1, 1, 4, 4 } };
        const layout pred_layout = { data_types::u8, format::bfyx, { 1, 1, 1, 1 } };
        auto input_mem = engine.allocate_memory(data_layout);
        auto predicate_mem = engine.allocate_memory(pred_layout);

        auto identity_branch = [&](const primitive_id& outer_input_id, const std::string& suffix) {
            return make_identity_branch(engine, config, data_layout, outer_input_id, suffix);
        };
        auto nested_branch = [&](const primitive_id& outer_input_id, const std::string& suffix) {
            condition::branch branch;
            topology body;
            body.add(input_layout("body_input" + suffix, data_layout),
                     input_layout("body_predicate" + suffix, pred_layout),
                     condition("body_cond" + suffix,
                               {input_info("body_predicate" + suffix), input_info("body_input" + suffix)},
                               identity_branch("body_input" + suffix, suffix + "_t"),
                               identity_branch("body_input" + suffix, suffix + "_f")));
            branch.inner_program = program::build_program(engine, body, config, false, false, true);
            branch.input_map.insert({outer_input_id, "body_input" + suffix});
            branch.input_map.insert({"predicate", "body_predicate" + suffix});
            branch.output_map.insert({0, "body_cond" + suffix});
            return branch;
        };
        auto cond_branch = [&](const primitive_id& outer_input_id, const std::string& suffix) {
            return nested ? nested_branch(outer_input_id, suffix) : identity_branch(outer_input_id, suffix);
        };

        topology topology;
        topology.add(input_layout("input", data_layout));
        topology.add(input_layout("predicate", pred_layout));
        topology.add(activation("abs", input_info("input"), activation_func::abs));
        topology.add(condition("cond", {input_info("predicate"), input_info("abs")}, cond_branch("abs", "_t0"), cond_branch("abs", "_f0")));
        topology.add(activation("scale", input_info("cond"), activation_func::linear, {2.f, 0.f}));
        topology.add(condition("cond2", {input_info("predicate"), input_info("scale")}, identity_branch("scale", "_t1"), identity_branch("scale", "_f1")));
        topology.add(eltwise("sum", {input_info("cond"), input_info("cond2")}, eltwise_mode::sum));

        network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        ASSERT_EQ(net->get_primitive("cond")->get_node().as<condition>().get_branch_false().inner_program->can_be_optimized(), !nested);
        ASSERT_FALSE(engine.is_the_same_buffer(*net->get_primitive("scale")->output_memory_ptr(), *net->get_primitive("abs")->output_memory_ptr()));

        std::vector<float> input_data(data_layout.count());
        for (size_t i = 0; i < input_data.size(); ++i) {
            input_data[i] = static_cast<float>(i) - 8.f;
        }
        set_values(input_mem, input_data);
        set_values<uint8_t>(predicate_mem, {0});
        net->set_input_data("input", input_mem);
        net->set_input_data("predicate", predicate_mem);

        auto outputs = net->execute();
        auto output = outputs.at("sum").get_memory();
        cldnn::mem_lock<float> output_ptr(output, get_test_stream());
        for (size_t i = 0; i < input_data.size(); ++i) {
            ASSERT_FLOAT_EQ(output_ptr[i], 3.f * std::fabs(input_data[i])) << "i=" << i;
        }
    }

    // "cond" is a network output, and its skipped (identity) body makes that output the buffer of "abs". "late" runs
    // after "cond" and after "sum3", the last user of "abs", and every user of "abs" reaches it, so the graph puts it on
    // no restriction list of "abs". Like "abs" it feeds a condition, so both get the same allocation type. The memory
    // pool must still not give "late" the buffer of "abs", which holds the output of "cond" until the network has
    // finished. Without cond_has_users, "abs_x2" reads "abs" instead of "cond", so "cond" is a user of "abs" that
    // "late" does not reach, and oooq_memory_dependencies separates them as well.
    void test_skipped_body_output_keeps_input_buffer(bool is_caching_test, bool cond_has_users) {
        auto& engine = get_test_engine();
        ExecutionConfig body_config = get_test_default_config(engine);
        body_config.set_property(ov::intel_gpu::optimize_data(true));
        ExecutionConfig config = body_config;
        config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"cond", "cond3"}));
        const layout data_layout = { data_types::f32, format::bfyx, { 1, 1, 4, 4 } };
        const layout pred_layout = { data_types::u8, format::bfyx, { 1, 1, 1, 1 } };
        auto input_mem = engine.allocate_memory(data_layout);
        auto predicate_mem = engine.allocate_memory(pred_layout);
        auto identity_branch = [&](const primitive_id& outer_input_id, const std::string& suffix) {
            return make_identity_branch(engine, body_config, data_layout, outer_input_id, suffix);
        };

        // Users are connected in id order: "abs_x2" before "cond", which places "cond" right after "abs"
        const primitive_id x2_input = cond_has_users ? "cond" : "abs";
        topology topology;
        topology.add(input_layout("input", data_layout));
        topology.add(input_layout("predicate", pred_layout));
        topology.add(activation("abs", input_info("input"), activation_func::abs));
        topology.add(condition("cond", {input_info("predicate"), input_info("abs")}, identity_branch("abs", "_t0"), identity_branch("abs", "_f0")));
        topology.add(activation("abs_x2", input_info(x2_input), activation_func::linear, {2.f, 0.f}));
        topology.add(condition("cond2", {input_info("predicate"), input_info("abs_x2")}, identity_branch("abs_x2", "_t1"), identity_branch("abs_x2", "_f1")));
        topology.add(eltwise("sum3", {input_info("abs"), input_info("cond2")}, eltwise_mode::sum));
        // Not an activation, which would be fused into "sum3"; the softmax over the single feature gives ones
        topology.add(softmax("late", input_info("sum3"), 1));
        topology.add(condition("cond3", {input_info("predicate"), input_info("late")}, identity_branch("late", "_t2"), identity_branch("late", "_f2")));

        network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        ASSERT_TRUE(net->get_primitive("cond")->get_node().as<condition>().get_branch_false().inner_program->can_be_optimized());
        ASSERT_EQ(net->get_primitive("cond")->get_node().get_users().size(), cond_has_users ? 1u : 0u);
        auto& prog = *net->get_program();
        auto processing_number = [&](const primitive_id& id) {
            return prog.get_processing_order().get_processing_number(&prog.get_node(id));
        };
        ASSERT_LT(processing_number("cond"), processing_number("late"));
        EXPECT_FALSE(engine.is_the_same_buffer(*net->get_primitive("late")->output_memory_ptr(), *net->get_primitive("abs")->output_memory_ptr()));

        std::vector<float> input_data(data_layout.count());
        for (size_t i = 0; i < input_data.size(); ++i) {
            input_data[i] = static_cast<float>(i) - 8.f;
        }
        set_values(input_mem, input_data);
        set_values<uint8_t>(predicate_mem, {0});
        net->set_input_data("input", input_mem);
        net->set_input_data("predicate", predicate_mem);

        auto outputs = net->execute();
        ASSERT_TRUE(engine.is_the_same_buffer(*outputs.at("cond").get_memory(), *net->get_primitive("abs")->output_memory_ptr()));
        cldnn::mem_lock<float> cond_ptr(outputs.at("cond").get_memory(), get_test_stream());
        cldnn::mem_lock<float> cond3_ptr(outputs.at("cond3").get_memory(), get_test_stream());
        for (size_t i = 0; i < input_data.size(); ++i) {
            ASSERT_FLOAT_EQ(cond_ptr[i], std::fabs(input_data[i])) << "i=" << i;
            ASSERT_FLOAT_EQ(cond3_ptr[i], 1.f) << "i=" << i;
        }
    }

#ifdef ENABLE_ONEDNN_FOR_GPU
    // The bodies of "cond" run. In them, "fc" is a oneDNN fully connected whose fused multiply runs in place
    // (binary_mul_inplace) on the output of a nested identity condition, which passes the body parameter, i.e. the
    // buffer of "abs", through. With dynamic shapes "fc" writes into the current buffer of that multiply input, and the
    // body output reorder is optimized out, so "cond" outputs the buffer of "abs". "scale" runs while "sum" has still to
    // read the output of "cond", so the memory pool must not give it the buffer of "abs".
    void test_onednn_inplace_body_keeps_input_buffer(bool is_caching_test) {
        auto& engine = get_test_engine();
        if (!engine.get_device_info().supports_immad)
            GTEST_SKIP();

        const int64_t ifm = 32;
        const int64_t ofm = 16;
        // can_use_mul_inplace() matches dynamic dims by symbol, as SymbolicPropagation sets them in the plugin
        auto batch = ov::Dimension::dynamic();
        batch.set_symbol(std::make_shared<ov::Symbol>());
        const layout x_layout{ov::PartialShape{batch, ifm}, data_types::f16, format::bfyx};
        const layout m_layout{ov::PartialShape{batch, ofm}, data_types::f16, format::bfyx};
        const layout pred_layout{ov::PartialShape{1}, data_types::u8, format::bfyx};

        tests::random_generator rg(GET_SUITE_NAME);
        auto weights_mem = engine.allocate_memory({ov::PartialShape{ofm, ifm}, data_types::f16, format::bfyx});
        set_values(weights_mem, rg.generate_random_1d<ov::float16>(weights_mem->count(), -1, 1));

        auto make_config = [&](bool enable_memory_pool) {
            ExecutionConfig config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::optimize_data(true));
            config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
            config.set_property(ov::intel_gpu::enable_memory_pool(enable_memory_pool));
            return config;
        };
        // Each network needs its own topology, see basic_memory_dependencies_mul_inplace
        auto make_topology = [&](const ExecutionConfig& config, program::ptr& body_program) {
            ExecutionConfig body_config = config;
            body_config.set_property(ov::intel_gpu::force_implementations(
                ov::intel_gpu::ImplForcingMap{{"fc", ov::intel_gpu::ImplementationDesc{format::bfyx, "", impl_types::onednn}}}));
            auto fc_branch = [&](const std::string& suffix) {
                topology body(input_layout("body_x", x_layout),
                              input_layout("body_m", m_layout),
                              input_layout("body_predicate", pred_layout),
                              data("weights", weights_mem),
                              condition("body_cond",
                                        {input_info("body_predicate"), input_info("body_m")},
                                        make_identity_branch(engine, config, m_layout, "body_m", suffix + "_t"),
                                        make_identity_branch(engine, config, m_layout, "body_m", suffix + "_f")),
                              fully_connected("fc", input_info("body_x"), "weights", "", 2),
                              eltwise("mul", {input_info("fc"), input_info("body_cond")}, eltwise_mode::prod),
                              reorder("body_result", input_info("mul"), m_layout));
                condition::branch branch;
                branch.inner_program = program::build_program(engine, body, body_config, false, false, true);
                branch.input_map.insert({"input", "body_x"});
                branch.input_map.insert({"abs", "body_m"});
                branch.input_map.insert({"predicate", "body_predicate"});
                branch.output_map.insert({0, "body_result"});
                return branch;
            };
            auto branch_true = fc_branch("_t");
            body_program = branch_true.inner_program;
            return topology(input_layout("input", x_layout),
                            input_layout("mul_in", m_layout),
                            input_layout("predicate", pred_layout),
                            activation("abs", input_info("mul_in"), activation_func::abs),
                            condition("cond", {input_info("predicate"), input_info("input"), input_info("abs")}, branch_true, fc_branch("_f")),
                            activation("scale", input_info("cond"), activation_func::linear, {2.f, 0.f}),
                            condition("cond2",
                                      {input_info("predicate"), input_info("scale")},
                                      make_identity_branch(engine, config, m_layout, "scale", "_t1"),
                                      make_identity_branch(engine, config, m_layout, "scale", "_f1")),
                            eltwise("sum", {input_info("cond"), input_info("cond2")}, eltwise_mode::sum));
        };

        const auto config_ref = make_config(false);
        const auto config = make_config(true);
        program::ptr body_ref;
        program::ptr body;
        network net_ref(engine, make_topology(config_ref, body_ref), config_ref);
        auto topology = make_topology(config, body);
        network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

        auto& fc_node = body->get_node("fc");
        ASSERT_EQ(fc_node.get_preferred_impl_type(), impl_types::onednn);
        const auto reused_idx = onednn_eltwise_fusing_helpers::get_reused_eltwmem_idx(fc_node);
        ASSERT_NE(reused_idx, -1) << "the multiply is expected to be fused into the oneDNN FC as binary_mul_inplace";
        ASSERT_EQ(fc_node.get_dependency(reused_idx).id(), "body_cond");
        ASSERT_FALSE(body->can_be_optimized());

        const auto& abs_node = net->get_primitive("abs")->get_node();
        const auto& scale_node = net->get_primitive("scale")->get_node();
        if (!is_caching_test) {
            const auto forwarded = net->get_primitive("cond")->get_node().as<condition>().get_inputs_forwarded_to_outputs();
            EXPECT_NE(std::find(forwarded.begin(), forwarded.end(), &abs_node), forwarded.end());
        }
        EXPECT_TRUE(scale_node.has_memory_dependency(static_cast<uint32_t>(abs_node.get_unique_id())));
        EXPECT_TRUE(abs_node.has_memory_dependency(static_cast<uint32_t>(scale_node.get_unique_id())));

        auto predicate_mem = engine.allocate_memory({ov::PartialShape{1}, data_types::u8, format::bfyx});
        const std::vector<int64_t> batches = {4, 8, 2};
        for (size_t iter = 0; iter < batches.size(); ++iter) {
            const auto b = batches[iter];
            auto input_mem = engine.allocate_memory({ov::PartialShape{b, ifm}, data_types::f16, format::bfyx});
            auto mul_mem = engine.allocate_memory({ov::PartialShape{b, ofm}, data_types::f16, format::bfyx});
            set_values(input_mem, rg.generate_random_1d<ov::float16>(input_mem->count(), -1, 1));
            set_values(mul_mem, rg.generate_random_1d<ov::float16>(mul_mem->count(), -1, 1));
            set_values<uint8_t>(predicate_mem, {static_cast<uint8_t>(iter % 2)});
            for (auto* n : {&net_ref, net.get()}) {
                n->set_input_data("input", input_mem);
                n->set_input_data("mul_in", mul_mem);
                n->set_input_data("predicate", predicate_mem);
            }
            auto out_ref = net_ref.execute().at("sum").get_memory();
            auto out = net->execute().at("sum").get_memory();

            const auto& abs_mem = *net->get_primitive("abs")->output_memory_ptr();
            // A loaded program does not keep the ov::Symbols, so binary_mul_inplace only runs without caching
            if (!is_caching_test) {
                ASSERT_TRUE(engine.is_the_same_buffer(*net->get_primitive("cond")->output_memory_ptr(), abs_mem)) << "batch=" << b;
            }
            EXPECT_FALSE(engine.is_the_same_buffer(*net->get_primitive("scale")->output_memory_ptr(), abs_mem)) << "batch=" << b;
            ASSERT_EQ(out_ref->count(), out->count());
            cldnn::mem_lock<ov::float16> ref_ptr(out_ref, get_test_stream());
            cldnn::mem_lock<ov::float16> out_ptr(out, get_test_stream());
            for (size_t i = 0; i < out_ref->count(); ++i) {
                ASSERT_NEAR(static_cast<float>(ref_ptr[i]), static_cast<float>(out_ptr[i]), 1e-2f) << "batch=" << b << ", i=" << i;
            }
        }
    }
#endif  // ENABLE_ONEDNN_FOR_GPU
};

TEST_F(condition_gpu_tests, basic_range_equal_comp) {
    this->test_basic_range_equal_comp(false);
}

TEST_F(condition_gpu_tests, basic_range_equal_comp_cached) {
    this->test_basic_range_equal_comp(true);
}

TEST_F(condition_gpu_tests, dynamic_shapes) {
    this->test_dynamic_shapes(false);
}

TEST_F(condition_gpu_tests, dynamic_shapes_cached) {
    this->test_dynamic_shapes(true);
}

TEST_F(condition_gpu_tests, dynamic_shapes_skip_condition) {
    this->test_dynamic_shapes_skip_condition(false);
}

TEST_F(condition_gpu_tests, dynamic_shapes_skip_condition_cached) {
    this->test_dynamic_shapes_skip_condition(true);
}

TEST_F(condition_gpu_tests, basic_stacked_ifs) {
    this->test_basic_stacked_ifs(false);
}

TEST_F(condition_gpu_tests, basic_stacked_ifs_cached) {
    this->test_basic_stacked_ifs(true);
}

TEST_F(condition_gpu_tests, basic_nested_ifs) {
    this->test_basic_nested_ifs(false);
}

TEST_F(condition_gpu_tests, basic_nested_ifs_cached) {
    this->test_basic_nested_ifs(true);
}

TEST_F(condition_gpu_tests, negative_predicate_wrong_layout) {
    this->test_negative_predicate_wrong_layout(false);
}

TEST_F(condition_gpu_tests, negative_predicate_wrong_layout_cache) {
    this->test_negative_predicate_wrong_layout(true);
}

TEST_F(condition_gpu_tests, negative_not_same_layouts) {
    this->test_negative_not_same_layouts(false);
}

TEST_F(condition_gpu_tests, negative_not_same_layouts_cache) {
    this->test_negative_not_same_layouts(true);
}

TEST_F(condition_gpu_tests, negative_same_names_within_different_networks) {
    this->test_negative_same_names_within_different_networks(false);
}

TEST_F(condition_gpu_tests, empty_body) {
    this->test_empty_body(false);
}

TEST_F(condition_gpu_tests, empty_body_cached) {
    this->test_empty_body(true);
}

TEST_F(condition_gpu_tests, skipped_body_keeps_input_buffer) {
    this->test_skipped_body_keeps_input_buffer(false);
}

TEST_F(condition_gpu_tests, skipped_body_keeps_input_buffer_cached) {
    this->test_skipped_body_keeps_input_buffer(true);
}

TEST_F(condition_gpu_tests, nested_skipped_body_keeps_input_buffer) {
    this->test_skipped_body_keeps_input_buffer(false, true);
}

TEST_F(condition_gpu_tests, nested_skipped_body_keeps_input_buffer_cached) {
    this->test_skipped_body_keeps_input_buffer(true, true);
}

TEST_F(condition_gpu_tests, skipped_body_output_keeps_input_buffer) {
    this->test_skipped_body_output_keeps_input_buffer(false, true);
}

TEST_F(condition_gpu_tests, skipped_body_output_keeps_input_buffer_cached) {
    this->test_skipped_body_output_keeps_input_buffer(true, true);
}

TEST_F(condition_gpu_tests, skipped_body_output_without_users_keeps_input_buffer) {
    this->test_skipped_body_output_keeps_input_buffer(false, false);
}

TEST_F(condition_gpu_tests, skipped_body_output_without_users_keeps_input_buffer_cached) {
    this->test_skipped_body_output_keeps_input_buffer(true, false);
}

#ifdef ENABLE_ONEDNN_FOR_GPU
TEST_F(condition_gpu_tests, onednn_inplace_body_keeps_input_buffer) {
    this->test_onednn_inplace_body_keeps_input_buffer(false);
}

TEST_F(condition_gpu_tests, onednn_inplace_body_keeps_input_buffer_cached) {
    this->test_onednn_inplace_body_keeps_input_buffer(true);
}
#endif  // ENABLE_ONEDNN_FOR_GPU

TEST(condition_gpu, empty_body_with_different_shapes) {
    ov::PartialShape oned_pshape = ov::PartialShape{ 1 };
    ov::PartialShape const_pshape = ov::PartialShape{ };
    cldnn::layout const_layout = { const_pshape, data_types::f32, format::bfyx };
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto input_mem = engine.allocate_memory({ oned_pshape, data_types::f32, format::bfyx });
    auto predicate_mem = engine.allocate_memory({ oned_pshape, data_types::u8, format::bfyx });
    auto const_mem = engine.allocate_memory({ oned_pshape, data_types::f32, format::bfyx });

    primitive_id input_id1           = "input1";
    primitive_id input_id2           = "input2";
    primitive_id input_id3           = "input3";
    primitive_id pred_id             = "predicate";
    primitive_id branch_input_id1    = "branch_input1";
    primitive_id branch_input_id2    = "branch_input2";
    primitive_id branch_input_id3    = "branch_input3";
    primitive_id cond_id             = "condi";

    condition::branch branch_true;
    {
        topology branch_true_topology;
        branch_true_topology.add(
            input_layout(branch_input_id1, { oned_pshape, data_types::f32, format::bfyx }),
            input_layout(branch_input_id2, { oned_pshape, data_types::f32, format::bfyx }),
            eltwise("eltwise", { input_info(branch_input_id1), input_info(branch_input_id2) }, eltwise_mode::sum)
        );
        branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
        branch_true.input_map.insert({input_id1, branch_input_id1});
        branch_true.input_map.insert({input_id2, branch_input_id2});
        branch_true.output_map.insert({0, "eltwise"});
    }

    condition::branch branch_false;
    {
        topology branch_false_topology;
        branch_false_topology.add(
            input_layout(branch_input_id3, { const_pshape, data_types::f32, format::bfyx }),
            reorder("result", input_info(branch_input_id3), format::bfyx, data_types::f32)
        );
        branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
        branch_false.input_map.insert({input_id3, branch_input_id3});
        branch_false.output_map.insert({0, "result"});
    }

    topology topology;
    topology.add(input_layout(input_id1, input_mem->get_layout()));
    topology.add(input_layout(input_id2, input_mem->get_layout()));
    topology.add(input_layout(input_id3, const_layout));
    topology.add(input_layout(pred_id, predicate_mem->get_layout()));
    topology.add(condition(cond_id, {input_info(pred_id), input_info(input_id1), input_info(input_id2), input_info(input_id3)}, branch_true, branch_false)
    );

    network net(engine, topology, config);
    ASSERT_FALSE(net.get_primitive(cond_id)->get_node().as<condition>().get_branch_true().inner_program->can_be_optimized());
    ASSERT_TRUE(net.get_primitive(cond_id)->get_node().as<condition>().get_branch_false().inner_program->can_be_optimized());

    set_values<int8_t>(predicate_mem, { 0 });
    net.set_input_data(pred_id, predicate_mem);
    set_values(input_mem, { 1 });
    net.set_input_data(input_id1, input_mem);
    net.set_input_data(input_id2, input_mem);
    set_values(const_mem, { 1 });
    auto const_zero_mem = engine.reinterpret_buffer(*const_mem, const_layout);
    net.set_input_data(input_id3, const_zero_mem);

    auto outputs = net.execute();
    ASSERT_EQ(outputs.size(), 1);
    auto output = outputs.begin()->second.get_memory();
    auto output_layout = output->get_layout();
    auto out_pshape = output_layout.get_partial_shape();
    ASSERT_EQ(out_pshape, oned_pshape);
}

TEST(condition_gpu, set_empty_tensor) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto empty_mem = engine.allocate_memory({ { 1, 1, 1, 1 }, data_types::f16, format::bfyx });
    auto empty_input_mem = engine.reinterpret_buffer(*empty_mem, { { 1, 1, 0, 1 }, data_types::f16, format::bfyx });
    auto input_mem = engine.allocate_memory({ { 1, 1, 4, 1 }, data_types::f32, format::bfyx });
    auto predicate_mem = engine.allocate_memory({ { 1, 1, 1, 1 }, data_types::u8, format::bfyx });
    auto concat_data = engine.allocate_memory({ { 1, 1, 4, 1 }, data_types::f32, format::bfyx });

    set_values<int8_t>(predicate_mem, {1});

    primitive_id empty_input_id      = "input1";
    primitive_id reorder_id          = "reorder";
    primitive_id input_id            = "input2";
    primitive_id pred_id             = "predicate";
    primitive_id branch_input_id1    = "branch_input1";
    primitive_id branch_input_id2    = "branch_input2";
    primitive_id concat_data_id      = "concat_data";
    primitive_id concat_id           = "concat";
    primitive_id cond_id             = "condi";

    condition::branch branch_true;
    {
        topology branch_true_topology;
        branch_true_topology.add(
            input_layout(branch_input_id1, { { 1, 1, -1, 1 }, data_types::f32, format::bfyx }),
            data(concat_data_id, concat_data),
            concatenation(concat_id, { input_info(branch_input_id1), input_info(concat_data_id) }, 2)
        );
        branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
        branch_true.input_map.insert({reorder_id, branch_input_id1});
        branch_true.output_map.insert({0, concat_id});
    }

    condition::branch branch_false;
    {
        topology branch_false_topology;
        branch_false_topology.add(
            input_layout(branch_input_id2, { { 1, 1, 4, 1 }, data_types::f32, format::bfyx }),
            reorder("result", input_info(branch_input_id2), format::bfyx, data_types::f32)
        );
        branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
        branch_false.input_map.insert({input_id, branch_input_id2});
        branch_false.output_map.insert({0, "result"});
    }

    auto empty_input_layout = layout({ 1, 1, -1, 1 }, data_types::f32, format::bfyx);

    topology topology;
    topology.add(input_layout(pred_id, predicate_mem->get_layout()));
    topology.add(input_layout(empty_input_id, empty_input_layout));
    topology.add(input_layout(input_id, input_mem->get_layout()));
    topology.add(reorder(reorder_id, input_info(empty_input_id), format::bfyx, data_types::f32));
    topology.add(condition(cond_id, {input_info(pred_id), input_info(reorder_id), input_info(input_id)}, branch_true, branch_false));

    network net(engine, topology, config);
    ASSERT_TRUE(net.get_primitive(cond_id)->get_node().as<condition>().get_branch_false().inner_program->can_be_optimized());
    ASSERT_FALSE(net.get_primitive(cond_id)->get_node().as<condition>().get_branch_true().inner_program->can_be_optimized());

    net.set_input_data(pred_id, predicate_mem);
    net.set_input_data(empty_input_id, empty_input_mem);
    net.set_input_data(input_id, input_mem);

    std::map<primitive_id, network_output> outputs;
    OV_ASSERT_NO_THROW(outputs = net.execute());
    OV_ASSERT_NO_THROW(outputs.at(cond_id).get_memory());
}

TEST(condition_gpu, skip_subgraph_multiple_outputs_and_swapped_ports) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    const ov::PartialShape in_pshape = {1, 4, 1};
    const ov::PartialShape out_concat_pshape = {1, 4, 2};
    auto in0_mem = engine.allocate_memory({in_pshape, data_types::f32, format::bfyx});
    auto in1_mem = engine.allocate_memory({in_pshape, data_types::f32, format::bfyx});
    auto pred_mem = engine.allocate_memory({{1}, data_types::u8, format::bfyx});

    set_values<float>(in0_mem, {1.f, 2.f, 3.f, 4.f});
    set_values<float>(in1_mem, {10.f, 20.f, 30.f, 40.f});

    condition::branch branch_true;
    {
        topology branch_true_topology;
        branch_true_topology.add(
            input_layout("true_in0", {in_pshape, data_types::f32, format::bfyx}),
            input_layout("true_in1", {in_pshape, data_types::f32, format::bfyx}),
            reorder("true_out0", input_info("true_in0"), format::bfyx, data_types::f32),
            reorder("true_out1", input_info("true_in1"), format::bfyx, data_types::f32)
        );
        branch_true.inner_program = program::build_program(engine, branch_true_topology, config, false, false, true);
        branch_true.input_map.insert({"in0", "true_in0"});
        branch_true.input_map.insert({"in1", "true_in1"});
        branch_true.output_map.insert({0, "true_out0"});
        branch_true.output_map.insert({1, "true_out1"});
    }

    condition::branch branch_false;
    {
        topology branch_false_topology;
        branch_false_topology.add(
            input_layout("false_in0", {in_pshape, data_types::f32, format::bfyx}),
            input_layout("false_in1", {in_pshape, data_types::f32, format::bfyx}),
            reorder("z_false_out_for_port0", input_info("false_in1"), format::bfyx, data_types::f32),
            reorder("a_false_out_for_port1", input_info("false_in0"), format::bfyx, data_types::f32)
        );
        branch_false.inner_program = program::build_program(engine, branch_false_topology, config, false, false, true);
        branch_false.input_map.insert({"in0", "false_in0"});
        branch_false.input_map.insert({"in1", "false_in1"});
        branch_false.output_map.insert({0, "z_false_out_for_port0"});
        branch_false.output_map.insert({1, "a_false_out_for_port1"});
    }

    topology topo;
    topo.add(
        input_layout("pred", pred_mem->get_layout()),
        input_layout("in0", in0_mem->get_layout()),
        input_layout("in1", in1_mem->get_layout()),
        condition("cond", {input_info("pred"), input_info("in0"), input_info("in1")}, branch_true, branch_false, 2),
        concatenation("concat", {input_info("cond", 0), input_info("cond", 1)}, 2),
        reorder("out", input_info("concat"), {out_concat_pshape, data_types::f32, format::bfyx})
    );

    network net(engine, topo, config);
    ASSERT_TRUE(net.get_primitive("cond")->get_node().as<condition>().get_branch_true().inner_program->can_be_optimized());
    ASSERT_TRUE(net.get_primitive("cond")->get_node().as<condition>().get_branch_false().inner_program->can_be_optimized());

    net.set_input_data("in0", in0_mem);
    net.set_input_data("in1", in1_mem);

    set_values<uint8_t>(pred_mem, {1});
    net.set_input_data("pred", pred_mem);
    auto outputs_true = net.execute();
    ASSERT_TRUE(is_output_equal(outputs_true.at("out").get_memory(), std::vector<float>{1.f, 10.f, 2.f, 20.f, 3.f, 30.f, 4.f, 40.f}));

    set_values<uint8_t>(pred_mem, {0});
    net.set_input_data("pred", pred_mem);
    auto outputs_false = net.execute();
    ASSERT_TRUE(is_output_equal(outputs_false.at("out").get_memory(), std::vector<float>{10.f, 1.f, 20.f, 2.f, 30.f, 3.f, 40.f, 4.f}));
}

TEST(condition_gpu, executed_multiple_outputs_distinct_layouts_ordered_by_port) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    const layout in0_layout{ov::PartialShape{1, 4, 1}, data_types::f32, format::bfyx};
    const layout in1_layout{ov::PartialShape{1, 4, 2}, data_types::f32, format::bfyx};
    auto in0_mem = engine.allocate_memory(in0_layout);
    auto in1_mem = engine.allocate_memory(in1_layout);
    auto pred_mem = engine.allocate_memory({{1}, data_types::u8, format::bfyx});

    const std::vector<float> in0_data{1.f, 2.f, 3.f, 4.f};
    const std::vector<float> in1_data{10.f, 20.f, 30.f, 40.f, 50.f, 60.f, 70.f, 80.f};
    set_values(in0_mem, in0_data);
    set_values(in1_mem, in1_data);

    // Inner output IDs sort opposite to the port order in output_map.
    auto build_branch = [&](const std::string& prefix, float scale, float shift) {
        topology branch_topo(
            input_layout(prefix + "_in0", in0_layout),
            input_layout(prefix + "_in1", in1_layout),
            activation("z_" + prefix + "_port0", input_info(prefix + "_in0"), activation_func::linear, {scale, shift}),
            activation("a_" + prefix + "_port1", input_info(prefix + "_in1"), activation_func::linear, {scale, shift})
        );
        condition::branch b;
        b.inner_program = program::build_program(engine, branch_topo, config, false, false, true);
        b.input_map.insert({"in0", prefix + "_in0"});
        b.input_map.insert({"in1", prefix + "_in1"});
        b.output_map.insert({0, "z_" + prefix + "_port0"});
        b.output_map.insert({1, "a_" + prefix + "_port1"});
        return b;
    };
    auto branch_true = build_branch("t", 1.f, 1.f);
    auto branch_false = build_branch("f", 2.f, 0.f);

    topology topo(
        input_layout("pred", pred_mem->get_layout()),
        input_layout("in0", in0_layout),
        input_layout("in1", in1_layout),
        condition("cond", {input_info("pred"), input_info("in0"), input_info("in1")}, branch_true, branch_false, 2),
        reorder("out0", input_info("cond", 0), format::bfyx, data_types::f32),
        reorder("out1", input_info("cond", 1), format::bfyx, data_types::f32)
    );

    network net(engine, topo, config);
    ASSERT_FALSE(branch_true.inner_program->can_be_optimized());
    ASSERT_FALSE(branch_false.inner_program->can_be_optimized());
    net.set_input_data("in0", in0_mem);
    net.set_input_data("in1", in1_mem);

    for (uint8_t pred : {uint8_t{1}, uint8_t{0}}) {
        const float scale = pred ? 1.f : 2.f;
        const float shift = pred ? 1.f : 0.f;
        set_values<uint8_t>(pred_mem, {pred});
        net.set_input_data("pred", pred_mem);
        auto outputs = net.execute();

        auto out0 = outputs.at("out0").get_memory();
        auto out1 = outputs.at("out1").get_memory();
        ASSERT_EQ(out0->get_layout().get_partial_shape(), in0_layout.get_partial_shape()) << "pred=" << int(pred);
        ASSERT_EQ(out1->get_layout().get_partial_shape(), in1_layout.get_partial_shape()) << "pred=" << int(pred);
        cldnn::mem_lock<float, mem_lock_type::read> out0_ptr(out0, get_test_stream());
        cldnn::mem_lock<float, mem_lock_type::read> out1_ptr(out1, get_test_stream());
        for (size_t i = 0; i < in0_data.size(); ++i)
            ASSERT_FLOAT_EQ(out0_ptr[i], in0_data[i] * scale + shift) << "pred=" << int(pred) << " i=" << i;
        for (size_t i = 0; i < in1_data.size(); ++i)
            ASSERT_FLOAT_EQ(out1_ptr[i], in1_data[i] * scale + shift) << "pred=" << int(pred) << " i=" << i;
    }
}

TEST(condition_gpu, executed_branch_switch_rebinds_static_consumer) {
    auto& engine = get_test_engine();
    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    const layout in_layout{ov::PartialShape{1, 4, 1}, data_types::f32, format::bfyx};
    auto in_mem = engine.allocate_memory(in_layout);
    auto pred_mem = engine.allocate_memory({{1}, data_types::u8, format::bfyx});
    const std::vector<float> in_data{1.f, 2.f, 3.f, 4.f};
    set_values(in_mem, in_data);

    auto build_branch = [&](const std::string& prefix, float scale, float shift) {
        topology branch_topo(
            input_layout(prefix + "_in", in_layout),
            activation(prefix + "_out", input_info(prefix + "_in"), activation_func::linear, {scale, shift})
        );
        condition::branch b;
        b.inner_program = program::build_program(engine, branch_topo, config, false, false, true);
        b.input_map.insert({"input", prefix + "_in"});
        b.output_map.insert({0, prefix + "_out"});
        return b;
    };
    auto branch_true = build_branch("t", 1.f, 1.f);
    auto branch_false = build_branch("f", 2.f, 0.f);

    topology topo(
        input_layout("pred", pred_mem->get_layout()),
        input_layout("input", in_layout),
        condition("cond", {input_info("pred"), input_info("input")}, branch_true, branch_false),
        activation("mid", input_info("cond"), activation_func::linear, {1.f, 100.f}),
        // Second user keeps "mid" a separate non-output primitive.
        reorder("out", input_info("mid"), format::bfyx, data_types::f32),
        reorder("out_f16", input_info("mid"), format::bfyx, data_types::f16)
    );

    network net(engine, topo, config);
    ASSERT_FALSE(net.get_primitive("cond")->is_dynamic());
    ASSERT_FALSE(net.get_primitive("mid")->is_output());
    ASSERT_FALSE(branch_true.inner_program->can_be_optimized());
    ASSERT_FALSE(branch_false.inner_program->can_be_optimized());
    net.set_input_data("input", in_mem);

    for (uint8_t pred : {uint8_t{1}, uint8_t{0}, uint8_t{1}, uint8_t{0}}) {
        const float scale = pred ? 1.f : 2.f;
        const float shift = pred ? 1.f : 0.f;
        set_values<uint8_t>(pred_mem, {pred});
        net.set_input_data("pred", pred_mem);
        auto outputs = net.execute();

        auto out = outputs.at("out").get_memory();
        cldnn::mem_lock<float, mem_lock_type::read> out_ptr(out, get_test_stream());
        for (size_t i = 0; i < in_data.size(); ++i)
            ASSERT_FLOAT_EQ(out_ptr[i], in_data[i] * scale + shift + 100.f) << "pred=" << int(pred) << " i=" << i;
    }
}

namespace {
// The branch programs of condition are built before the outer program, so their parameters are in the default format
// without padding, and condition passes its input buffers to them as they are. These tests feed condition from producers
// which select another layout for their output.
enum class condition_input_producer {
    reorder,         // reorder to the producer format
    reorder_shared,  // reorder to the producer format which is also used by another primitive
    reorder_twice,   // reorder to the producer format which is passed to condition twice
    convolution,     // convolution with output in the producer format
    crop,            // crop optimized in place, whose output is a padded view of its input
};

void test_condition_input_layout(condition_input_producer producer,
                                 const ov::PartialShape& shape,
                                 bool is_caching_test,
                                 format::type producer_format = format::b_fs_yx_fsv16) {
    auto& engine = get_test_engine();
    ExecutionConfig branch_config = get_test_default_config(engine);
    branch_config.set_property(ov::intel_gpu::optimize_data(true));
    ExecutionConfig config = branch_config;

    const primitive_id producer_id = "producer";
    if (producer == condition_input_producer::reorder_shared) {
        config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"condi", "other"}));
    } else if (producer == condition_input_producer::convolution) {
        ov::intel_gpu::ImplementationDesc conv_impl = {producer_format, ""};
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{producer_id, conv_impl}}));
    }

    // crop takes the second half of the features of its input
    auto input_shape = shape;
    if (producer == condition_input_producer::crop) {
        input_shape[1] *= 2;
    }
    const layout in_layout{input_shape, data_types::f16, format::bfyx};
    const layout branch_in_layout{shape, data_types::f16, format::bfyx};
    auto input_mem = engine.allocate_memory(in_layout);
    auto predicate_mem = engine.allocate_memory({ov::PartialShape{1}, data_types::u8, format::bfyx});

    std::vector<ov::float16> input_data(in_layout.count());
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = ov::float16(static_cast<float>(static_cast<int>(i % 9) - 4));
    }
    set_values(input_mem, input_data);

    auto create_branch = [&](const primitive_id& output_id, float a, float b) {
        condition::branch branch;
        cldnn::topology branch_topology(input_layout("branch_input", branch_in_layout),
                                        activation(output_id, input_info("branch_input"), activation_func::linear, {a, b}));
        branch.inner_program = program::build_program(engine, branch_topology, branch_config, false, false, true);
        branch.input_map.insert({producer_id, "branch_input"});
        branch.output_map.insert({0, output_id});
        return branch;
    };
    auto branch_true = create_branch("when_true", 2.f, 0.f);    // x * 2
    auto branch_false = create_branch("when_false", 1.f, 1.f);  // x + 1

    const size_t batch = shape[0].get_length();
    const size_t features = shape[1].get_length();
    const size_t spatial = ov::shape_size(shape.get_shape()) / (batch * features);
    const size_t feature_offset = input_shape[1].get_length() - features;

    cldnn::topology topology(input_layout("input", in_layout), input_layout("predicate", predicate_mem->get_layout()));
    if (producer == condition_input_producer::crop) {
        const auto crop_size = branch_in_layout.get_tensor();
        topology.add(crop(producer_id, input_info("input"), crop_size, tensor(0, crop_size.feature[0], 0, 0)));
    } else if (producer == condition_input_producer::convolution) {
        // 1x1 convolution with identity weights
        const auto weights_dim = static_cast<int64_t>(features);
        auto weights_mem = engine.allocate_memory({ov::PartialShape{weights_dim, weights_dim, 1, 1}, data_types::f16, format::bfyx});
        std::vector<ov::float16> weights_data(features * features, ov::float16(0.f));
        for (size_t f = 0; f < features; ++f) {
            weights_data[f * features + f] = ov::float16(1.f);
        }
        set_values(weights_mem, weights_data);
        topology.add(data("weights", weights_mem));
        topology.add(convolution(producer_id, input_info("input"), "weights", "", 1, {1, 1}, {1, 1}, {0, 0}, {0, 0}, false));
    } else {
        topology.add(reorder(producer_id, input_info("input"), producer_format, data_types::f16));
    }
    if (producer == condition_input_producer::reorder_shared) {
        topology.add(activation("other", input_info(producer_id), activation_func::relu));
    }
    std::vector<input_info> cond_inputs = {input_info("predicate"), input_info(producer_id)};
    if (producer == condition_input_producer::reorder_twice) {
        cond_inputs.push_back(input_info(producer_id));
    }
    topology.add(condition("condi", cond_inputs, branch_true, branch_false));

    network::ptr net = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);

    // condition passes the buffers of the inputs in the input map to the branch parameter as they are, so they must be in its
    // layout. An input passed twice is in the map only once.
    const auto cond = net->get_primitive("condi");
    const auto input_map = cond->get_node().as<condition>().get_branch_true().input_map;
    const auto& cond_deps = cond->dependencies();
    ASSERT_EQ(cond_deps.size(), cond_inputs.size());
    size_t mapped_inputs = 0;
    for (const auto& [dep, port] : cond_deps) {
        if (input_map.count(dep->id()) != 0) {
            ++mapped_inputs;
            EXPECT_EQ(dep->get_output_layout(port).to_short_string(), branch_in_layout.to_short_string());
        }
    }
    EXPECT_EQ(mapped_inputs, 1u);

    for (const uint8_t pred : {1, 0}) {
        set_values<uint8_t>(predicate_mem, {pred});
        net->set_input_data("input", input_mem);
        net->set_input_data("predicate", predicate_mem);
        auto outputs = net->execute();

        auto output = outputs.at("condi").get_memory();
        ASSERT_EQ(output->get_layout().get_partial_shape(), shape);
        cldnn::mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
        for (size_t b = 0; b < batch; ++b) {
            for (size_t f = 0; f < features; ++f) {
                for (size_t s = 0; s < spatial; ++s) {
                    const float x = input_data[((b * input_shape[1].get_length()) + feature_offset + f) * spatial + s];
                    const float expected = pred ? x * 2.f : x + 1.f;
                    ASSERT_EQ(static_cast<float>(output_ptr[(b * features + f) * spatial + s]), expected)
                        << "pred=" << static_cast<int>(pred) << " b=" << b << " f=" << f << " s=" << s;
                }
            }
        }
    }
}
}  // namespace

TEST(condition_gpu, input_in_blocked_format) {
    test_condition_input_layout(condition_input_producer::reorder, {1, 20, 4, 4}, false);
}

TEST(condition_gpu, input_in_blocked_format_cached) {
    test_condition_input_layout(condition_input_producer::reorder, {1, 20, 4, 4}, true);
}

TEST(condition_gpu, input_in_blocked_format_fsv32) {
    test_condition_input_layout(condition_input_producer::reorder, {1, 24, 4, 4}, false, format::b_fs_yx_fsv32);
}

TEST(condition_gpu, input_in_byxf_format) {
    test_condition_input_layout(condition_input_producer::reorder, {1, 20, 4, 4}, false, format::byxf);
}

TEST(condition_gpu, input_in_blocked_format_shared_with_other_user) {
    test_condition_input_layout(condition_input_producer::reorder_shared, {2, 20, 4, 4}, false);
}

TEST(condition_gpu, input_in_blocked_format_passed_twice) {
    test_condition_input_layout(condition_input_producer::reorder_twice, {1, 20, 4, 4}, false);
}

TEST(condition_gpu, input_from_convolution_in_blocked_format) {
    test_condition_input_layout(condition_input_producer::convolution, {1, 20, 4, 4}, false);
}

// The reorder from b_fs_yx_fsv16 to bfyx can be replaced by a reinterpretation of the buffer when the spatial size is 1x1,
// which adds feature padding when the feature size is not a multiple of 16.
TEST(condition_gpu, input_from_convolution_in_blocked_format_spatial_1x1) {
    test_condition_input_layout(condition_input_producer::convolution, {1, 20, 1, 1}, false);
}

TEST(condition_gpu, input_from_in_place_crop) {
    test_condition_input_layout(condition_input_producer::crop, {1, 16, 4, 4}, false);
}
