// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <test_utils.h>

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/range.hpp>
#include <intel_gpu/primitives/select.hpp>
#include <intel_gpu/primitives/data.hpp>
#include "range_inst.h"

using namespace ::tests;
using namespace testing;

namespace cldnn {
namespace {

struct RangeArg {
    primitive_id name;
    memory::ptr p;
    RangeArg(data_types dt, const char name[]) :
        name { name }, p { tests::get_test_engine().allocate_memory( { ov::PartialShape{}, dt, format::bfyx}) } {
    }
    void addTo(topology &t) const {
        t.add(input_layout { name, p->get_layout() });
    }
    void setData(network &n) const {
        n.set_input_data(name, p);
    }
};

struct RangeArgs {
    data_types dt;
    RangeArg start { dt, "start" };
    RangeArg stop { dt, "stop" };
    RangeArg step { dt, "step" };
    explicit RangeArgs(data_types dt) : dt { dt } {}

    memory::ptr run(int outLen, bool use_new_shape_infer) const {
        topology topology;
        start.addTo(topology);
        stop.addTo(topology);
        step.addTo(topology);
        topology.add(range { "range", { input_info(start.name), input_info(stop.name), input_info(step.name) }, { dt, format::bfyx, tensor{batch(outLen)} } });

        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(use_new_shape_infer));

        network network { engine, topology, config };

        start.setData(network);
        stop.setData(network);
        step.setData(network);

        auto outputs = network.execute();
        return outputs.at("range").get_memory();
    }
};

struct range_test_params {
    data_types d_types;
    double start;
    double stop;
    double step;
    bool use_new_shape_infer;
};

std::ostream& operator<<(std::ostream& ost, const range_test_params& params) {
    ost << ov::element::Type(params.d_types) << ",";
    ost << "{start:" << params.start << ",stop:" << params.stop << ",step:" << params.step << "},";
    ost << " use_new_shape_infer(" << (params.use_new_shape_infer?"True":"False") << ")";
    return ost;
}

template<typename T>
void doSmokeRange(range_test_params& params) {

    RangeArgs args(params.d_types);

    T start_val = static_cast<T>(params.start);
    T stop_val = static_cast<T>(params.stop);
    T step_val = static_cast<T>(params.step);

    tests::set_values(args.start.p, { start_val });
    tests::set_values(args.stop.p, { stop_val });
    tests::set_values(args.step.p, { step_val });

    T outLen = (stop_val - start_val) / step_val;

    auto output = args.run(outLen, params.use_new_shape_infer);
    mem_lock<T> output_ptr { output, tests::get_test_stream() };

    for (std::size_t i = 0; i < static_cast<size_t>(outLen); ++i) {
        ASSERT_EQ(start_val + i * step_val, output_ptr[i]);
    }
}

void doSmokeRange_fp16(range_test_params& params) {

    RangeArgs args(params.d_types);

    auto start_val = static_cast<float>(params.start);
    auto stop_val = static_cast<float>(params.stop);
    auto step_val = static_cast<float>(params.step);

    tests::set_values(args.start.p, { ov::float16(start_val).to_bits() });
    tests::set_values(args.stop.p, { ov::float16(stop_val).to_bits() });
    tests::set_values(args.step.p, { ov::float16(step_val).to_bits() });

    auto outLen = (stop_val - start_val) / step_val;

    auto output = args.run(outLen, params.use_new_shape_infer);
    mem_lock<uint16_t> output_ptr { output, tests::get_test_stream() };

    for (std::size_t i = 0; i < static_cast<size_t>(outLen); ++i) {
        ASSERT_EQ(start_val + i * step_val, half_to_float(output_ptr[i]));
    }
}



struct smoke_range_test : testing::TestWithParam<range_test_params> {};
TEST_P(smoke_range_test, basic) {
    auto params = GetParam();

    switch(params.d_types) {
        case data_types::f32:
            doSmokeRange<float>(params);
            break;
        case data_types::i32:
            doSmokeRange<int>(params);
            break;
        case data_types::i8:
            doSmokeRange<std::int8_t>(params);
            break;
        case data_types::u8:
            doSmokeRange<std::uint8_t>(params);
            break;
        case data_types::i64:
            doSmokeRange<std::int64_t>(params);
            break;
        case data_types::f16:
            doSmokeRange_fp16(params);
        default:
            break;
    }

}

struct range_test_param_generator : std::vector<range_test_params> {
    range_test_param_generator& add(range_test_params params) {
        push_back(params);
        return *this;
    }

    range_test_param_generator& simple_params(std::vector<data_types>& data_types_list, double start, double stop, double step) {
        std::vector<bool> flags_use_new_si = {true, false};
        for (auto use_new_si : flags_use_new_si) {
            for (auto type : data_types_list) {
                push_back(range_test_params{ type, start, stop, step, use_new_si});
            }
        }
        return *this;
    }
};

std::vector<data_types> signed_types  = {data_types::i8};
std::vector<data_types> general_types = {data_types::u8, data_types::i32, data_types::i32, data_types::f16, data_types::f32};
std::vector<data_types> float_types   = {data_types::f16, data_types::f32};

INSTANTIATE_TEST_SUITE_P(range_gpu_test,
                        smoke_range_test,
                        testing::ValuesIn(
                            range_test_param_generator()
                            .simple_params(general_types,   2,  23,   3)
                            .simple_params(general_types,   1,  21,   2)
                            .simple_params(float_types,     1,  2.5f, 0.5f)
                            .simple_params(signed_types,    23, 2,    -3)
                            .simple_params(signed_types,    4,  0,    -1)
                        ));

TEST(range_gpu_test, range_with_select) {
    auto& engine = get_test_engine();

    int32_t start_val = 0;
    int32_t step_val = 1;
    int32_t expected_dim = 25;

    auto select_input1 = engine.allocate_memory({ {1}, data_types::u8,  format::bfyx });
    auto select_input2 = engine.allocate_memory({ { }, data_types::i32, format::bfyx });
    auto select_mask   = engine.allocate_memory({ {1}, data_types::i32, format::bfyx });
    auto input0        = engine.allocate_memory({ { }, data_types::i32, format::bfyx });
    auto input2        = engine.allocate_memory({ { }, data_types::i32, format::bfyx });

    topology topology;
    topology.add(data("select_input1", select_input1));
    topology.add(data("select_input2", select_input2));
    topology.add(data("select_mask",   select_mask));
    topology.add(data("input0", input0));
    topology.add(data("input2", input2));
    topology.add(cldnn::select("select", input_info("select_input1"), input_info("select_input2"), input_info("select_mask")));
    topology.add(range{ "range", { input_info("input0"), input_info("select"), input_info("input2") }, data_types::i32 });

    set_values<uint8_t>(select_input1, {0});
    set_values<int32_t>(select_input2, {384});
    set_values<int32_t>(select_mask, {expected_dim});
    set_values<int32_t>(input0, {start_val});
    set_values<int32_t>(input2, {step_val});

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network { tests::get_test_engine(), topology, config };

    auto outputs = network.execute();
    auto output = outputs.at("range").get_memory();

    mem_lock<int32_t> output_ptr { output, tests::get_test_stream() };

    for (size_t i = 0; i < static_cast<size_t>(expected_dim); ++i) {
        ASSERT_EQ(start_val + i * step_val, output_ptr[i]);
    }
}

TEST(range_gpu_test, constant_folding) {
    auto& engine = get_test_engine();

    int32_t start_val = 0;
    int32_t step_val = 1;
    int32_t expected_dim = 25;

    auto input0 = engine.allocate_memory({ ov::PartialShape::dynamic(0), data_types::i32, format::bfyx });
    auto input1 = engine.allocate_memory({ ov::PartialShape::dynamic(0), data_types::i32, format::bfyx });
    auto input2 = engine.allocate_memory({ ov::PartialShape::dynamic(0), data_types::i32, format::bfyx });

    set_values<int32_t>(input0, { start_val });
    set_values<int32_t>(input1, { expected_dim });
    set_values<int32_t>(input2, { step_val });

    topology topology;
    topology.add(data("input0", input0));
    topology.add(data("input1", input1));
    topology.add(data("input2", input2));
    topology.add(range{ "range", { input_info("input0"), input_info("input1"), input_info("input2") }, data_types::i32});

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    auto outputs = network.execute();
    auto output = outputs.at("range").get_memory();

    mem_lock<int32_t> output_ptr(output, tests::get_test_stream());

    for (size_t i = 0; i < static_cast<size_t>(expected_dim); ++i) {
        ASSERT_EQ(start_val + i * step_val, output_ptr[i]);
    }
}

TEST(range_gpu_test, dynamic_all) {
    auto& engine = get_test_engine();

    int32_t start_val = 0;
    int32_t step_val = 1;
    int32_t expected_dim = 25;

    auto dynamic_input_layout = layout{ ov::PartialShape::dynamic(0), data_types::i32, format::bfyx };

    auto input0 = engine.allocate_memory({ {}, data_types::i32, format::bfyx });
    auto input1 = engine.allocate_memory({ {}, data_types::i32, format::bfyx });
    auto input2 = engine.allocate_memory({ {}, data_types::i32, format::bfyx });

    set_values<int32_t>(input0, { start_val });
    set_values<int32_t>(input1, { expected_dim });
    set_values<int32_t>(input2, { step_val });

    topology topology;
    topology.add(input_layout("input0", dynamic_input_layout));
    topology.add(input_layout("input1", dynamic_input_layout));
    topology.add(input_layout("input2", dynamic_input_layout));
    topology.add(range{ "range", { input_info("input0"), input_info("input1"), input_info("input2") }, data_types::i32});

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("range");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    auto output = outputs.at("range").get_memory();

    mem_lock<int32_t> output_ptr(output, tests::get_test_stream());

    for (size_t i = 0; i < static_cast<size_t>(expected_dim); ++i) {
        ASSERT_EQ(start_val + i * step_val, output_ptr[i]);
    }
}

TEST(range_gpu_test, dynamic_stop) {
    auto& engine = get_test_engine();

    int32_t start_val = 0;
    int32_t step_val = 1;
    int32_t expected_dim = 25;

    auto dynamic_input_layout = layout{ ov::PartialShape::dynamic(0), data_types::i32, format::bfyx };

    auto input0 = engine.allocate_memory({ {}, data_types::i32, format::bfyx });
    auto input1 = engine.allocate_memory({ {}, data_types::i32, format::bfyx });
    auto input2 = engine.allocate_memory({ {}, data_types::i32, format::bfyx });

    set_values<int32_t>(input0, { start_val });
    set_values<int32_t>(input1, { expected_dim });
    set_values<int32_t>(input2, { step_val });

    topology topology;
    topology.add(data("input0", input0));
    topology.add(input_layout("input1", dynamic_input_layout));
    topology.add(data("input2", input2));
    topology.add(range{ "range", { input_info("input0"), input_info("input1"), input_info("input2") }, data_types::i32});

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);
    network.set_input_data("input1", input1);

    auto inst = network.get_primitive("range");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    auto output = outputs.at("range").get_memory();

    mem_lock<int32_t> output_ptr(output, tests::get_test_stream());

    for (size_t i = 0; i < static_cast<size_t>(expected_dim); ++i) {
        ASSERT_EQ(start_val + i * step_val, output_ptr[i]);
    }
}

TEST(range_cpu_impl_test, dynamic_all) {
    auto& engine = get_test_engine();

    int32_t start_val = 0;
    int32_t step_val = 1;
    int32_t expected_dim = 25;

    auto dynamic_input_layout = layout{ ov::PartialShape::dynamic(0), data_types::i32, format::bfyx };

    auto input0 = engine.allocate_memory({ {}, data_types::i32, format::bfyx });
    auto input1 = engine.allocate_memory({ {}, data_types::i32, format::bfyx });
    auto input2 = engine.allocate_memory({ {}, data_types::i32, format::bfyx });

    set_values<int32_t>(input0, { start_val });
    set_values<int32_t>(input1, { expected_dim });
    set_values<int32_t>(input2, { step_val });

    topology topology;
    topology.add(input_layout("input0", dynamic_input_layout));
    topology.add(input_layout("input1", dynamic_input_layout));
    topology.add(input_layout("input2", dynamic_input_layout));
    topology.add(range{ "range", { input_info("input0"), input_info("input1"), input_info("input2") }, data_types::i32});

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"range", {format::bfyx, "", impl_types::cpu}} }));

    network network(engine, topology, config);
    network.set_input_data("input0", input0);
    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto inst = network.get_primitive("range");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    auto output = outputs.at("range").get_memory();

    mem_lock<int32_t> output_ptr(output, tests::get_test_stream());

    for (size_t i = 0; i < static_cast<size_t>(expected_dim); ++i) {
        ASSERT_EQ(start_val + i * step_val, output_ptr[i]);
    }
}

TEST(range_cpu_impl_test, dynamic_stop) {
    auto& engine = get_test_engine();

    int32_t start_val = 0;
    int32_t step_val = 1;
    int32_t expected_dim = 25;

    auto dynamic_input_layout = layout{ ov::PartialShape::dynamic(0), data_types::i32, format::bfyx };

    auto input0 = engine.allocate_memory({ {}, data_types::i32, format::bfyx });
    auto input1 = engine.allocate_memory({ {}, data_types::i32, format::bfyx });
    auto input2 = engine.allocate_memory({ {}, data_types::i32, format::bfyx });

    set_values<int32_t>(input0, { start_val });
    set_values<int32_t>(input1, { expected_dim });
    set_values<int32_t>(input2, { step_val });

    topology topology;
    topology.add(data("input0", input0));
    topology.add(input_layout("input1", dynamic_input_layout));
    topology.add(data("input2", input2));
    topology.add(range{ "range", { input_info("input0"), input_info("input1"), input_info("input2") }, data_types::i32});

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"range", {format::bfyx, "", impl_types::cpu}} }));

    network network(engine, topology, config);
    network.set_input_data("input1", input1);

    auto inst = network.get_primitive("range");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    auto output = outputs.at("range").get_memory();

    mem_lock<int32_t> output_ptr(output, tests::get_test_stream());

    for (size_t i = 0; i < static_cast<size_t>(expected_dim); ++i) {
        ASSERT_EQ(start_val + i * step_val, output_ptr[i]);
    }
}

}  // namespace
}  // namespace cldnn
