// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "primitive_inst.h"

#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/broadcast.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/resample.hpp>
#include <intel_gpu/primitives/shape_of.hpp>

namespace {

class remote_output_lifecycle : public testing::TestWithParam<cldnn::allocation_type> {};

class RemoteOutputLifecycle {
public:
    explicit RemoteOutputLifecycle(cldnn::network& network,
                                                                     cldnn::allocation_type output_allocation,
                                   bool require_direct_output = true,
                                   const cldnn::primitive_id& operation_id = "operation")
                : network(network), engine(network.get_engine()), output_allocation(output_allocation),
                    require_direct_output(require_direct_output), operation_id(operation_id) {}

    void run(const ov::Shape& input_shape,
             const ov::Shape& output_shape,
             const std::vector<float>& input_values,
             const std::vector<float>& expected,
             bool expect_skip,
             bool reuse_output = false,
               bool is_remote = true) {
        SCOPED_TRACE("iteration " + std::to_string(iteration++));
           auto input = engine.allocate_memory({input_shape, cldnn::data_types::f32, cldnn::format::bfyx});
        tests::set_values(input, input_values);
        network.set_input_data("input", input);

        const auto output_layout = cldnn::layout{output_shape, cldnn::data_types::f32, cldnn::format::bfyx};
        auto destination = reuse_output ? engine.reinterpret_buffer(*last_output, output_layout)
                                        : engine.allocate_memory(output_layout, output_allocation);
        if (!reuse_output)
            tests::set_values(destination, std::vector<float>(expected.size(), -99.f));
        network.set_output_memory("output", destination, is_remote);
        auto outputs = network.execute();
        auto result = outputs.at("output").get_memory();
        auto operation = network.get_primitive(operation_id);
        EXPECT_EQ(operation->can_be_optimized(), expect_skip);
        if (!expect_skip) {
            EXPECT_FALSE(engine.is_the_same_buffer(operation->input_memory(), operation->output_memory()));
        }
        const bool direct_output = engine.is_the_same_buffer(*result, *destination);
        if (require_direct_output) {
            EXPECT_TRUE(direct_output);
        }
        expect_values(result, expected);
        expect_values(input, input_values);

        for (auto& retained : retained_outputs) {
            if (engine.is_the_same_buffer(*retained.first, *destination)) {
                retained.second = expected;
            } else {
                EXPECT_FALSE(engine.is_the_same_buffer(network.get_primitive("producer")->output_memory(), *retained.first));
                expect_values(retained.first, retained.second);
            }
        }
        if (!direct_output)
            result->copy_to(network.get_stream(), *destination, true);
        expect_values(destination, expected);
        if (!reuse_output)
            retained_outputs.emplace_back(destination, expected);
        last_output = destination;
        network.reset_output_remote_memory_ptrs();
    }

private:
    void expect_values(const cldnn::memory::ptr& memory, const std::vector<float>& expected) {
        cldnn::mem_lock<float, cldnn::mem_lock_type::read> values(memory, network.get_stream());
        for (size_t index = 0; index < expected.size(); ++index) {
            EXPECT_FLOAT_EQ(values[index], expected[index]) << "index " << index;
        }
    }

    cldnn::network& network;
    cldnn::engine& engine;
    cldnn::allocation_type output_allocation;
    bool require_direct_output;
    cldnn::primitive_id operation_id;
    cldnn::memory::ptr last_output;
    std::vector<std::pair<cldnn::memory::ptr, std::vector<float>>> retained_outputs;
    size_t iteration = 0;
};

TEST_P(remote_output_lifecycle, gather) {
    auto& engine = tests::get_test_engine();
    if (!engine.supports_allocation(GetParam())) {
        GTEST_SKIP() << "Output allocation type is not supported";
    }
    const auto input_layout = cldnn::layout{ov::PartialShape{1, ov::Dimension(1, 6)}, cldnn::data_types::f32, cldnn::format::bfyx};
    const auto index_layout = cldnn::layout{ov::PartialShape{ov::Dimension(1, 6)}, cldnn::data_types::i32, cldnn::format::bfyx};
    cldnn::topology topology(cldnn::input_layout("input", input_layout),
                            cldnn::input_layout("indices", index_layout),
                            cldnn::activation("producer", cldnn::input_info("input"), cldnn::activation_func::relu),
                            cldnn::gather("operation", cldnn::input_info("producer"), cldnn::input_info("indices"), 1, 2, ov::Shape{}, 0, true),
                            cldnn::reorder("output", cldnn::input_info("operation"), cldnn::format::bfyx, cldnn::data_types::f32));
    auto config = tests::get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    cldnn::network network(engine, topology, config);
    ASSERT_TRUE(network.get_primitive("operation")->get_node().is_runtime_skippable());
    auto indices = engine.allocate_memory({{6}, cldnn::data_types::i32, cldnn::format::bfyx});
    network.set_input_data("indices", indices);
    RemoteOutputLifecycle lifecycle(network, GetParam(), false);
    tests::set_values(indices, std::vector<int32_t>{0, 1, 2, 3, 4, 5});
    lifecycle.run({1, 6}, {1, 6}, {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, true);
    tests::set_values(indices, std::vector<int32_t>{5, 4, 3, 2, 1, 0});
    lifecycle.run({1, 6}, {1, 6}, {7, 8, 9, 10, 11, 12}, {12, 11, 10, 9, 8, 7}, false);
    tests::set_values(indices, std::vector<int32_t>{0, 1, 2, 3, 4, 5});
    lifecycle.run({1, 6}, {1, 6}, {2, 3, 4, 5, 6, 7}, {2, 3, 4, 5, 6, 7}, true, true);
    tests::set_values(indices, std::vector<int32_t>{5, 4, 3, 2, 1, 0});
    lifecycle.run({1, 6}, {1, 6}, {3, 4, 5, 6, 7, 8}, {8, 7, 6, 5, 4, 3}, false, true);
    tests::set_values(indices, std::vector<int32_t>{0, 1, 2, 3, 4, 5});
    lifecycle.run({1, 6}, {1, 6}, {4, 5, 6, 7, 8, 9}, {4, 5, 6, 7, 8, 9}, true, false, false);
}

TEST_P(remote_output_lifecycle, broadcast) {
    auto& engine = tests::get_test_engine();
    if (!engine.supports_allocation(GetParam())) {
        GTEST_SKIP() << "Output allocation type is not supported";
    }
    const auto input_layout = cldnn::layout{ov::PartialShape{1, ov::Dimension(1, 2), 3}, cldnn::data_types::f32, cldnn::format::bfyx};
    cldnn::topology topology(cldnn::input_layout("input", input_layout),
                            cldnn::activation("producer", cldnn::input_info("input"), cldnn::activation_func::relu),
                            cldnn::broadcast("operation", cldnn::input_info("producer"), ov::Shape{1, 2, 3}, ov::AxisSet{},
                                             ov::op::BroadcastType::NUMPY),
                            cldnn::shape_of("shape", cldnn::input_info("operation"), cldnn::data_types::i32),
                            cldnn::reorder("output", cldnn::input_info("operation"), cldnn::format::bfyx, cldnn::data_types::f32));
    auto config = tests::get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    cldnn::network network(engine, topology, config);
    ASSERT_TRUE(network.get_primitive("operation")->get_node().is_runtime_skippable());
    RemoteOutputLifecycle lifecycle(network, GetParam(), false);
    lifecycle.run({1, 2, 3}, {1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, true);
    lifecycle.run({1, 1, 3}, {1, 2, 3}, {7, 8, 9}, {7, 8, 9, 7, 8, 9}, false);
    lifecycle.run({1, 2, 3}, {1, 2, 3}, {2, 3, 4, 5, 6, 7}, {2, 3, 4, 5, 6, 7}, true, true);
    lifecycle.run({1, 1, 3}, {1, 2, 3}, {3, 4, 5}, {3, 4, 5, 3, 4, 5}, false, true);
    lifecycle.run({1, 2, 3}, {1, 2, 3}, {4, 5, 6, 7, 8, 9}, {4, 5, 6, 7, 8, 9}, true, false, false);
}

TEST_P(remote_output_lifecycle, reorder) {
    auto& engine = tests::get_test_engine();
    if (!engine.supports_allocation(GetParam())) {
        GTEST_SKIP() << "Output allocation type is not supported";
    }
    const auto input_layout = cldnn::layout{ov::PartialShape{1, 1, ov::Dimension(1, 6), ov::Dimension(1, 6)},
                                           cldnn::data_types::f32, cldnn::format::bfyx};
    cldnn::topology topology(cldnn::input_layout("input", input_layout),
                            cldnn::activation("producer", cldnn::input_info("input"), cldnn::activation_func::relu),
                            cldnn::reorder("output", cldnn::input_info("producer"), cldnn::format::bfyx, cldnn::data_types::f32));
    auto config = tests::get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    cldnn::network network(engine, topology, config);
    ASSERT_TRUE(network.get_primitive("output")->get_node().is_runtime_skippable());
    RemoteOutputLifecycle lifecycle(network, GetParam(), true, "output");
    lifecycle.run({1, 1, 2, 3}, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, true);
    lifecycle.run({1, 1, 1, 6}, {1, 1, 1, 6}, {7, 8, 9, 10, 11, 12}, {7, 8, 9, 10, 11, 12}, true);
    lifecycle.run({1, 1, 1, 6}, {1, 1, 1, 6}, {2, 3, 4, 5, 6, 7}, {2, 3, 4, 5, 6, 7}, true, true);
    lifecycle.run({1, 1, 2, 3}, {1, 1, 2, 3}, {3, 4, 5, 6, 7, 8}, {3, 4, 5, 6, 7, 8}, true, true);
    lifecycle.run({1, 1, 2, 3}, {1, 1, 2, 3}, {4, 5, 6, 7, 8, 9}, {4, 5, 6, 7, 8, 9}, true, false, false);
}

TEST_P(remote_output_lifecycle, resample) {
    using Interpolate = cldnn::resample::InterpolateOp;
    auto& engine = tests::get_test_engine();
    if (!engine.supports_allocation(GetParam())) {
        GTEST_SKIP() << "Output allocation type is not supported";
    }
    const auto input_layout = cldnn::layout{ov::PartialShape{1, 1, ov::Dimension(1, 6), ov::Dimension(1, 6)},
                                           cldnn::data_types::f32, cldnn::format::bfyx};
    cldnn::topology topology(cldnn::input_layout("input", input_layout),
                            cldnn::activation("producer", cldnn::input_info("input"), cldnn::activation_func::relu),
                            cldnn::resample("operation", cldnn::input_info("producer"), std::vector<int64_t>{2, 3},
                                            std::vector<float>{}, std::vector<int64_t>{2, 3}, std::vector<size_t>{},
                                            std::vector<size_t>{}, 0, -0.75f, Interpolate::InterpolateMode::NEAREST,
                                            Interpolate::ShapeCalcMode::SIZES, Interpolate::CoordinateTransformMode::ASYMMETRIC,
                                            Interpolate::NearestMode::FLOOR),
                            cldnn::reorder("output", cldnn::input_info("operation"), cldnn::format::bfyx, cldnn::data_types::f32));
    auto config = tests::get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));
    cldnn::network network(engine, topology, config);
    ASSERT_TRUE(network.get_primitive("operation")->get_node().is_runtime_skippable());
    RemoteOutputLifecycle lifecycle(network, GetParam());
    lifecycle.run({1, 1, 2, 3}, {1, 1, 2, 3}, {1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6}, true);
    lifecycle.run({1, 1, 1, 6}, {1, 1, 2, 3}, {7, 8, 9, 10, 11, 12}, {7, 9, 11, 7, 9, 11}, false);
    lifecycle.run({1, 1, 2, 3}, {1, 1, 2, 3}, {2, 3, 4, 5, 6, 7}, {2, 3, 4, 5, 6, 7}, true, true);
    lifecycle.run({1, 1, 1, 6}, {1, 1, 2, 3}, {3, 4, 5, 6, 7, 8}, {3, 5, 7, 3, 5, 7}, false, true);
    lifecycle.run({1, 1, 2, 3}, {1, 1, 2, 3}, {4, 5, 6, 7, 8, 9}, {4, 5, 6, 7, 8, 9}, true, false, false);
}

INSTANTIATE_TEST_SUITE_P(smoke,
                         remote_output_lifecycle,
                         testing::Values(cldnn::allocation_type::cl_mem, cldnn::allocation_type::usm_host));

}