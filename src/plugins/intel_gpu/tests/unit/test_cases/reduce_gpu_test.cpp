// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <cstring>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reduce.hpp>
#include <intel_gpu/primitives/reorder.hpp>

#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"
#include "random_generator.hpp"
#include "reduce_inst.h"
#include "registry/implementation_manager.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

static void test_weighted_reduce_x16_matches_multiply_reduce_bit_exact(bool is_caching_test) {
    auto& engine = get_test_engine();
    const layout values_layout(data_types::f16, format::bfyx, tensor(2, 32, 16, 1025));
    const layout weights_layout(data_types::f16, format::bfyx, tensor(2, 1, 16, 1025));
    auto values = engine.allocate_memory(values_layout);
    auto weights = engine.allocate_memory(weights_layout);
    VF<ov::float16> values_data(values_layout.count());
    VF<ov::float16> weights_data(weights_layout.count());
    for (size_t i = 0; i < values_data.size(); ++i)
        values_data[i] = ov::float16((static_cast<int>(i % 29) - 14) * 0.0731f + 0.0113f);
    for (size_t i = 0; i < weights_data.size(); ++i)
        weights_data[i] = ov::float16((static_cast<int>(i % 13) - 6) * 0.0473f - 0.0097f);
    const float tie_values[16] = {-1.1669921875f,
                                  -0.71533203125f,
                                  -0.72509765625f,
                                  -1.4033203125f,
                                  -2.767578125f,
                                  -3.353515625f,
                                  -2.685546875f,
                                  -2.5f,
                                  -0.1978759765625f,
                                  -0.1888427734375f,
                                  0.259521484375f,
                                  0.541015625f,
                                  0.947265625f,
                                  0.9755859375f,
                                  1.263671875f,
                                  0.0f};
    const float tie_weights[16] = {0.0019483566284179688f,
                                   0.0168609619140625f,
                                   0.031280517578125f,
                                   0.0096588134765625f,
                                   0.23291015625f,
                                   0.05780029296875f,
                                   0.037567138671875f,
                                   0.0026397705078125f,
                                   0.00949859619140625f,
                                   0.013580322265625f,
                                   0.00011110305786132812f,
                                   0.00013875961303710938f,
                                   0.005435943603515625f,
                                   0.0040283203125f,
                                   0.0025119781494140625f,
                                   0.57421875f};
    for (size_t i = 0; i < 16; ++i) {
        values_data[i] = ov::float16(tie_values[i]);
        weights_data[i] = ov::float16(tie_weights[i]);
    }
    set_values(values, values_data);
    set_values(weights, weights_data);
    {
        mem_lock<ov::float16, mem_lock_type::write> values_ptr(values, get_test_stream());
        mem_lock<ov::float16, mem_lock_type::write> weights_ptr(weights, get_test_stream());
        for (size_t x = 0; x < 16; ++x) {
            const auto value_offset = values_layout.get_linear_offset(tensor(batch(1), feature(25), spatial(x, 754, 0, 0)));
            const auto weight_offset = weights_layout.get_linear_offset(tensor(batch(1), feature(0), spatial(x, 754, 0, 0)));
            values_ptr[value_offset] = ov::float16(tie_values[x]);
            weights_ptr[weight_offset] = ov::float16(tie_weights[x]);
            const auto positive_value_offset = values_layout.get_linear_offset(tensor(batch(1), feature(26), spatial(x, 754, 0, 0)));
            values_ptr[positive_value_offset] = ov::float16(-tie_values[x]);
        }
    }

    topology ref_topology;
    ref_topology.add(input_layout("values", values_layout));
    ref_topology.add(input_layout("weights", weights_layout));
    ref_topology.add(eltwise("multiply", {input_info("values"), input_info("weights")}, eltwise_mode::prod));
    ref_topology.add(reorder("multiply_fsv16", input_info("multiply"), format::b_fs_yx_fsv16, data_types::f16));
    ref_topology.add(reduce("reference", input_info("multiply_fsv16"), reduce_mode::sum, {3}, false));
    ref_topology.add(reorder("reference_out", input_info("reference"), format::bfyx, data_types::f16));
    ExecutionConfig ref_config = get_test_default_config(engine);
    ref_config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"reference_out"}));
    ref_config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"multiply", {format::bfyx, "generic_eltwise_ref"}}}));
    network ref_network(engine, ref_topology, ref_config);
    ref_network.set_input_data("values", values);
    ref_network.set_input_data("weights", weights);
    auto ref_output = ref_network.execute().at("reference_out").get_memory();

    topology opt_topology;
    opt_topology.add(input_layout("values", values_layout));
    opt_topology.add(input_layout("weights", weights_layout));
    opt_topology.add(reduce("weighted", input_info("values"), input_info("weights"), reduce_mode::sum, {3}, false));
    opt_topology.add(reorder("weighted_out", input_info("weighted"), format::bfyx, data_types::f16));
    ExecutionConfig opt_config = get_test_default_config(engine);
    opt_config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"weighted_out"}));
    auto opt_network = get_network(engine, opt_topology, opt_config, get_test_stream_ptr(), is_caching_test);
    opt_network->set_input_data("values", values);
    opt_network->set_input_data("weights", weights);
    const auto info = opt_network->get_primitive_info("weighted");
    ASSERT_NE(info.find("weighted_reduce_x16"), std::string::npos) << info;
    auto opt_output = opt_network->execute().at("weighted_out").get_memory();

    mem_lock<ov::float16, mem_lock_type::read> ref_ptr(ref_output, get_test_stream());
    mem_lock<ov::float16, mem_lock_type::read> opt_ptr(opt_output, get_test_stream());
    ASSERT_EQ(ref_output->get_layout().count(), opt_output->get_layout().count());
    size_t mismatch_count = 0;
    for (size_t i = 0; i < ref_output->get_layout().count(); ++i) {
        if (ref_ptr[i].to_bits() != opt_ptr[i].to_bits()) {
            if (mismatch_count < 20) {
                std::cout << "mismatch " << i << " ref_bits=" << ref_ptr[i].to_bits() << " opt_bits=" << opt_ptr[i].to_bits()
                          << " ref=" << static_cast<float>(ref_ptr[i]) << " opt=" << static_cast<float>(opt_ptr[i]) << std::endl;
            }
            ++mismatch_count;
        }
    }
    ASSERT_EQ(mismatch_count, 0);
}

TEST(reduce_gpu, weighted_reduce_x16_matches_multiply_reduce_bit_exact) {
    test_weighted_reduce_x16_matches_multiply_reduce_bit_exact(false);
}

TEST(reduce_gpu, weighted_reduce_x16_cached_matches_multiply_reduce_bit_exact) {
    test_weighted_reduce_x16_matches_multiply_reduce_bit_exact(true);
}

TEST(reduce_gpu, weighted_reduce_x16_f32_matches_multiply_reduce_bit_exact) {
    auto& engine = get_test_engine();
    const layout values_layout(data_types::f32, format::bfyx, tensor(1, 4, 16, 1025));
    const layout weights_layout(data_types::f32, format::bfyx, tensor(1, 1, 16, 1025));
    auto values = engine.allocate_memory(values_layout);
    auto weights = engine.allocate_memory(weights_layout);
    VF<float> values_data(values_layout.count());
    VF<float> weights_data(weights_layout.count());
    for (size_t i = 0; i < values_data.size(); ++i)
        values_data[i] = (static_cast<int>(i % 29) - 14) * 0.0731f + 0.0113f;
    for (size_t i = 0; i < weights_data.size(); ++i)
        weights_data[i] = (static_cast<int>(i % 13) - 6) * 0.0473f - 0.0097f;
    set_values(values, values_data);
    set_values(weights, weights_data);

    topology ref_topology;
    ref_topology.add(input_layout("values", values_layout));
    ref_topology.add(input_layout("weights", weights_layout));
    ref_topology.add(eltwise("multiply", {input_info("values"), input_info("weights")}, eltwise_mode::prod));
    ref_topology.add(reorder("multiply_fsv16", input_info("multiply"), format::b_fs_yx_fsv16, data_types::f32));
    ref_topology.add(reduce("reference", input_info("multiply_fsv16"), reduce_mode::sum, {3}, false));
    ref_topology.add(reorder("reference_out", input_info("reference"), format::bfyx, data_types::f32));
    ExecutionConfig ref_config = get_test_default_config(engine);
    ref_config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"reference_out"}));
    ref_config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"multiply", {format::bfyx, "generic_eltwise_ref"}}}));
    network ref_network(engine, ref_topology, ref_config);
    ref_network.set_input_data("values", values);
    ref_network.set_input_data("weights", weights);
    auto ref_output = ref_network.execute().at("reference_out").get_memory();

    topology opt_topology;
    opt_topology.add(input_layout("values", values_layout));
    opt_topology.add(input_layout("weights", weights_layout));
    opt_topology.add(reduce("weighted", input_info("values"), input_info("weights"), reduce_mode::sum, {3}, false));
    opt_topology.add(reorder("weighted_out", input_info("weighted"), format::bfyx, data_types::f32));
    ExecutionConfig opt_config = get_test_default_config(engine);
    opt_config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"weighted_out"}));
    network opt_network(engine, opt_topology, opt_config);
    opt_network.set_input_data("values", values);
    opt_network.set_input_data("weights", weights);
    const auto info = opt_network.get_primitive_info("weighted");
    ASSERT_NE(info.find("weighted_reduce_x16"), std::string::npos) << info;
    auto opt_output = opt_network.execute().at("weighted_out").get_memory();

    mem_lock<float, mem_lock_type::read> ref_ptr(ref_output, get_test_stream());
    mem_lock<float, mem_lock_type::read> opt_ptr(opt_output, get_test_stream());
    ASSERT_EQ(ref_output->get_layout().bytes_count(), opt_output->get_layout().bytes_count());
    ASSERT_EQ(std::memcmp(ref_ptr.data(), opt_ptr.data(), ref_output->get_layout().bytes_count()), 0);
}

namespace {

struct weighted_reduce_graph_case {
    int64_t y;
    int64_t x;
    int64_t axis;
    bool reverse_inputs;
    bool multiple_consumers;
    bool dynamic_y;
    bool keep_dims;
    bool relu;
    ov::element::Type element_type;
    bool expect_weighted;
    bool expect_reference;
    const char* name;
    int64_t values_batch = 1;
    int64_t weights_batch = 1;
    int64_t values_features = 32;
    int64_t weights_features = 1;
    int64_t rank = 4;
    std::vector<int64_t> axes;
    int64_t weights_y = 0;
    int64_t weights_x = 0;
};

std::vector<ov::Dimension> make_weighted_reduce_dims(const weighted_reduce_graph_case& test_case, int64_t batch, int64_t features, bool is_weights) {
    const auto y = is_weights && test_case.weights_y > 0 ? test_case.weights_y : test_case.y;
    const auto x = is_weights && test_case.weights_x > 0 ? test_case.weights_x : test_case.x;
    std::vector<ov::Dimension> dims;
    if (test_case.rank == 3) {
        dims = {features, y, x};
    } else {
        dims = {batch, features, y, x};
    }
    if (test_case.dynamic_y)
        dims[dims.size() - 2] = ov::Dimension::dynamic();
    return dims;
}

ov::Shape make_weighted_reduce_shape(const weighted_reduce_graph_case& test_case, int64_t batch, int64_t features, bool is_weights) {
    const auto y = is_weights && test_case.weights_y > 0 ? test_case.weights_y : test_case.y;
    const auto x = is_weights && test_case.weights_x > 0 ? test_case.weights_x : test_case.x;
    if (test_case.rank == 3)
        return {static_cast<size_t>(features), static_cast<size_t>(y), static_cast<size_t>(x)};
    return {static_cast<size_t>(batch), static_cast<size_t>(features), static_cast<size_t>(y), static_cast<size_t>(x)};
}

std::shared_ptr<ov::Model> make_weighted_reduce_model(const weighted_reduce_graph_case& test_case, bool force_multiply_output = false) {
    auto values = std::make_shared<ov::op::v0::Parameter>(
        test_case.element_type,
        ov::PartialShape(make_weighted_reduce_dims(test_case, test_case.values_batch, test_case.values_features, false)));
    auto weights = std::make_shared<ov::op::v0::Parameter>(
        test_case.element_type,
        ov::PartialShape(make_weighted_reduce_dims(test_case, test_case.weights_batch, test_case.weights_features, true)));
    values->set_friendly_name("values");
    weights->set_friendly_name("weights");
    auto multiply =
        test_case.reverse_inputs ? std::make_shared<ov::op::v1::Multiply>(weights, values) : std::make_shared<ov::op::v1::Multiply>(values, weights);
    multiply->set_friendly_name("multiply");
    const auto axes_values = test_case.axes.empty() ? std::vector<int64_t>{test_case.axis} : test_case.axes;
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes_values.size()}, axes_values);
    auto reduce = std::make_shared<ov::op::v1::ReduceSum>(multiply, axes, test_case.keep_dims);
    reduce->set_friendly_name("reduce");
    std::shared_ptr<ov::Node> output = reduce;
    if (test_case.relu) {
        output = std::make_shared<ov::op::v0::Relu>(reduce);
        output->set_friendly_name("relu");
    }
    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(output)};
    if (test_case.multiple_consumers || force_multiply_output)
        results.push_back(std::make_shared<ov::op::v0::Result>(multiply));
    return std::make_shared<ov::Model>(results, ov::ParameterVector{values, weights});
}

class weighted_reduce_graph_test : public ::testing::TestWithParam<weighted_reduce_graph_case> {};

}  // namespace

TEST_P(weighted_reduce_graph_test, selects_or_falls_back) {
    const auto test_case = GetParam();
    ov::Core core;
    auto compiled = core.compile_model(make_weighted_reduce_model(test_case), "GPU", ov::enable_profiling(true));
    auto reference = core.compile_model(make_weighted_reduce_model(test_case, true), "GPU", ov::enable_profiling(true));
    auto request = compiled.create_infer_request();
    auto reference_request = reference.create_infer_request();
    ov::Tensor values(test_case.element_type, make_weighted_reduce_shape(test_case, test_case.values_batch, test_case.values_features, false));
    ov::Tensor weights(test_case.element_type, make_weighted_reduce_shape(test_case, test_case.weights_batch, test_case.weights_features, true));
    const auto value_at = [&](size_t i) {
        return test_case.relu ? -1.0f : (static_cast<int>(i % 29) - 14) * 0.125f;
    };
    const auto weight_at = [&](size_t i) {
        return test_case.relu ? 1.0f : (static_cast<int>(i % 13) - 6) * 0.0625f;
    };
    if (test_case.element_type == ov::element::f32) {
        for (size_t i = 0; i < values.get_size(); ++i)
            values.data<float>()[i] = value_at(i);
        for (size_t i = 0; i < weights.get_size(); ++i)
            weights.data<float>()[i] = weight_at(i);
    } else if (test_case.element_type == ov::element::i32) {
        for (size_t i = 0; i < values.get_size(); ++i)
            values.data<int32_t>()[i] = static_cast<int32_t>(value_at(i));
        for (size_t i = 0; i < weights.get_size(); ++i)
            weights.data<int32_t>()[i] = static_cast<int32_t>(weight_at(i));
    } else if (test_case.element_type == ov::element::bf16) {
        for (size_t i = 0; i < values.get_size(); ++i)
            values.data<ov::bfloat16>()[i] = ov::bfloat16(value_at(i));
        for (size_t i = 0; i < weights.get_size(); ++i)
            weights.data<ov::bfloat16>()[i] = ov::bfloat16(weight_at(i));
    } else {
        for (size_t i = 0; i < values.get_size(); ++i)
            values.data<ov::float16>()[i] = ov::float16(value_at(i));
        for (size_t i = 0; i < weights.get_size(); ++i)
            weights.data<ov::float16>()[i] = ov::float16(weight_at(i));
    }
    request.set_input_tensor(0, values);
    request.set_input_tensor(1, weights);
    reference_request.set_input_tensor(0, values);
    reference_request.set_input_tensor(1, weights);
    request.infer();
    reference_request.infer();

    const auto output = request.get_output_tensor(0);
    const auto reference_output = reference_request.get_output_tensor(0);
    ASSERT_EQ(output.get_element_type(), reference_output.get_element_type());
    ASSERT_EQ(output.get_shape(), reference_output.get_shape());
    ASSERT_EQ(output.get_byte_size(), reference_output.get_byte_size());
    if (test_case.relu) {
        const auto* output_data = output.data<const ov::float16>();
        const auto* reference_data = reference_output.data<const ov::float16>();
        for (size_t i = 0; i < output.get_size(); ++i) {
            ASSERT_EQ(static_cast<float>(output_data[i]), static_cast<float>(reference_data[i])) << "Mismatch at element " << i;
            ASSERT_EQ(static_cast<float>(output_data[i]), 0.0f) << "ReLU mismatch at element " << i;
        }
    } else {
        ASSERT_EQ(std::memcmp(output.data(), reference_output.data(), output.get_byte_size()), 0);
    }

    size_t weighted_count = 0;
    size_t weighted_reference_count = 0;
    size_t multiply_count = 0;
    for (const auto& info : request.get_profiling_info()) {
        if (info.exec_type.find("weighted_reduce_x16") != std::string::npos)
            ++weighted_count;
        if (info.node_name == "reduce" && info.exec_type.find("reduce_ref") != std::string::npos)
            ++weighted_reference_count;
        if (info.node_name == "multiply" && info.exec_type != "undef")
            ++multiply_count;
    }
    EXPECT_EQ(weighted_count, test_case.expect_weighted ? 1 : 0);
    if (test_case.expect_reference) {
        EXPECT_EQ(weighted_reference_count, 1);
    } else if (test_case.expect_weighted) {
        EXPECT_EQ(weighted_reference_count, 0);
    }
    EXPECT_EQ(multiply_count, (test_case.expect_weighted || test_case.expect_reference) ? 0 : 1);
}

INSTANTIATE_TEST_SUITE_P(
    weighted_reduce_graph,
    weighted_reduce_graph_test,
    ::testing::Values(
        weighted_reduce_graph_case{1025, 16, 3, false, false, false, false, false, ov::element::f16, true, false, "positive_f16"},
        weighted_reduce_graph_case{1025, 16, 3, false, false, false, false, false, ov::element::f32, true, false, "positive_f32"},
        weighted_reduce_graph_case{1025, 16, 3, true, false, false, false, false, ov::element::f16, true, false, "reversed_inputs"},
        weighted_reduce_graph_case{1025, 16, -1, false, false, false, false, false, ov::element::f16, true, false, "negative_last_axis"},
        weighted_reduce_graph_case{1025, 16, 3, false, false, false, true, false, ov::element::f16, true, false, "keep_dims"},
        weighted_reduce_graph_case{1025, 16, 3, false, false, false, false, false, ov::element::f16, true, false, "batch_2", 2, 2},
        weighted_reduce_graph_case{1025, 16, 3, false, false, false, true, true, ov::element::f16, false, true, "fused_relu"},
        weighted_reduce_graph_case{1024, 16, 3, false, false, false, false, false, ov::element::f16, false, false, "threshold_boundary"},
        weighted_reduce_graph_case{1025, 8, 3, false, false, false, false, false, ov::element::f16, false, false, "wrong_x"},
        weighted_reduce_graph_case{1025, 16, 2, false, false, false, false, false, ov::element::f16, false, false, "wrong_axis"},
        weighted_reduce_graph_case{1025, 16, 3, false, true, false, false, false, ov::element::f16, false, false, "multiple_consumers"},
        weighted_reduce_graph_case{1025, 16, 3, false, false, true, false, false, ov::element::f16, false, false, "dynamic_y"},
        weighted_reduce_graph_case{1025, 16, 3, false, false, false, false, false, ov::element::i32, false, false, "unsupported_i32"},
        weighted_reduce_graph_case{1025, 16, 3, false, false, false, false, false, ov::element::bf16, true, false, "bf16_converted_to_supported_precision"},
        weighted_reduce_graph_case{1025, 16, 3, false, false, false, false, false, ov::element::f16, false, false, "both_full", 1, 1, 32, 32},
        weighted_reduce_graph_case{1025, 16, 3, false, false, false, false, false, ov::element::f16, false, false, "both_feature_one", 1, 1, 1, 1},
        weighted_reduce_graph_case{1025, 16, 2, false, false, false, false, false, ov::element::f16, false, false, "rank_3", 1, 1, 32, 1, 3},
        weighted_reduce_graph_case{1025, 16, 3, false, false, false, false, false, ov::element::f16, false, false, "batch_broadcast", 2, 1},
        weighted_reduce_graph_case{1025, 16, 3, false, false, false, false, false, ov::element::f16, false, false, "mismatched_y", 1, 1, 32, 1, 4, {}, 1, 0},
        weighted_reduce_graph_case{1025, 16, 3, false, false, false, false, false, ov::element::f16, false, false, "mismatched_x", 1, 1, 32, 1, 4, {}, 0, 1},
        weighted_reduce_graph_case{1025, 16, 3, false, false, false, false, false, ov::element::f16, false, false, "multiple_axes", 1, 1, 32, 1, 4, {2, 3}}),
    [](const ::testing::TestParamInfo<weighted_reduce_graph_case>& info) {
        return info.param.name;
    });

namespace {

struct weighted_reduce_layout_case {
    format::type data_format;
    bool pad_values;
    bool pad_weights;
    bool pad_output;
    bool use_f32;
    bool cache;
    const char* name;
};

class weighted_reduce_layout_fallback_test : public ::testing::TestWithParam<weighted_reduce_layout_case> {};

}  // namespace

TEST_P(weighted_reduce_layout_fallback_test, selects_reference_and_executes) {
    const auto test_case = GetParam();
    auto& engine = get_test_engine();
    const auto data_type = test_case.use_f32 ? data_types::f32 : data_types::f16;
    const tensor values_shape(1, 32, 16, 1025);
    const tensor weights_shape(1, 1, 16, 1025);
    const layout values_input_layout(data_type, format::bfyx, values_shape);
    const layout weights_input_layout(data_type, format::bfyx, weights_shape);
    const padding input_padding({0, 0, 0, 1}, {0, 0, 0, 0});
    const layout values_prepared_layout(data_type, test_case.data_format, values_shape, test_case.pad_values ? input_padding : padding{});
    const layout weights_prepared_layout(data_type, test_case.data_format, weights_shape, test_case.pad_weights ? input_padding : padding{});

    auto values = engine.allocate_memory(values_input_layout);
    auto weights = engine.allocate_memory(weights_input_layout);
    VF<float> expected_f32(32 * 1025);
    VF<ov::float16> expected_f16(32 * 1025);
    if (test_case.use_f32) {
        VF<float> values_data(values_input_layout.count());
        VF<float> weights_data(weights_input_layout.count());
        for (size_t i = 0; i < values_data.size(); ++i)
            values_data[i] = (static_cast<int>(i % 29) - 14) * 0.0731f + 0.0113f;
        for (size_t i = 0; i < weights_data.size(); ++i)
            weights_data[i] = (static_cast<int>(i % 13) - 6) * 0.0473f - 0.0097f;
        set_values(values, values_data);
        set_values(weights, weights_data);
        for (size_t f = 0; f < 32; ++f) {
            for (size_t y = 0; y < 1025; ++y) {
                float acc = 0.0f;
                for (size_t x = 0; x < 16; ++x)
                    acc += values_data[(f * 1025 + y) * 16 + x] * weights_data[y * 16 + x];
                expected_f32[f * 1025 + y] = acc;
            }
        }
    } else {
        VF<ov::float16> values_data(values_input_layout.count());
        VF<ov::float16> weights_data(weights_input_layout.count());
        for (size_t i = 0; i < values_data.size(); ++i)
            values_data[i] = ov::float16((static_cast<int>(i % 29) - 14) * 0.0731f + 0.0113f);
        for (size_t i = 0; i < weights_data.size(); ++i)
            weights_data[i] = ov::float16((static_cast<int>(i % 13) - 6) * 0.0473f - 0.0097f);
        set_values(values, values_data);
        set_values(weights, weights_data);
        for (size_t f = 0; f < 32; ++f) {
            for (size_t y = 0; y < 1025; ++y) {
                float acc = 0.0f;
                for (size_t x = 0; x < 16; ++x) {
                    const auto product = ov::float16(static_cast<float>(values_data[(f * 1025 + y) * 16 + x]) * static_cast<float>(weights_data[y * 16 + x]));
                    acc += static_cast<float>(product);
                }
                expected_f16[f * 1025 + y] = ov::float16(acc);
            }
        }
    }

    topology topology;
    topology.add(input_layout("values", values_input_layout));
    topology.add(input_layout("weights", weights_input_layout));
    topology.add(reorder("values_prepared", input_info("values"), values_prepared_layout));
    topology.add(reorder("weights_prepared", input_info("weights"), weights_prepared_layout));
    auto weighted = reduce("weighted", input_info("values_prepared"), input_info("weights_prepared"), reduce_mode::sum, {3}, false);
    if (test_case.pad_output)
        weighted.output_paddings = {padding({0, 0, 0, 1}, {0, 0, 0, 0})};
    topology.add(weighted);
    topology.add(reorder("output", input_info("weighted"), format::bfyx, data_type));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::custom_outputs(std::vector<std::string>{"output"}));
    auto network = get_network(engine, topology, config, get_test_stream_ptr(), test_case.cache);
    network->set_input_data("values", values);
    network->set_input_data("weights", weights);
    const auto info = network->get_primitive_info("weighted");
    ASSERT_NE(info.find("reduce_ref"), std::string::npos) << info;
    ASSERT_EQ(info.find("weighted_reduce_x16"), std::string::npos) << info;
    auto output = network->execute().at("output").get_memory();
    if (test_case.use_f32) {
        mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());
        for (size_t i = 0; i < output->get_layout().count(); ++i)
            ASSERT_EQ(output_ptr[i], expected_f32[i]) << "Mismatch at element " << i;
    } else {
        mem_lock<ov::float16, mem_lock_type::read> output_ptr(output, get_test_stream());
        for (size_t i = 0; i < output->get_layout().count(); ++i)
            ASSERT_EQ(output_ptr[i].to_bits(), expected_f16[i].to_bits()) << "Mismatch at element " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(weighted_reduce_layout_fallback,
                         weighted_reduce_layout_fallback_test,
                         ::testing::Values(weighted_reduce_layout_case{format::b_fs_yx_fsv16, false, false, false, false, false, "blocked_f16"},
                                           weighted_reduce_layout_case{format::b_fs_yx_fsv16, false, false, false, true, false, "blocked_f32"},
                                           weighted_reduce_layout_case{format::bfyx, true, false, false, false, false, "padded_values"},
                                           weighted_reduce_layout_case{format::bfyx, false, true, false, false, false, "padded_weights"},
                                           weighted_reduce_layout_case{format::bfyx, false, false, true, false, false, "padded_output"},
                                           weighted_reduce_layout_case{format::b_fs_yx_fsv16, false, false, false, false, true, "blocked_cached"}),
                         [](const ::testing::TestParamInfo<weighted_reduce_layout_case>& info) {
                             return info.param.name;
                         });

template <typename InputT>
struct accumulator_type {
    using type = float;
};

template <typename InputT>
struct output_type {
    using type = float;
};

template <typename AccT>
AccT get_min_value() {
    return std::numeric_limits<AccT>::lowest();
}

template<>
int get_min_value<int>() {
    return std::numeric_limits<int>::min();
}

template <typename VecT>
struct Comparator {
    std::vector<VecT> order;
    bool operator()(const int lhs, const int rhs) const {
        auto lhs_index = std::distance(order.begin(), std::find(order.begin(), order.end(), lhs));
        auto rhs_index = std::distance(order.begin(), std::find(order.begin(), order.end(), rhs));
        return lhs_index < rhs_index;
    }
};

template <typename InputT, typename AccT>
struct reduce_accumulator {
    AccT set_accumulator_value(cldnn::reduce_mode reduce_mode) {
        AccT acc;
        if (reduce_mode == cldnn::reduce_mode::max)
            acc = get_min_value<AccT>();
        else if (reduce_mode == cldnn::reduce_mode::min)
            acc = std::numeric_limits<AccT>::max();
        else if (reduce_mode == cldnn::reduce_mode::prod || reduce_mode == cldnn::reduce_mode::logical_and)
            acc = 1;
        else
            acc = 0;

        return acc;
    };

    AccT accumulate(AccT& acc, AccT& input_val, cldnn::reduce_mode reduce_mode, bool sum_only) {
        if (reduce_mode == cldnn::reduce_mode::max) {
            acc = input_val > acc ? input_val : acc;
        } else if (reduce_mode == cldnn::reduce_mode::sum || reduce_mode ==  cldnn::reduce_mode::mean ||
            reduce_mode == cldnn::reduce_mode::log_sum) {
            acc += input_val;
        } else if (reduce_mode == cldnn::reduce_mode::min) {
            acc = input_val < acc ? input_val : acc;
        } else if (reduce_mode == cldnn::reduce_mode::prod) {
            acc = acc * input_val;
        } else if (reduce_mode == cldnn::reduce_mode::logical_and) {
            acc = acc && input_val;
        } else if (reduce_mode == cldnn::reduce_mode::logical_or) {
            acc = acc || input_val;
        } else if (reduce_mode == cldnn::reduce_mode::sum_square) {
            if (sum_only)
                acc += input_val;
            else
                acc += input_val * input_val;
        } else if (reduce_mode == cldnn::reduce_mode::l1) {
            acc += abs(input_val);
        } else if (reduce_mode == cldnn::reduce_mode::l2) {
            if (sum_only)
                acc += input_val;
            else
                acc += input_val * input_val;
        } else if (reduce_mode == cldnn::reduce_mode::log_sum_exp) {
            if (sum_only)
                acc += input_val;
            else
                acc += exp(input_val);
        }

        return acc;
    };

    AccT get(AccT& acc, size_t counter, cldnn::reduce_mode reduce_mode) {
        if (reduce_mode == cldnn::reduce_mode::mean)
            acc /= counter;
        else if (reduce_mode == cldnn::reduce_mode::l2)
            acc = sqrt(acc);
        else if (reduce_mode == cldnn::reduce_mode::log_sum || reduce_mode == cldnn::reduce_mode::log_sum_exp)
            acc = log(acc);

        return acc;
    };

    std::map<char, int, Comparator<char>> create_coords_map(std::vector<char>& coords) {
        auto coord_cmp = Comparator<char>();
        coord_cmp.order = coords;
        std::map<char, int, Comparator<char>> coord_map({}, coord_cmp);
        int index = 0;
        for (auto& coord : coord_cmp.order) {
            coord_map[coord] = index;
            index++;
        }

        return coord_map;
    }

    void remap(std::vector<size_t> &out_dims,
                    std::map<char, int, Comparator<char>>& remap_coords,
                    std::vector<uint16_t>& axis_to_remove,
                    std::map<uint16_t, int, Comparator<uint16_t>>& axes_map,
                    std::vector<char>& coords,
                    int dims) {

        if (dims == 5) {
            remap_coords.erase('w');
        } else if (dims == 4) {
            remap_coords.erase('w');
            remap_coords.erase('z');
        }

        // Dimensions reshape
        std::vector<size_t> updated_dims;
        std::vector<char> updated_coords;

        for (int index = 0; index < static_cast<int>(out_dims.size()); index++) {
            if ((dims == 4 && (index == 2 || index == 3)) || (dims == 5 && index == 2))
                continue;

            auto index_to_remove = std::find(axis_to_remove.begin(), axis_to_remove.end(), axes_map.find(index)->second) !=
                     axis_to_remove.end();
            if ((out_dims[index] != 1) || (out_dims[index] == 1 && !index_to_remove)) {
                updated_dims.push_back(out_dims[index]);
                updated_coords.push_back(coords[index]);
            }
        }

	    if (updated_dims.size() > 2) {
            if (dims == 4) {
                updated_dims.insert(updated_dims.begin() + 2, 2, 1);
            } else {
                updated_dims.insert(updated_dims.begin() + 2, out_dims.size() - updated_dims.size(), 1);
            }
        }

        while (updated_dims.size() < out_dims.size())
            updated_dims.push_back(1);

        out_dims = std::move(updated_dims);

        // Coordinates remap
        std::map<uint16_t, int, std::greater<uint16_t>> ordered_axes;

        for (auto& axis : axis_to_remove)
            ordered_axes[axes_map.find(axis)->second] = axis;

        int i = 0;
        for (auto& coord : coords) {
            if ((dims == 4 && (coord == 'w' || coord == 'z')) || (dims == 5 && coord == 'w'))
                continue;

            if (ordered_axes.find(remap_coords[coord]) != ordered_axes.end()) {
                if (dims != 4)
                    updated_coords.insert(updated_coords.begin() + 2 + i, 1, coord);
                else
                    updated_coords.push_back(coord);
                ++i;
            }
        }

        int j = 0;
        for (auto& coord : coords) {
            if ((dims == 4 && (coord == 'w' || coord == 'z')) || (dims == 5 && coord == 'w'))
                continue;

            auto temp_coords = updated_coords.at(j);
            remap_coords[coord] = static_cast<int>(std::distance(coords.begin(), std::find(coords.begin(), coords.end(), temp_coords)));
            ++j;
        }

        if (dims == 4) {
            remap_coords['w'] = 2;
            remap_coords['z'] = 3;
        } else if (dims == 5) {
            remap_coords['w'] = 2;
        }

    }
};

template <typename InputT, typename AccT = typename accumulator_type<InputT>::type, typename OutputT = typename output_type<InputT>::type>
VVVVVVF<OutputT> reference_reduce(VVVVVVF<InputT>& input,
                                  reduce_mode reduce_mode,
                                  std::vector<int64_t> reduce_axis_ov_order,
                                  const int /* batch */,
                                  const int /* input_f */,
                                  const int /* input_w */,
                                  const int /* input_z */,
                                  const int /* input_y */,
                                  const int /* input_x */,
                                  const int dims,
                                  bool keepDims = false) {

    auto reduce = reduce_accumulator<InputT, AccT>();

    std::vector<uint16_t> reduce_axis;
    for (auto& ov_axis : reduce_axis_ov_order) {
        if (ov_axis == 0 || ov_axis == 1) {
            reduce_axis.push_back(ov_axis);
            continue;
        }

        reduce_axis.push_back(dims + 1 - ov_axis);
    }

    auto axis_cmp = Comparator<uint16_t>();
    axis_cmp.order = {0, 1, 5, 4, 3, 2};
    std::map<uint16_t, int, Comparator<uint16_t>> axes_map({}, axis_cmp);

    int index = 0;
    for (auto& axis : axis_cmp.order) {
        axes_map[axis] = index;
        index++;
    }

    // Initial input order is b, f, x, y, w
    std::vector<size_t> input_dims = {
        input.size(),                   // b
        input[0].size(),                // f
        input[0][0][0][0][0].size(),    // w
        input[0][0][0][0].size(),       // z
        input[0][0][0].size(),          // y
        input[0][0].size(),             // x
    };

    VVVVVVF<AccT> previous(input_dims[0],
                           VVVVVF<AccT>(input_dims[1],
                           VVVVF<AccT>(input_dims[2],
                           VVVF<AccT>(input_dims[3],
                           VVF<AccT>(input_dims[4],
                           VF<AccT>(input_dims[5], 0))))));

    for (size_t bi = 0; bi < input_dims[0]; ++bi)
        for (size_t fi = 0; fi < input_dims[1]; ++fi)
            for (size_t wi = 0; wi < input_dims[2]; ++wi)
                for (size_t zi = 0; zi < input_dims[3]; ++zi)
                    for (size_t yi = 0; yi < input_dims[4]; ++yi)
                        for (size_t xi = 0; xi < input_dims[5]; ++xi)
                            previous[bi][fi][wi][zi][yi][xi] = static_cast<AccT>(input[bi][fi][xi][yi][zi][wi]);

    std::vector<size_t> temp_dims = input_dims;
    size_t max_counter_value = 1;

    for (auto& axis : reduce_axis) {
        auto out_dims = temp_dims;
        out_dims[axes_map.at(axis)] = 1;
        VVVVVVF<AccT> temp_output(out_dims[0],
                                  VVVVVF<AccT>(out_dims[1],
                                  VVVVF<AccT>(out_dims[2],
                                  VVVF<AccT>(out_dims[3],
                                  VVF<AccT>(out_dims[4],
                                  VF<AccT>(out_dims[5], reduce.set_accumulator_value(reduce_mode)))))));

        max_counter_value *= input_dims[axes_map.at(axis)];

        for (size_t bi = 0; bi < temp_dims[0]; ++bi)
            for (size_t fi = 0; fi < temp_dims[1]; ++fi)
                for (size_t wi = 0; wi < temp_dims[2]; ++wi)
                    for (size_t zi = 0; zi < temp_dims[3]; ++zi)
                        for (size_t yi = 0; yi < temp_dims[4]; ++yi) {
                            for (size_t xi = 0; xi < temp_dims[5]; ++xi) {
                                auto input_val = static_cast<AccT>(previous[bi][fi][wi][zi][yi][xi]);

                                AccT acc = static_cast<AccT>(temp_output[bi % out_dims[0]][fi % out_dims[1]]
                                                                        [wi % out_dims[2]][zi % out_dims[3]]
                                                                        [yi % out_dims[4]][xi % out_dims[5]]);

                                temp_output[bi % out_dims[0]][fi % out_dims[1]]
                                           [wi % out_dims[2]][zi % out_dims[3]]
                                           [yi % out_dims[4]][xi % out_dims[5]] = reduce.accumulate(acc, input_val, reduce_mode, &axis != &reduce_axis.front());
                            }
                        }
        if (&axis == &reduce_axis.back())
            if (reduce_mode == cldnn::reduce_mode::mean || reduce_mode == cldnn::reduce_mode::l2 ||
                reduce_mode == cldnn::reduce_mode::log_sum || reduce_mode == cldnn::reduce_mode::log_sum_exp) {
                for (size_t bi = 0; bi < temp_output.size(); ++bi)
                    for (size_t fi = 0; fi < temp_output[0].size(); ++fi)
                        for (size_t wi = 0; wi < temp_output[0][0].size(); ++wi)
                            for (size_t zi = 0; zi < temp_output[0][0][0].size(); ++zi)
                                for (size_t yi = 0; yi < temp_output[0][0][0][0].size(); ++yi) {
                                    for (size_t xi = 0; xi < temp_output[0][0][0][0][0].size(); ++xi) {
                                        auto current_acc_val = static_cast<AccT>(temp_output[bi % out_dims[0]][fi % out_dims[1]][wi % out_dims[2]]
                                                                                            [zi % out_dims[3]][yi % out_dims[4]][xi % out_dims[5]]);
                                            temp_output[bi % out_dims[0]][fi % out_dims[1]][wi % out_dims[2]]
                                                       [zi % out_dims[3]][yi % out_dims[4]][xi % out_dims[5]] = reduce.get(current_acc_val, max_counter_value, reduce_mode);
                                    }
                                }
            }

        previous = std::move(temp_output);
        temp_dims = {previous.size(),                 // b
                     previous[0].size(),              // f
                     previous[0][0].size(),           // w
                     previous[0][0][0].size(),        // z
                     previous[0][0][0][0].size(),     // y
                     previous[0][0][0][0][0].size(),  // x
        };
    }

    VVVVVVF<AccT> output;

    if (keepDims) {
        output = std::move(previous);
    } else {
        std::vector<size_t> actual_dims = temp_dims;
        std::vector<char> coords = {'b', 'f', 'w', 'z', 'y', 'x'};
        std::map<char, int, Comparator<char>> remap_coords = reduce.create_coords_map(coords);
        reduce.remap(actual_dims, remap_coords, reduce_axis, axes_map, coords, dims);

        VVVVVVF<AccT>actual_output(actual_dims[0],
                                   VVVVVF<AccT>(actual_dims[1],
                                   VVVVF<AccT>(actual_dims[2],
                                   VVVF<AccT>(actual_dims[3],
                                   VVF<AccT>(actual_dims[4],
                                   VF<AccT>(actual_dims[5], 0))))));

        for (size_t bi = 0; bi < previous.size(); ++bi)
            for (size_t fi = 0; fi < previous[0].size(); ++fi)
                for (size_t wi = 0; wi < previous[0][0].size(); ++wi)
                    for (size_t zi = 0; zi < previous[0][0][0].size(); ++zi)
                        for (size_t yi = 0; yi < previous[0][0][0][0].size(); ++yi)
                            for (size_t xi = 0; xi < previous[0][0][0][0][0].size(); ++xi) {
                                std::vector<size_t> coords = {bi, fi, wi, zi, yi, xi};
                                actual_output[coords.at(remap_coords['b'])][coords.at(remap_coords['f'])]
                                             [coords.at(remap_coords['w'])][coords.at(remap_coords['z'])]
                                             [coords.at(remap_coords['y'])][coords.at(remap_coords['x'])] = previous[bi][fi][wi][zi][yi][xi];
                            }

        output = std::move(actual_output);
    }

    VVVVVVF<OutputT> final_output(output.size(),
                                  VVVVVF<OutputT>(output[0].size(),
                                  VVVVF<OutputT>(output[0][0].size(),
                                  VVVF<OutputT>(output[0][0][0].size(),
                                  VVF<OutputT>(output[0][0][0][0].size(),
                                  VF<OutputT>(output[0][0][0][0][0].size(), 0))))));

    for (size_t bi = 0; bi < output.size(); ++bi)
        for (size_t fi = 0; fi < output[0].size(); ++fi)
            for (size_t wi = 0; wi < output[0][0].size(); ++wi)
                for (size_t zi = 0; zi < output[0][0][0].size(); ++zi)
                    for (size_t yi = 0; yi < output[0][0][0][0].size(); ++yi)
                        for (size_t xi = 0; xi < output[0][0][0][0][0].size(); ++xi)
                            final_output[bi][fi][wi][zi][yi][xi] = static_cast<OutputT>(output[bi][fi][wi][zi][yi][xi]);

    return final_output;
}

using TestParamType_general_reduce_gpu = ::testing::tuple<int, int, int,          // 0, 1, 2  -  b, f, w
                                                          int, int, int,          // 3, 4, 5  -  z, y, x
                                                          format,                 // 6  -  input_dt format
                                                          reduce_mode,            // 7  -  reduce mode
                                                          std::vector<int64_t>,   // 8  -  reduce axis
                                                          std::string,            // 9  -  kernel name
                                                          bool,                   // 10 -  keepDims
                                                          data_types,             // 11 -  input_dt
                                                          bool,                   // 12 -  force_output_dt
                                                          data_types>;            // 13 -  output_dt

 struct general_reduce_gpu : public ::testing::TestWithParam<TestParamType_general_reduce_gpu> {
    static std::string PrintToStringParamName(testing::TestParamInfo<TestParamType_general_reduce_gpu> param_info) {
        const std::vector<int64_t> reduce_axes = testing::get<8>(param_info.param);
        std::string string_axes;
        for (auto& axis : reduce_axes) string_axes += std::to_string(axis) + "_";

        // Readable name
        return "in_b_" + std::to_string(testing::get<0>(param_info.param)) +
               "_f_" + std::to_string(testing::get<1>(param_info.param)) +
               "_w_" + std::to_string(testing::get<2>(param_info.param)) +
               "_z_" + std::to_string(testing::get<3>(param_info.param)) +
               "_y_" + std::to_string(testing::get<4>(param_info.param)) +
               "_x_" + std::to_string(testing::get<5>(param_info.param)) +
               "_format_" + std::to_string(testing::get<6>(param_info.param)) +
               "_reduce_mode_" + std::to_string(static_cast<std::underlying_type<cldnn::reduce_mode>::type>(testing::get<7>(param_info.param))) +
               "_axes_" + string_axes +
               "_kernel_name_" + testing::get<9>(param_info.param) +
               "_keepDims_" + std::to_string(testing::get<10>(param_info.param));
    }
};

template <data_types InputT>
struct input_data_type {
    using type = float;
};

template <>
struct input_data_type <data_types::f16> {
    using type = ov::float16;
};

template <>
struct input_data_type <data_types::i8> {
    using type = int8_t;
};

template <>
struct input_data_type <data_types::u8> {
    using type = uint8_t;
};

template <data_types OutputT>
struct output_data_type {
    using type = float;
};

template <>
struct output_data_type<data_types::f16> {
    using type = ov::float16;
};

template <>
struct output_data_type<data_types::i8> {
    using type = int8_t;
};

template <>
struct output_data_type<data_types::u8> {
    using type = uint8_t;
};

template <data_types InputT, data_types OutputT>
class ReduceTestBase : public ::testing::TestWithParam<TestParamType_general_reduce_gpu> {
protected:
    cldnn::engine& engine = get_test_engine();
    int batch_num, input_f, input_w, input_z, input_y, input_x;
    cldnn::format input_format = format::any;
    cldnn::reduce_mode reduce_mode;
    std::vector<int64_t> reduce_axis;
    std::string kernel_name;
    bool keep_dims;
    cldnn::data_types input_dt;
    cldnn::data_types output_dt;
    bool force_output_dt;
    // cldnn::impl_types impl_type;

    static std::vector<std::tuple<cldnn::reduce_mode,double, double, double>> perf_data;

    tests::random_generator rg;
    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    ReduceTestBase() {
        this->batch_num = testing::get<0>(GetParam());
        this->input_f = testing::get<1>(GetParam());
        this->input_w = testing::get<2>(GetParam());
        this->input_z = testing::get<3>(GetParam());
        this->input_y = testing::get<4>(GetParam());
        this->input_x = testing::get<5>(GetParam());
        this->input_format = testing::get<6>(GetParam());
        this->reduce_mode = testing::get<7>(GetParam());
        this->reduce_axis = testing::get<8>(GetParam());
        this->kernel_name = testing::get<9>(GetParam());
        this->keep_dims = testing::get<10>(GetParam());
        this->input_dt = testing::get<11>(GetParam());
        this->force_output_dt = testing::get<12>(GetParam());
        this->output_dt = testing::get<13>(GetParam());
    }

public:
    void execute(bool is_caching_test) {
        if (engine.get_device_info().supports_immad)
            return;
        int input_dim = static_cast<int>(input_format.dimension());
        cldnn::format layout_format = input_format;

        if (input_dim == 4)
            layout_format = format::bfyx;
        else if (input_dim == 5)
            layout_format = format::bfzyx;
        else
            layout_format = format::bfwzyx;

        using input_t = typename input_data_type<InputT>::type;
        using output_t = typename output_data_type<OutputT>::type;

        auto input_size = tensor(batch(batch_num), feature(input_f), spatial(input_x, input_y, input_z, input_w));
        auto input_data = rg.generate_random_6d<input_t>(batch_num, input_f, input_x, input_y, input_z, input_w, 1, 10);
        auto input_lay = layout(input_dt, layout_format, input_size);
        auto input_mem = engine.allocate_memory(input_lay);

        {
            cldnn::mem_lock<input_t> input_ptr(input_mem, get_test_stream());
            for (int fi = 0; fi < input_f; fi++)
                for (int wi = 0; wi < input_w; wi++)
                    for (int zi = 0; zi < input_z; zi++)
                        for (int yi = 0; yi < input_y; yi++)
                            for (int xi = 0; xi < input_x; xi++) {
                                for (int bi = 0; bi < batch_num; bi++) {
                                    tensor coords = tensor(batch(bi), feature(fi), spatial(xi, yi, zi, wi));
                                    size_t offset = input_lay.get_linear_offset(coords);
                                    input_ptr[offset] = input_data[bi][fi][xi][yi][zi][wi];
                                }
                            }
        }

        auto reference_result = reference_reduce(input_data, reduce_mode, reduce_axis, batch_num,
                                                 input_f, input_w, input_z, input_y,
                                                 input_x, input_dim, keep_dims);
        topology topology;
        auto red = reduce("reduce", input_info("input"), reduce_mode, reduce_axis, keep_dims);
        if (force_output_dt) {
            red.output_data_types = {output_dt};
        }
        topology.add(input_layout("input", input_mem->get_layout()));
        topology.add(red);
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        ov::intel_gpu::ImplementationDesc reduce_impl = {input_format, kernel_name};
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"reduce", reduce_impl}}));
        cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input_mem);
        auto outputs = network->execute();

        auto out_mem = outputs.at("reduce").get_memory();
        cldnn::mem_lock<output_t, mem_lock_type::read> out_ptr(out_mem, get_test_stream());
        auto out_lay = out_mem->get_layout();

        ASSERT_EQ(out_lay.get_tensor().sizes()[0], reference_result.size());                 // b
        ASSERT_EQ(out_lay.get_tensor().sizes()[1], reference_result[0].size());              // f
        ASSERT_EQ(out_lay.spatial(3), reference_result[0][0].size());           // w
        ASSERT_EQ(out_lay.spatial(2), reference_result[0][0][0].size());        // z
        ASSERT_EQ(out_lay.spatial(1), reference_result[0][0][0][0].size());     // y
        ASSERT_EQ(out_lay.spatial(0), reference_result[0][0][0][0][0].size());  // x

        for (size_t bi = 0; bi < reference_result.size(); bi++)
            for (size_t fi = 0; fi < reference_result[0].size(); fi++)
                for (size_t wi = 0; wi < reference_result[0][0].size(); wi++)
                    for (size_t zi = 0; zi < reference_result[0][0][0].size(); zi++)
                        for (size_t yi = 0; yi < reference_result[0][0][0][0].size(); yi++) {
                            for (size_t xi = 0; xi < reference_result[0][0][0][0][0].size(); xi++) {
                                tensor coords = tensor(batch(bi), feature(fi), spatial(xi, yi, zi, wi));
                                size_t offset = out_lay.get_linear_offset(coords);
                                auto val = out_ptr[offset];
                                auto val_ref = static_cast<output_t>(reference_result[bi][fi][wi][zi][yi][xi]);
                                auto equal = are_equal(val_ref, val, 1e-1f);

                                if (!equal)
                                    std::cout << "Reference value at batch: " << bi << " output_f: " << fi
                                              << " y: " << yi << " x: " << xi << " = " << val_ref << " Val = " << val
                                              << std::endl;

                                ASSERT_TRUE(equal);

                                if (!equal)
                                    break;
                            }
                        }
    }
};

class general_reduce_gpu_i8_i8 : public ReduceTestBase<data_types::i8, data_types::i8> {};
TEST_P(general_reduce_gpu_i8_i8, base) { execute(false); }

class general_reduce_gpu_i8_f32 : public ReduceTestBase<data_types::i8, data_types::f32> {};
TEST_P(general_reduce_gpu_i8_f32, base) { execute(false); }

class general_reduce_gpu_f32_f32 : public ReduceTestBase<data_types::f32, data_types::f32> {};
TEST_P(general_reduce_gpu_f32_f32, base) { execute(false); }

INSTANTIATE_TEST_SUITE_P(reduce_gpu_b_fs_yx_fsv16_i8_i8,
                        general_reduce_gpu_i8_i8,
                        ::testing::Values(
                            TestParamType_general_reduce_gpu(2, 12, 1, 1, 3, 2, format::b_fs_yx_fsv16, reduce_mode::logical_or, {2, 3}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, true, data_types::i8),
                            TestParamType_general_reduce_gpu(2, 3, 1, 1, 8, 5, format::b_fs_yx_fsv16, reduce_mode::logical_and, {0, 2}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, true, data_types::i8),
                            TestParamType_general_reduce_gpu(3, 3, 1, 1, 3, 6, format::b_fs_yx_fsv16, reduce_mode::logical_or, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, true, data_types::i8),
                            TestParamType_general_reduce_gpu(3, 5, 1, 1, 3, 2, format::b_fs_yx_fsv16, reduce_mode::logical_and, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, true, data_types::i8),
                            TestParamType_general_reduce_gpu(3, 7, 1, 1, 3, 2, format::b_fs_yx_fsv16, reduce_mode::logical_or, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, true, data_types::i8),
                            TestParamType_general_reduce_gpu(1, 3, 1, 1, 6, 12, format::b_fs_yx_fsv16, reduce_mode::logical_and, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, true, data_types::i8),
                            TestParamType_general_reduce_gpu(3, 3, 1, 1, 2, 11, format::b_fs_yx_fsv16, reduce_mode::min, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::i8),
                            TestParamType_general_reduce_gpu(16, 4, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::max, {1, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::i8),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::min, {2, 3}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::i8),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 18, 11, format::b_fs_yx_fsv16, reduce_mode::max, {2, 1}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::i8),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 17, 8, format::b_fs_yx_fsv16, reduce_mode::min, {2, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::i8),
                            TestParamType_general_reduce_gpu(16, 4, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::max, {1, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::i8),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::min, {2, 3}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::i8),
                            TestParamType_general_reduce_gpu(2, 5, 1, 1, 3, 3, format::b_fs_yx_fsv16, reduce_mode::max, {3, 1}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::i8),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 18, 11, format::b_fs_yx_fsv16, reduce_mode::max, {2, 1}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::i8),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 17, 8, format::b_fs_yx_fsv16, reduce_mode::min, {2, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::i8)
                            ),
                            general_reduce_gpu::PrintToStringParamName);

 INSTANTIATE_TEST_SUITE_P(reduce_gpu_b_fs_yx_fsv16_i8_f32,
                        general_reduce_gpu_i8_f32,
                        ::testing::Values(
                            TestParamType_general_reduce_gpu(3, 3, 1, 1, 3, 2, format::b_fs_yx_fsv16, reduce_mode::sum, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(3, 3, 1, 1, 3, 3, format::b_fs_yx_fsv16, reduce_mode::l1, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 3, 1, 1, 13, 11, format::b_fs_yx_fsv16, reduce_mode::mean, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(26, 12, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::sum, {3, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 12, 1, 1, 13, 11, format::b_fs_yx_fsv16, reduce_mode::l1, {2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 13, 12, format::b_fs_yx_fsv16, reduce_mode::l2, {0, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(1, 2, 1, 1, 5, 5, format::b_fs_yx_fsv16, reduce_mode::prod, {3, 1}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 26, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::mean, {3, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 16, 11, format::b_fs_yx_fsv16, reduce_mode::l1, {1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 15, 8, format::b_fs_yx_fsv16, reduce_mode::log_sum_exp, {0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 14, 11, format::b_fs_yx_fsv16, reduce_mode::l2, {1}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 12, 8, format::b_fs_yx_fsv16, reduce_mode::sum_square, {2}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 12, 11, format::b_fs_yx_fsv16, reduce_mode::log_sum, {3}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 3, 1, 1, 13, 11, format::b_fs_yx_fsv16, reduce_mode::mean, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(26, 12, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::sum, {3, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 12, 1, 1, 13, 11, format::b_fs_yx_fsv16, reduce_mode::l1, {2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 13, 9, format::b_fs_yx_fsv16, reduce_mode::l2, {0, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 26, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::mean, {3, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 16, 15, format::b_fs_yx_fsv16, reduce_mode::l1, {1, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 15, 8, format::b_fs_yx_fsv16, reduce_mode::log_sum_exp, {0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 3, 1, 1, 14, 11, format::b_fs_yx_fsv16, reduce_mode::mean, {1}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 12, 8, format::b_fs_yx_fsv16, reduce_mode::sum_square, {2}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 12, 11, format::b_fs_yx_fsv16, reduce_mode::log_sum, {3}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::f32)
                            ),
                            general_reduce_gpu::PrintToStringParamName);

 INSTANTIATE_TEST_SUITE_P(reduce_gpu_b_fs_yx_fsv16_f32_f32,
                        general_reduce_gpu_f32_f32,
                        ::testing::Values(
                            TestParamType_general_reduce_gpu(7, 3, 1, 1, 13, 11, format::b_fs_yx_fsv16, reduce_mode::mean, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(26, 12, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::sum, {3, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 12, 1, 1, 13, 11, format::b_fs_yx_fsv16, reduce_mode::l1, {2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 4, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::max, {1, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 13, 12, format::b_fs_yx_fsv16, reduce_mode::l2, {0, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::min, {2, 3}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(1, 2, 1, 1, 5, 5, format::b_fs_yx_fsv16, reduce_mode::prod, {3, 1}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 26, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::mean, {3, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 18, 11, format::b_fs_yx_fsv16, reduce_mode::max, {2, 1}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 17, 8, format::b_fs_yx_fsv16, reduce_mode::min, {2, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 16, 11, format::b_fs_yx_fsv16, reduce_mode::l1, {1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 15, 8, format::b_fs_yx_fsv16, reduce_mode::log_sum, {0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 14, 11, format::b_fs_yx_fsv16, reduce_mode::l2, {1}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 12, 8, format::b_fs_yx_fsv16, reduce_mode::sum_square, {2}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 12, 11, format::b_fs_yx_fsv16, reduce_mode::log_sum, {3}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),

                            TestParamType_general_reduce_gpu(7, 3, 1, 1, 13, 11, format::b_fs_yx_fsv16, reduce_mode::mean, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(26, 12, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::sum, {3, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 12, 1, 1, 13, 11, format::b_fs_yx_fsv16, reduce_mode::l1, {2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 4, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::max, {1, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 13, 9, format::b_fs_yx_fsv16, reduce_mode::l2, {0, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::min, {2, 3}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(1, 34, 1, 1, 13, 13, format::b_fs_yx_fsv16, reduce_mode::sum, {3, 1}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 26, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::mean, {3, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 18, 11, format::b_fs_yx_fsv16, reduce_mode::max, {2, 1}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 17, 8, format::b_fs_yx_fsv16, reduce_mode::min, {2, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 16, 15, format::b_fs_yx_fsv16, reduce_mode::l1, {1, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 15, 8, format::b_fs_yx_fsv16, reduce_mode::log_sum, {0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 14, 11, format::b_fs_yx_fsv16, reduce_mode::l2, {1}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 12, 8, format::b_fs_yx_fsv16, reduce_mode::sum_square, {2}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 12, 11, format::b_fs_yx_fsv16, reduce_mode::log_sum, {3}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),

                            TestParamType_general_reduce_gpu(1, 2, 1, 1, 12, 12, format::b_fs_yx_fsv16, reduce_mode::sum_square, {3, 2}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(1, 2, 1, 1, 12, 12, format::b_fs_yx_fsv16, reduce_mode::sum_square, {1, 3, 2}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(1, 2, 1, 1, 12, 12, format::b_fs_yx_fsv16, reduce_mode::l2, {3, 2}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(1, 2, 1, 1, 12, 12, format::b_fs_yx_fsv16, reduce_mode::log_sum, {3, 2}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(1, 2, 1, 1, 12, 12, format::b_fs_yx_fsv16, reduce_mode::log_sum_exp, {3, 2}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32)
                        ),
                        general_reduce_gpu::PrintToStringParamName);

 INSTANTIATE_TEST_SUITE_P(reduce_gpu_ref_f32_f32,
                        general_reduce_gpu_f32_f32,
                        ::testing::Values(
                            TestParamType_general_reduce_gpu(2, 4, 4, 5, 8, 8, format::bfwzyx, reduce_mode::mean, {1, 4, 2, 3}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(6, 16, 6, 3, 8, 15, format::bfwzyx, reduce_mode::mean, {5, 4, 1, 0}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(6, 16, 3, 7, 12, 12, format::bfwzyx, reduce_mode::mean, {0, 4, 1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(6, 9, 3, 7, 7, 17, format::bfwzyx, reduce_mode::sum, {5, 1, 0}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(6, 6, 3, 7, 3, 8, format::bfwzyx, reduce_mode::mean, {4, 1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(2, 3, 4, 5, 6, 7, format::bfwzyx, reduce_mode::mean, {1, 4}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 8, 5, 5, 4, 4, format::bfwzyx, reduce_mode::mean, {3}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 8, 5, 5, 3, 6, format::bfwzyx, reduce_mode::mean, {2}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 8, 5, 5, 8, 8, format::bfwzyx, reduce_mode::mean, {4}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 8, 5, 5, 3, 6, format::bfwzyx, reduce_mode::mean, {1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 8, 5, 5, 3, 6, format::bfwzyx, reduce_mode::mean, {5}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 16, 4, 5, 3, 6, format::bfwzyx, reduce_mode::mean, {0}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 8, 2, 4, 2, 5, format::bfwzyx, reduce_mode::mean, {0}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(6, 2, 1, 7, 8, 3, format::bfzyx, reduce_mode::log_sum_exp, {0, 3, 4}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(3, 11, 1, 7, 2, 2, format::bfzyx, reduce_mode::l1, {3, 1, 0}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(8, 4, 1, 7, 2, 2, format::bfzyx, reduce_mode::l2, {1, 3, 4}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(4, 5, 1, 7, 12, 4, format::bfzyx, reduce_mode::l1, {3, 4}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 7, 12, 4, format::bfzyx, reduce_mode::sum, {4, 1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 7, 4, 12, format::bfzyx, reduce_mode::max, {3, 1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 4, 1, 7, 12, 12, format::bfzyx, reduce_mode::min, {3, 4}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(3, 11, 1, 1, 7, 17, format::bfyx, reduce_mode::l1, {1, 0}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(6, 6, 1, 7, 2, 9, format::bfzyx, reduce_mode::l1, {0}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(6, 6, 1, 7, 2, 9, format::bfzyx, reduce_mode::l1, {1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(6, 6, 1, 7, 2, 9, format::bfzyx, reduce_mode::l1, {3}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(6, 6, 1, 7, 2, 9, format::bfzyx, reduce_mode::l1, {4}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(3, 7, 1, 1, 7, 17, format::bfyx, reduce_mode::sum, {0}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(6, 9, 1, 1, 7, 17, format::bfyx, reduce_mode::l2, {1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(5, 5, 1, 1, 17, 17, format::bfyx, reduce_mode::mean, {2}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(3, 12, 1, 1, 7, 17, format::bfyx, reduce_mode::log_sum, {3}, "reduce_ref", false, data_types::f32, false, data_types::f32),

                            TestParamType_general_reduce_gpu(7, 3, 6, 6, 12, 12, format::bfwzyx, reduce_mode::log_sum_exp, {0, 4, 5}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 4, 6, 6, 12, 12, format::bfwzyx, reduce_mode::l1, {4, 1, 0}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 6, 6, 4, 7, format::bfwzyx, reduce_mode::l2, {1, 4, 5}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 16, 6, 6, 7, 12, format::bfwzyx, reduce_mode::l1, {4, 5}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 6, 6, 7, 2, format::bfwzyx, reduce_mode::sum, {5, 1}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(8, 16, 6, 6, 3, 7, format::bfwzyx, reduce_mode::mean, {5, 0}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(2, 16, 6, 6, 12, 3, format::bfwzyx, reduce_mode::max, {4, 1}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 16, 6, 6, 8, 12, format::bfwzyx, reduce_mode::min, {4, 0}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(3, 11, 1, 6, 7, 17, format::bfzyx, reduce_mode::l1, {1, 0}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(3, 7, 1, 6, 7, 3, format::bfzyx, reduce_mode::sum, {0}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(6, 9, 1, 6, 7, 7, format::bfzyx, reduce_mode::l2, {1}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(5, 5, 1, 6, 8, 3, format::bfzyx, reduce_mode::mean, {3}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(3, 8, 1, 6, 7, 3, format::bfzyx, reduce_mode::log_sum, {4}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 12, 4, format::bfyx, reduce_mode::mean, {3, 2, 1, 0}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 4, 1, 1, 12, 12, format::bfyx, reduce_mode::mean, {0, 2, 1}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(4, 3, 1, 1, 7, 12, format::bfyx, reduce_mode::sum, {3, 1, 0}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 4, 11, format::bfyx, reduce_mode::mean, {2, 1}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(2, 16, 1, 1, 3, 6, format::bfyx, reduce_mode::mean, {1, 2}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 4, 1, 1, 3, 6, format::bfyx, reduce_mode::mean, {3}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 32, 1, 1, 3, 6, format::bfyx, reduce_mode::mean, {0}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 4, 1, 1, 12, 15, format::bfyx, reduce_mode::mean, {0}, "reduce_ref", true, data_types::f32, false, data_types::f32)
                        ), general_reduce_gpu::PrintToStringParamName);

INSTANTIATE_TEST_SUITE_P(DISABLED_reduce_gpu_ref_f32_f32,
                        general_reduce_gpu_f32_f32,
                        ::testing::Values(
                            TestParamType_general_reduce_gpu(1, 7, 1, 1, 4, 3,format::bfyx, reduce_mode::mean, {0, 1, 2}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(3, 7, 7, 6, 4, 3, format::bfwzyx, reduce_mode::l1, {0, 5, 4}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(1, 5, 7, 6, 4, 3, format::bfwzyx, reduce_mode::l2, {5, 2}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(1, 2, 1, 2, 4, 3, format::bfzyx, reduce_mode::prod, {2, 1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(2, 7, 1, 1, 4, 3, format::fyxb, reduce_mode::sum, {0, 1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(1, 7, 1, 1, 4, 3, format::bfyx, reduce_mode::max, {2, 1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(12, 2, 1, 1, 4, 3, format::yxfb, reduce_mode::min, {2, 0}, "reduce_ref" ,false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(1, 7, 1, 1, 4, 3, format::bfyx, reduce_mode::mean, {0, 3, 1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 7, 1, 1, 4, 3, format::b_fs_yx_fsv4, reduce_mode::l2, {1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(9, 7, 1, 1, 4, 3, format::b_fs_yx_fsv16, reduce_mode::prod, {3, 1, 0}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(9, 9, 1, 1, 4, 3, format::b_fs_yx_fsv32, reduce_mode::l1, {0, 1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(11, 10, 1, 1, 4, 3, format::bs_fs_yx_bsv16_fsv16, reduce_mode::max, {0}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(12, 7, 1, 1, 4, 3, format::fs_b_yx_fsv32, reduce_mode::l2, {3, 1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(13, 7, 1, 1, 4, 3, format::bs_fs_zyx_bsv16_fsv16, reduce_mode::sum, {1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(14, 7, 1, 1, 4, 3, format::b_fs_zyx_fsv16, reduce_mode::sum, {1}, "reduce_ref", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(18, 7, 1, 1, 4, 3,  format::b_fs_zyx_fsv32, reduce_mode::sum, {4}, "reduce_ref", false, data_types::f32, false, data_types::f32),

                            TestParamType_general_reduce_gpu(5, 7, 1, 1, 4, 3, format::bfwzyx, reduce_mode::sum, {5, 2}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(6, 7, 1, 1, 4, 3, format::bfzyx, reduce_mode::max, {2, 1}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(3, 7, 1, 1, 4, 3, format::fyxb, reduce_mode::min, {3, 1}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(11, 7, 1, 1, 4, 3, format::byxf, reduce_mode::prod, {3, 1}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(12, 7, 1, 1, 4, 3, format::bfyx, reduce_mode::l1, {3, 1}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(13, 7, 1, 1, 4, 3, format::yxfb, reduce_mode::sum, {1, 0}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(14, 7, 1, 1, 4, 3, format::bfyx, reduce_mode::l2, {3, 1}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(9, 7, 1, 1, 4, 3, format::b_fs_yx_fsv4, reduce_mode::log_sum, {1}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(8, 7, 1, 1, 4, 3, format::b_fs_yx_fsv16, reduce_mode::log_sum_exp, {3, 1, 0}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 7, 1, 1, 4, 3, format::b_fs_yx_fsv32, reduce_mode::l1, {0, 1}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(6, 7, 1, 1, 4, 3, format::bs_fs_yx_bsv16_fsv16, reduce_mode::max, {0}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(1, 12, 1, 1, 4, 3, format::fs_b_yx_fsv32, reduce_mode::l2, {3, 1}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(3, 7, 1, 1, 4, 3, format::bs_fs_zyx_bsv16_fsv16, reduce_mode::sum, {1}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(1, 7, 1, 1, 4, 12, format::b_fs_zyx_fsv16, reduce_mode::sum, {1}, "reduce_ref", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(1, 7, 1, 1, 8, 3, format::b_fs_zyx_fsv32, reduce_mode::sum, {4}, "reduce_ref", true, data_types::f32, false, data_types::f32)
                        ),
                        general_reduce_gpu::PrintToStringParamName);

template <typename T>
void test_common_bfyx(bool is_caching_test) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 1, 1, 1}});

    set_values(input, {1.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::sum, {0}, 0));

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input);

    auto outputs = network->execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<T> ref_data = {1.0f};

    cldnn::mem_lock<T, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfyx) {
    test_common_bfyx<float>(false);
}

TEST(reduce_gpu, common_bfyx_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 3, 4, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::sum, {3, 2}, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {6.0f, 22.0f, 38.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, regr_bfyx_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, {1, 3, 2, 2} });

    set_values(input, { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::sum, { 0, 3 }, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = { 1.0f, 5.0f, 9.0f, 13.0f, 17.0f, 21.0f };

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfzyx) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfzyx, {1, 1, 1, 1, 1}});

    set_values(input, {1.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::sum, {0}, 0));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {1.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfzyx_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfzyx, {1, 1, 1, 1, 1}});

    set_values(input, {1.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::sum, {0}, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {1.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, tensor(format::bfwzyx, {1, 3, 4, 1, 1, 1})});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::sum, {2, 3, 4, 5}, 0));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {6.0f, 22.0f, 38.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, tensor(format::bfwzyx, {1, 3, 4, 1, 1, 1})});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::sum, {1, 2, 3}, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {66.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_max_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 4, 1, 1, 1}});

    set_values(input, {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
                       12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::max, {0, 1}, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {20.0f, 21.0f, 22.0f, 23.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_min) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::min, {1, 2}, 0));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {0.0f, 3.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_min_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::min, {1, 2}, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {0.0f, 3.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_mean) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::mean, {1, 2}, 0));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {1.0f, 4.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_mean_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::mean, {1, 2}, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {1.0f, 4.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_prod) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::prod, {1, 2}, 0));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {0.0f, 60.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_prod_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::prod, {1, 2}, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {0.0f, 60.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_sum_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 4, 1, 1, 1}});

    set_values(input, {0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
                       12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::sum, {0, 1}, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {60.0f, 66.0f, 72.0f, 78.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_logical_and) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::logical_and, {1, 2}, 0));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<char> ref_data = {0, 1};

    cldnn::mem_lock<char, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_logical_and_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::logical_and, {1, 2}, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<char> ref_data = {0, 1};

    cldnn::mem_lock<char, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_logical_or) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::logical_or, {1, 2}, 0));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<char> ref_data = {1, 1};

    cldnn::mem_lock<char, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_logical_or_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::logical_or, {1, 2}, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<char> ref_data = {1, 1};

    cldnn::mem_lock<char, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_sum_square) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::sum_square, {1, 2}, 0));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {5.0f, 50.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_sum_square_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::sum_square, {1, 2}, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {5.0f, 50.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_l1) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, -2.0f, 3.0f, 4.0f, -5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::l1, {1, 2}, 0));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {3.0f, 12.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_l1_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, -2.0f, 3.0f, 4.0f, -5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::l1, {1, 2}, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {3.0f, 12.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_l2) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, -2.0f, 3.0f, 4.0f, -5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::l2, {1, 2}, 0));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {2.236067977f, 7.071067812f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_l2_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, -2.0f, 3.0f, 4.0f, -5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::l2, {1, 2}, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {2.236067977f, 7.071067812f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_log_sum) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::log_sum, {1, 2}, 0));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {1.0986122887f, 2.4849066498f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_log_sum_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::log_sum, {1, 2}, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {1.0986122887f, 2.4849066498f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_log_sum_exp) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::log_sum_exp, {1, 2}, 0));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {2.407605964f, 5.407605964f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, common_bfwzyx_log_sum_exp_keepdims) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::log_sum_exp, {1, 2}, 1));

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {2.407605964f, 5.407605964f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, cpu_impl_int32) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({{4}, data_types::i32, format::bfyx});

    set_values<int32_t>(input, {1, 2, 3, 4});

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::prod, {0}, true));

    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"reduce", {format::bfyx, "", impl_types::cpu}}}));
    network network(engine, topology, config);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<int32_t> ref_data = {24};

    cldnn::mem_lock<int32_t, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_EQ(ref_data[i], output_ptr[i]);
    }
}

TEST(reduce_gpu, dynamic) {
    auto& engine = get_test_engine();
    auto input = engine.allocate_memory({data_types::f32, format::bfwzyx, {2, 3, 1, 1, 1, 1}});

    layout in_dyn_layout { ov::PartialShape::dynamic(6), data_types::f32, format::bfwzyx };

    set_values(input, {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f});

    topology topology;
    topology.add(input_layout("input", in_dyn_layout));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::prod, {1, 2}, 1));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("reduce");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {0.0f, 60.0f};
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, b_fs_yx_fsv16_min_dynamic) {
    auto& engine = get_test_engine();
    if (engine.get_device_info().supports_immad)
            return;
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 17, 1, 2}});

    set_values(input, {
        1.0f, -1.0f,
        2.0f, -2.0f,
        3.0f, -3.0f,
        4.0f, -4.0f,
        5.0f, -5.0f,
        6.0f, -6.0f,
        7.0f, -7.0f,
        8.0f, -8.0f,
        9.0f, -9.0f,
        8.0f, -8.0f,
        7.0f, -7.0f,
        6.0f, -6.0f,
        5.0f, -5.0f,
        4.0f, -4.0f,
        3.0f, -3.0f,
        2.0f, -2.0f,
        1.0f, -1.0f
    });

    topology topology;
    auto in_layout = layout(ov::PartialShape::dynamic(4), data_types::f32, format::bfyx);
    const auto used_layout = layout({1, 17, 1, 2}, data_types::f32, format::b_fs_yx_fsv16);

    topology.add(input_layout("input", in_layout));
    topology.add(reorder("reorder", input_info("input"), used_layout));
    topology.add(reduce("reduce", input_info("reorder"), reduce_mode::min, {1}, 0));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {1.0f, -9.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, b_fs_yx_fsv16_max_dynamic) {
    auto& engine = get_test_engine();
    if (engine.get_device_info().supports_immad)
            return;
    auto input = engine.allocate_memory({data_types::f32, format::bfyx, {1, 17, 1, 2}});

    set_values(input, {
        1.0f, -1.0f,
        2.0f, -2.0f,
        3.0f, -3.0f,
        4.0f, -4.0f,
        5.0f, -5.0f,
        6.0f, -6.0f,
        7.0f, -7.0f,
        8.0f, -8.0f,
        9.0f, -9.0f,
        8.0f, -8.0f,
        7.0f, -7.0f,
        6.0f, -6.0f,
        5.0f, -5.0f,
        4.0f, -4.0f,
        3.0f, -3.0f,
        2.0f, -2.0f,
        1.0f, -1.0f
    });

    topology topology;
    auto in_layout = layout(ov::PartialShape::dynamic(4), data_types::f32, format::bfyx);
    const auto used_layout = layout({1, 17, 1, 2}, data_types::f32, format::b_fs_yx_fsv16);

    topology.add(input_layout("input", in_layout));
    topology.add(reorder("reorder", input_info("input"), used_layout));
    topology.add(reduce("reduce", input_info("reorder"), reduce_mode::max, {1}, 0));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    network network(engine, topology, config);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output = outputs.at("reduce").get_memory();

    std::vector<float> ref_data = {9.0f, -1.0f};

    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_data.size(); ++i) {
        ASSERT_TRUE(are_equal(ref_data[i], output_ptr[i]));
    }
}

TEST(reduce_gpu, reduce_min_max_default_output_element_type_should_be_same_to_input_element_type) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({data_types::i8, format::bfyx, {1, 1, 2, 2}});
    set_values<int8_t>(input, {1, 1, 1, 1});

    topology topology1(
        input_layout("input", input->get_layout()),
        reduce("reduce", input_info("input"), reduce_mode::max, {1}, 0),
        reorder("reorder", input_info("reduce"), format::bfyx, data_types::i8)
    );

    ExecutionConfig config = get_test_default_config(engine);
    network network1(engine, topology1, config);
    network1.set_input_data("input", input);
    auto output = network1.execute();

    ASSERT_EQ(network1.get_program()->get_node("reduce").get_output_layout().data_type, data_types::i8);

    topology topology2(
        input_layout("input", input->get_layout()),
        reduce("reduce", input_info("input"), reduce_mode::min, {1}, 0),
        reorder("reorder", input_info("reduce"), format::bfyx, data_types::i8)
    );

    network network2(engine, topology2, config);
    network2.set_input_data("input", input);
    output = network2.execute();

    ASSERT_EQ(network2.get_program()->get_node("reduce").get_output_layout().data_type, data_types::i8);
}

template <data_types InputT, data_types OutputT>
class ReduceXYWithBigTensorTestBase : public ::testing::TestWithParam<TestParamType_general_reduce_gpu> {
protected:
    cldnn::engine& engine = get_test_engine();
    int batch_num, input_f, input_w, input_z, input_y, input_x;
    cldnn::format input_format = format::any;
    cldnn::reduce_mode reduce_mode;
    std::vector<int64_t> reduce_axis;
    std::string kernel_name;
    bool keep_dims;
    cldnn::data_types input_dt;
    cldnn::data_types output_dt;
    bool force_output_dt;

    static std::vector<std::tuple<cldnn::reduce_mode,double, double, double>> perf_data;

    tests::random_generator rg;
    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    ReduceXYWithBigTensorTestBase() {
        this->batch_num = testing::get<0>(GetParam());
        this->input_f = testing::get<1>(GetParam());
        this->input_w = testing::get<2>(GetParam());
        this->input_z = testing::get<3>(GetParam());
        this->input_y = testing::get<4>(GetParam());
        this->input_x = testing::get<5>(GetParam());
        this->input_format = testing::get<6>(GetParam());
        this->reduce_mode = testing::get<7>(GetParam()); // not used
        this->reduce_axis = testing::get<8>(GetParam());
        this->kernel_name = testing::get<9>(GetParam());
        this->keep_dims = testing::get<10>(GetParam());
        this->input_dt = testing::get<11>(GetParam());
        this->force_output_dt = testing::get<12>(GetParam());
        this->output_dt = testing::get<13>(GetParam());
    }

public:
    void execute(bool is_caching_test) {

        int input_dim = static_cast<int>(input_format.dimension());
        cldnn::format layout_format = input_format;

        if (input_dim == 4)
            layout_format = format::bfyx;
        else if (input_dim == 5)
            layout_format = format::bfzyx;
        else
            layout_format = format::bfwzyx;

        using input_t = typename input_data_type<InputT>::type;
        using output_t = typename output_data_type<OutputT>::type;

        auto input_size = tensor(batch(batch_num), feature(input_f), spatial(input_x, input_y, input_z, input_w));
        auto input_data = rg.generate_random_6d<input_t>(batch_num, input_f, input_x, input_y, input_z, input_w, 1, 5, 9);
        auto input_lay = layout(input_dt, layout_format, input_size);
        auto input_mem = engine.allocate_memory(input_lay);

        {
            cldnn::mem_lock<input_t> input_ptr(input_mem, get_test_stream());

            for (int fi = 0; fi < input_f; fi++)
                for (int wi = 0; wi < input_w; wi++)
                    for (int zi = 0; zi < input_z; zi++)
                        for (int yi = 0; yi < input_y; yi++)
                            for (int xi = 0; xi < input_x; xi++) {
                                for (int bi = 0; bi < batch_num; bi++) {
                                    tensor coords = tensor(batch(bi), feature(fi), spatial(xi, yi, zi, wi));
                                    size_t offset = input_lay.get_linear_offset(coords);
                                    input_ptr[offset] = input_data[bi][fi][xi][yi][zi][wi];
                                }
                            }
        }

        std::vector<cldnn::reduce_mode> modes {
            cldnn::reduce_mode::max,
            cldnn::reduce_mode::min,
            cldnn::reduce_mode::mean,
            // reduce_mode::prod,
            cldnn::reduce_mode::sum,
            cldnn::reduce_mode::logical_and,
            cldnn::reduce_mode::logical_or,
            // reduce_mode::sum_square,
            cldnn::reduce_mode::l1,
            // reduce_mode::l2,
            // reduce_mode::log_sum,
            cldnn::reduce_mode::log_sum_exp
        };

        for (auto& target_mode : modes)
        {
            auto reference_result = reference_reduce(input_data, target_mode, reduce_axis, batch_num,
                                                    input_f, input_w, input_z, input_y,
                                                    input_x, input_dim, keep_dims);

            topology topology;
            auto red = reduce("reduce", input_info("input"), target_mode, reduce_axis, keep_dims);
            if (force_output_dt) {
                red.output_data_types = {output_dt};
            }
            topology.add(input_layout("input", input_mem->get_layout()));
            topology.add(red);
            ExecutionConfig config = get_test_default_config(engine);
            config.set_property(ov::intel_gpu::optimize_data(true));
            ov::intel_gpu::ImplementationDesc reduce_impl = {input_format, kernel_name, impl_types::ocl};
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"reduce", reduce_impl}}));
            cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), is_caching_test);
            network->set_input_data("input", input_mem);

            auto outputs = network->execute();

            auto out_mem = outputs.at("reduce").get_memory();
            cldnn::mem_lock<output_t, mem_lock_type::read> out_ptr(out_mem, get_test_stream());
            auto out_lay = out_mem->get_layout();

            ASSERT_EQ(out_lay.get_tensor().sizes()[0], reference_result.size());                 // b
            ASSERT_EQ(out_lay.get_tensor().sizes()[1], reference_result[0].size());              // f
            ASSERT_EQ(out_lay.spatial(3), reference_result[0][0].size());           // w
            ASSERT_EQ(out_lay.spatial(2), reference_result[0][0][0].size());        // z
            ASSERT_EQ(out_lay.spatial(1), reference_result[0][0][0][0].size());     // y
            ASSERT_EQ(out_lay.spatial(0), reference_result[0][0][0][0][0].size());  // x

            bool need_adjust_threshold = (typeid(output_t) == typeid(output_data_type<data_types::i8>::type));
            for (size_t bi = 0; bi < reference_result.size(); bi++)
                for (size_t fi = 0; fi < reference_result[0].size(); fi++)
                    for (size_t wi = 0; wi < reference_result[0][0].size(); wi++)
                        for (size_t zi = 0; zi < reference_result[0][0][0].size(); zi++)
                            for (size_t yi = 0; yi < reference_result[0][0][0][0].size(); yi++) {
                                for (size_t xi = 0; xi < reference_result[0][0][0][0][0].size(); xi++) {
                                    tensor coords = tensor(batch(bi), feature(fi), spatial(xi, yi, zi, wi));
                                    size_t offset = out_lay.get_linear_offset(coords);
                                    auto val = out_ptr[offset];
                                    auto val_ref = static_cast<output_t>(reference_result[bi][fi][wi][zi][yi][xi]);
                                    bool equal = need_adjust_threshold ?
                                        are_equal(val_ref, val, 1e-1f, 1.0f, 10.0f) : are_equal(val_ref, val, 1e-1f);

                                    if (!equal)
                                        std::cout << "Reduce mode: " << (int)target_mode << ", "
                                                << "Reference value at batch: " << bi << " output_f: " << fi
                                                << " y: " << yi << " x: " << xi << " = " << val_ref << " Val = " << val
                                                << std::endl;

                                    ASSERT_TRUE(equal);

                                    if (!equal)
                                        break;
                                }
                            }
        }
    }
};


class general_reduce_gpu_xy_f32 : public ReduceXYWithBigTensorTestBase<data_types::f32, data_types::f32> {};
TEST_P(general_reduce_gpu_xy_f32, base) { execute(false); }

class general_reduce_gpu_xy_i8 : public ReduceXYWithBigTensorTestBase<data_types::i8, data_types::i8> {};
TEST_P(general_reduce_gpu_xy_i8, base) { execute(false); }

INSTANTIATE_TEST_SUITE_P(reduce_gpu_b_fs_yx_fsv16_xy_f32,
                        general_reduce_gpu_xy_f32,
                        ::testing::Values(
                            TestParamType_general_reduce_gpu(1, 32, 1, 1,  18,  18, format::b_fs_yx_fsv16, reduce_mode::max, {3, 2}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, true, data_types::f32),
                            TestParamType_general_reduce_gpu(1, 32, 1, 1, 256, 256, format::b_fs_yx_fsv16, reduce_mode::max, {3, 2}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, true, data_types::f32)
                        ),
                        general_reduce_gpu::PrintToStringParamName);

INSTANTIATE_TEST_SUITE_P(reduce_gpu_b_fs_yx_fsv16_xy_i8,
                        general_reduce_gpu_xy_i8,
                        ::testing::Values(
                            TestParamType_general_reduce_gpu(1, 32, 1, 1,  18,  18, format::b_fs_yx_fsv16, reduce_mode::max, {3, 2}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, true, data_types::i8),
                            TestParamType_general_reduce_gpu(1, 32, 1, 1, 256, 256, format::b_fs_yx_fsv16, reduce_mode::max, {3, 2}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, true, data_types::i8)
                        ),
                        general_reduce_gpu::PrintToStringParamName);

#ifdef ENABLE_ONEDNN_FOR_GPU
template <data_types InputT, data_types OutputT>
class ReduceOnednnTestBase : public ::testing::TestWithParam<TestParamType_general_reduce_gpu> {
protected:
    cldnn::engine& engine = get_test_engine();
    int batch_num, input_f, input_w, input_z, input_y, input_x;
    cldnn::format input_format = format::any;
    cldnn::reduce_mode reduce_mode;
    std::vector<int64_t> reduce_axis;
    std::string kernel_name;
    bool keep_dims;
    cldnn::data_types input_dt;
    cldnn::data_types output_dt;
    bool force_output_dt;
    tests::random_generator rg;

    static std::vector<std::tuple<cldnn::reduce_mode,double, double, double>> perf_data;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    ReduceOnednnTestBase() {
        this->batch_num = testing::get<0>(GetParam());
        this->input_f = testing::get<1>(GetParam());
        this->input_w = testing::get<2>(GetParam());
        this->input_z = testing::get<3>(GetParam());
        this->input_y = testing::get<4>(GetParam());
        this->input_x = testing::get<5>(GetParam());
        this->input_format = testing::get<6>(GetParam());
        this->reduce_mode = testing::get<7>(GetParam());
        this->reduce_axis = testing::get<8>(GetParam());
        this->kernel_name = testing::get<9>(GetParam());
        this->keep_dims = testing::get<10>(GetParam());
        this->input_dt = testing::get<11>(GetParam());
        this->force_output_dt = testing::get<12>(GetParam());
        this->output_dt = testing::get<13>(GetParam());
    }

public:
    void execute_onednn() {
        if (!engine.get_device_info().supports_immad)
            return;
        int input_dim = static_cast<int>(input_format.dimension());
        cldnn::format layout_format = input_format;

        if (input_dim == 4)
            layout_format = format::bfyx;
        else if (input_dim == 5)
            layout_format = format::bfzyx;
        else
            layout_format = format::bfwzyx;

        using input_t = typename input_data_type<InputT>::type;
        using output_t = typename output_data_type<OutputT>::type;

        auto input_size = tensor(batch(batch_num), feature(input_f), spatial(input_x, input_y, input_z, input_w));
        auto input_data = rg.generate_random_6d<input_t>(batch_num, input_f, input_x, input_y, input_z, input_w, 1, 10);
        auto input_lay = layout(input_dt, layout_format, input_size);
        auto input_mem = engine.allocate_memory(input_lay);

        {
            cldnn::mem_lock<input_t> input_ptr(input_mem, get_test_stream());
            for (int fi = 0; fi < input_f; fi++)
                for (int wi = 0; wi < input_w; wi++)
                    for (int zi = 0; zi < input_z; zi++)
                        for (int yi = 0; yi < input_y; yi++)
                            for (int xi = 0; xi < input_x; xi++) {
                                for (int bi = 0; bi < batch_num; bi++) {
                                    tensor coords = tensor(batch(bi), feature(fi), spatial(xi, yi, zi, wi));
                                    size_t offset = input_lay.get_linear_offset(coords);
                                    input_ptr[offset] = input_data[bi][fi][xi][yi][zi][wi];
                                }
                            }
        }

        auto reference_result = reference_reduce(input_data, reduce_mode, reduce_axis, batch_num,
                                                 input_f, input_w, input_z, input_y,
                                                 input_x, input_dim, true);
        topology topology;
        auto red = reduce("reduce", input_info("input"), reduce_mode, reduce_axis, true);
        if (force_output_dt) {
            red.output_data_types = {output_dt};
        }
        topology.add(input_layout("input", input_mem->get_layout()));
        topology.add(red);
        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        ov::intel_gpu::ImplementationDesc reduce_impl = {input_format, kernel_name, impl_types::onednn};
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"reduce", reduce_impl}}));
        config.set_property(ov::intel_gpu::queue_type(QueueTypes::in_order));
        network network(engine, topology, config);
        network.set_input_data("input", input_mem);

        auto outputs = network.execute();

        auto out_mem = outputs.at("reduce").get_memory();
        cldnn::mem_lock<output_t, mem_lock_type::read> out_ptr(out_mem, get_test_stream());
        auto out_lay = out_mem->get_layout();

        ASSERT_EQ(out_lay.get_tensor().sizes()[0], reference_result.size());    // b
        ASSERT_EQ(out_lay.get_tensor().sizes()[1], reference_result[0].size()); // f
        ASSERT_EQ(out_lay.spatial(3), reference_result[0][0].size());           // w
        ASSERT_EQ(out_lay.spatial(2), reference_result[0][0][0].size());        // z
        ASSERT_EQ(out_lay.spatial(1), reference_result[0][0][0][0].size());     // y
        ASSERT_EQ(out_lay.spatial(0), reference_result[0][0][0][0][0].size());  // x

        for (size_t bi = 0; bi < reference_result.size(); bi++)
            for (size_t fi = 0; fi < reference_result[0].size(); fi++)
                for (size_t wi = 0; wi < reference_result[0][0].size(); wi++)
                    for (size_t zi = 0; zi < reference_result[0][0][0].size(); zi++)
                        for (size_t yi = 0; yi < reference_result[0][0][0][0].size(); yi++) {
                            for (size_t xi = 0; xi < reference_result[0][0][0][0][0].size(); xi++) {
                                tensor coords = tensor(batch(bi), feature(fi), spatial(xi, yi, zi, wi));
                                size_t offset = out_lay.get_linear_offset(coords);
                                auto val = out_ptr[offset];
                                auto val_ref = static_cast<output_t>(reference_result[bi][fi][wi][zi][yi][xi]);
                                auto equal = are_equal(val_ref, val, 1e-1f);

                                if (!equal)
                                    std::cout << "Reference value at batch: " << bi << " output_f: " << fi
                                              << " y: " << yi << " x: " << xi << " = " << static_cast<float>(val_ref) << " Val = " << static_cast<float>(val)
                                              << std::endl;

                                ASSERT_TRUE(equal);

                                if (!equal)
                                    break;
                            }
                        }
    }
};

class onednn_reduce_gpu_i8_f32 : public ReduceOnednnTestBase<data_types::i8, data_types::f32> {};
TEST_P(onednn_reduce_gpu_i8_f32, base) { execute_onednn(); }

class onednn_reduce_gpu_f16_f16 : public ReduceOnednnTestBase<data_types::f16, data_types::f16> {};
TEST_P(onednn_reduce_gpu_f16_f16, base) { execute_onednn(); }

class onednn_reduce_gpu_f32_f32 : public ReduceOnednnTestBase<data_types::f32, data_types::f32> {};
TEST_P(onednn_reduce_gpu_f32_f32, base) { execute_onednn(); }

INSTANTIATE_TEST_SUITE_P(onednn_reduce_gpu_b_fs_yx_fsv16_i8_f32,
                        onednn_reduce_gpu_i8_f32,
                        ::testing::Values(
                            TestParamType_general_reduce_gpu(3, 3, 1, 1, 3, 2, format::b_fs_yx_fsv16, reduce_mode::sum, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 3, 1, 1, 13, 11, format::b_fs_yx_fsv16, reduce_mode::mean, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::i8, false, data_types::f32),

                            TestParamType_general_reduce_gpu(7, 3, 1, 1, 13, 11, format::b_fs_yx_fsv16, reduce_mode::mean, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 3, 1, 1, 14, 11, format::b_fs_yx_fsv16, reduce_mode::mean, {1}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::i8, false, data_types::f32)
                        ), general_reduce_gpu::PrintToStringParamName);

INSTANTIATE_TEST_SUITE_P(onednn_reduce_gpu_b_fs_yx_fsv16_f16_f16,
                        onednn_reduce_gpu_f16_f16,
                        ::testing::Values(
                            TestParamType_general_reduce_gpu(3, 3, 1, 1, 3, 2, format::b_fs_yx_fsv16, reduce_mode::sum, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(3, 3, 1, 1, 3, 3, format::b_fs_yx_fsv16, reduce_mode::l1, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(3, 3, 1, 1, 2, 11, format::b_fs_yx_fsv16, reduce_mode::min, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(7, 3, 1, 1, 13, 11, format::b_fs_yx_fsv16, reduce_mode::mean, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(16, 4, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::max, {1, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 13, 12, format::b_fs_yx_fsv16, reduce_mode::l2, {0, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::min, {2, 3}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 18, 11, format::b_fs_yx_fsv16, reduce_mode::max, {2, 1}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 16, 11, format::b_fs_yx_fsv16, reduce_mode::l1, {1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 14, 11, format::b_fs_yx_fsv16, reduce_mode::l2, {1}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f16, false, data_types::f16),

                            TestParamType_general_reduce_gpu(7, 3, 1, 1, 13, 11, format::b_fs_yx_fsv16, reduce_mode::mean, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(16, 4, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::max, {1, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 13, 9, format::b_fs_yx_fsv16, reduce_mode::l2, {0, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::min, {2, 3}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 18, 11, format::b_fs_yx_fsv16, reduce_mode::max, {2, 1}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 16, 15, format::b_fs_yx_fsv16, reduce_mode::l1, {1, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(17, 3, 1, 1, 14, 11, format::b_fs_yx_fsv16, reduce_mode::mean, {1}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f16, false, data_types::f16)
                        ), general_reduce_gpu::PrintToStringParamName);

                        INSTANTIATE_TEST_SUITE_P(onednn_reduce_gpu_b_fs_yx_fsv16_f32_f32,
                        onednn_reduce_gpu_f32_f32,
                        ::testing::Values(
                            TestParamType_general_reduce_gpu(3, 3, 1, 1, 3, 2, format::b_fs_yx_fsv16, reduce_mode::sum, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(3, 3, 1, 1, 3, 3, format::b_fs_yx_fsv16, reduce_mode::l1, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(3, 3, 1, 1, 2, 11, format::b_fs_yx_fsv16, reduce_mode::min, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(7, 3, 1, 1, 13, 11, format::b_fs_yx_fsv16, reduce_mode::mean, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 4, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::max, {1, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 13, 12, format::b_fs_yx_fsv16, reduce_mode::l2, {0, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::min, {2, 3}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 18, 11, format::b_fs_yx_fsv16, reduce_mode::max, {2, 1}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 16, 11, format::b_fs_yx_fsv16, reduce_mode::l1, {1, 0}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 14, 11, format::b_fs_yx_fsv16, reduce_mode::l2, {1}, "reduce_gpu_b_fs_yx_fsv16", false, data_types::f32, false, data_types::f32),

                            TestParamType_general_reduce_gpu(7, 3, 1, 1, 13, 11, format::b_fs_yx_fsv16, reduce_mode::mean, {3, 2, 1, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 4, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::max, {1, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 13, 9, format::b_fs_yx_fsv16, reduce_mode::l2, {0, 2, 3}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(16, 16, 1, 1, 16, 8, format::b_fs_yx_fsv16, reduce_mode::min, {2, 3}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 18, 11, format::b_fs_yx_fsv16, reduce_mode::max, {2, 1}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 16, 15, format::b_fs_yx_fsv16, reduce_mode::l1, {1, 0}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 3, 1, 1, 14, 11, format::b_fs_yx_fsv16, reduce_mode::mean, {1}, "reduce_gpu_b_fs_yx_fsv16", true, data_types::f32, false, data_types::f32)
                        ), general_reduce_gpu::PrintToStringParamName);

INSTANTIATE_TEST_SUITE_P(onednn_reduce_gpu_byxf_i8_f32,
                        onednn_reduce_gpu_i8_f32,
                        ::testing::Values(
                            TestParamType_general_reduce_gpu(7, 3, 1, 1, 13, 11, format::byxf, reduce_mode::mean, {3, 2, 1, 0}, "reduce_gpu_byxf", true, data_types::i8, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 3, 1, 1, 14, 11, format::byxf, reduce_mode::mean, {1}, "reduce_gpu_byxf", true, data_types::i8, false, data_types::f32)
                        ), general_reduce_gpu::PrintToStringParamName);

INSTANTIATE_TEST_SUITE_P(onednn_reduce_gpu_byxf_f16_f16,
                        onednn_reduce_gpu_f16_f16,
                        ::testing::Values(
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 16, 15, format::byxf, reduce_mode::l1, {1, 0}, "reduce_gpu_byxf", true, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(17, 3, 1, 1, 14, 11, format::byxf, reduce_mode::mean, {1}, "reduce_gpu_byxf", true, data_types::f16, false, data_types::f16)
                        ), general_reduce_gpu::PrintToStringParamName);

INSTANTIATE_TEST_SUITE_P(onednn_reduce_gpu_byxf_f32_f32,
                        onednn_reduce_gpu_f32_f32,
                        ::testing::Values(
                            TestParamType_general_reduce_gpu(17, 34, 1, 1, 16, 15, format::byxf, reduce_mode::l1, {1, 0}, "reduce_gpu_byxf", true, data_types::f32, false, data_types::f32),
                            TestParamType_general_reduce_gpu(17, 3, 1, 1, 14, 11, format::byxf, reduce_mode::mean, {1}, "reduce_gpu_byxf", true, data_types::f32, false, data_types::f32)
                        ), general_reduce_gpu::PrintToStringParamName);

// Regression test: oneDNN derives its src/dst memory descriptor rank from format::dimension(),
// not from the layout's partial shape rank. If an input layout carries a higher-rank partial shape
// than its assigned format (e.g. a 5D shape squeezed into a 4D bfyx format by an upstream
// reshape/optimization), oneDNN would build a truncated descriptor. reorder_inputs therefore
// realigns the reduce input format to the input rank, so the node must never be compiled with a
// layout whose format rank disagrees with its shape rank.
static void test_reduce_input_format_rank_mismatch(int64_t reduce_axis) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    const int f_size = 6;
    const int x_size = 8;
    // Partial shape is 5D (b, f, z, y, x) but the format below is 4D (bfyx): intentional mismatch.
    ov::PartialShape input_pshape{1, f_size, 1, 1, x_size};
    layout input_lay(input_pshape, data_types::f16, format::bfyx);
    auto input_mem = engine.allocate_memory(input_lay);

    std::vector<ov::float16> input_data(f_size * x_size);
    for (size_t i = 0; i < input_data.size(); i++)
        input_data[i] = ov::float16(static_cast<float>(i) * 0.1f);
    set_values(input_mem, input_data);

    topology topology;
    topology.add(input_layout("input", input_mem->get_layout()));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::mean, {reduce_axis}, false));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    config.set_property(ov::intel_gpu::optimize_data(true));

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    ASSERT_NO_THROW(network.execute());

    auto inst = network.get_primitive("reduce");
    ASSERT_NE(inst->get_impl(), nullptr);
    const auto& in_lay = inst->get_input_layout(0);
    ASSERT_EQ(in_lay.format.dimension(), in_lay.get_partial_shape().size())
        << "reduce input format rank must be realigned to the input shape rank, got " << in_lay.to_short_string();
}

TEST(reduce_gpu, onednn_reduce_input_format_rank_mismatch_blocked_axis) {
    test_reduce_input_format_rank_mismatch(1);
}

// Reducing a spatial axis exercises the tensor-field remapping in reorder_unreduced_axis_no_fusion(),
// which silently zeroed the wrong field when the input tensor rank and format rank disagreed.
TEST(reduce_gpu, onednn_reduce_input_format_rank_mismatch_spatial_axis) {
    test_reduce_input_format_rank_mismatch(4);
}

// Regression test, natural-selection variant: oneDNN's ReduceImplementationManager
// is only registered for shape_types::static_shape (see registry/reduce_impls.cpp), so a
// dynamic-shape reduce node has no oneDNN candidate at compile time. But b_fs_yx_fsv16 (blocked)
// is only registered for OCL's *static* reduce impl, not its dynamic one (see impls/ocl/reduce.cpp
// dyn_formats), so once the shape is realized at runtime there is no valid dynamic candidate
// either. The runtime static-shape fallback then picks oneDNN, without force_implementations,
// exercising the same reorder_unreduced_axis_no_fusion() rank-normalization path.
static void test_reduce_dynamic_node_static_fallback(int64_t reduce_axis) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    const int f_size = 17;
    const int x_size = 2;
    auto used_layout = layout({1, f_size, 1, x_size}, data_types::f16, format::b_fs_yx_fsv16);
    auto input_mem = engine.allocate_memory(used_layout);

    std::vector<float> input_data(f_size * x_size);
    {
        cldnn::mem_lock<ov::float16> input_ptr(input_mem, get_test_stream());
        for (int fi = 0; fi < f_size; fi++)
            for (int xi = 0; xi < x_size; xi++) {
                float val = static_cast<float>(fi * x_size + xi) * 0.1f;
                input_data[fi * x_size + xi] = val;
                tensor coords(batch(0), feature(fi), spatial(xi, 0, 0, 0));
                input_ptr[used_layout.get_linear_offset(coords)] = ov::float16(val);
            }
    }

    topology topology;
    // The runtime memory keeps its concrete b_fs_yx_fsv16 layout; only the declared input_layout
    // is dynamic, which is what makes the node's compile-time shape_type dynamic.
    topology.add(input_layout("input", layout(ov::PartialShape::dynamic(4), data_types::f16, format::b_fs_yx_fsv16)));
    topology.add(reduce("reduce", input_info("input"), reduce_mode::mean, {reduce_axis}, false));
    topology.add(reorder("reduce_bfyx", input_info("reduce"), format::bfyx, data_types::f16));

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    // No force_implementations: verify oneDNN is picked because it is the only valid candidate
    // once the OCL dynamic reduce impl rejects the blocked b_fs_yx_fsv16 format.

    network network(engine, topology, config);
    network.set_input_data("input", input_mem);

    auto outputs = network.execute();

    auto inst = network.get_primitive("reduce");
    auto impl = inst->get_impl();
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->m_manager, nullptr);
    ASSERT_EQ(impl->m_manager->get_impl_type(), impl_types::onednn)
        << "expected oneDNN static fallback to be selected without force_implementations";

    auto out_mem = outputs.at("reduce_bfyx").get_memory();
    cldnn::mem_lock<ov::float16, mem_lock_type::read> out_ptr(out_mem, get_test_stream());

    // axis 1 reduces feature (output indexed by x), axis 3 reduces x (output indexed by feature).
    const int out_size = reduce_axis == 1 ? x_size : f_size;
    const int reduced_size = reduce_axis == 1 ? f_size : x_size;
    for (int oi = 0; oi < out_size; oi++) {
        float expected = 0.f;
        for (int ri = 0; ri < reduced_size; ri++) {
            const int fi = reduce_axis == 1 ? ri : oi;
            const int xi = reduce_axis == 1 ? oi : ri;
            expected += input_data[fi * x_size + xi];
        }
        expected /= reduced_size;
        ASSERT_NEAR(static_cast<float>(out_ptr[oi]), expected, 1e-1f) << "out=" << oi;
    }
}

TEST(reduce_gpu, onednn_reduce_dynamic_node_static_fallback_selection) {
    test_reduce_dynamic_node_static_fallback(1);
}

TEST(reduce_gpu, onednn_reduce_dynamic_node_static_fallback_spatial_axis) {
    test_reduce_dynamic_node_static_fallback(3);
}
#endif  // ENABLE_ONEDNN_FOR_GPU


#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_P(general_reduce_gpu_i8_i8, base_cached) { execute(true); }

TEST_P(general_reduce_gpu_i8_f32, base_cached) { execute(true); }

TEST_P(general_reduce_gpu_f32_f32, base_cached) { execute(true); }

TEST_P(general_reduce_gpu_xy_f32, base_cached) { execute(true); }

TEST_P(general_reduce_gpu_xy_i8, base_cached) { execute(true); }
#endif  // RUN_ALL_MODEL_CACHING_TESTS

TEST(reduce_gpu, common_bfyx_cached) {
    test_common_bfyx<float>(true);
}

class reduce_scalar_output_f16_f16 : public ReduceTestBase<data_types::f16, data_types::f16> {};
TEST_P(reduce_scalar_output_f16_f16, base) { execute(false); }

INSTANTIATE_TEST_SUITE_P(reduce_scalar_output_f16_f16,
                        reduce_scalar_output_f16_f16,
                        ::testing::Values(
                            TestParamType_general_reduce_gpu(1, 1, 1, 1, 1013, 2, format::bfyx, reduce_mode::sum, {3, 2, 1, 0},  "reduce_simple_to_scalar", false, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(1, 1, 1, 1, 1013, 2, format::bfyx, reduce_mode::min, {3, 2, 1, 0},  "reduce_simple_to_scalar", false, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(1, 1, 1, 1, 1013, 2, format::bfyx, reduce_mode::mean, {3, 2, 1, 0}, "reduce_simple_to_scalar", false, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(1, 1, 1, 1, 1013, 2, format::bfyx, reduce_mode::max, {3, 2, 1, 0},  "reduce_simple_to_scalar", false, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(1, 1, 1, 1, 1024, 1, format::bfyx, reduce_mode::min, {3, 2, 1, 0},  "reduce_simple_to_scalar", false, data_types::f16, false, data_types::f16),
                            TestParamType_general_reduce_gpu(1, 1, 1, 1, 1025, 1, format::bfyx, reduce_mode::min, {3, 2, 1, 0},  "reduce_simple_to_scalar", false, data_types::f16, false, data_types::f16)
                        ), general_reduce_gpu::PrintToStringParamName);

TEST(reduce_f32_fw_gpu, large_buffer) {
    // This test is used for manual testing only.
    GTEST_SKIP();

    auto engine = create_test_engine();
    engine->set_enable_large_allocations(true);

    size_t s0 = 16384;
    size_t s1 = 256 * (256 + 128);
    ov::Shape sz_6gb = { 1, 1, s1, s0 }; // *4 bytes;
    size_t peak_mem_usage = (ov::shape_size(sz_6gb) + s0) * sizeof(float);
    if (engine->get_device_info().max_global_mem_size < peak_mem_usage)
        GTEST_SKIP();

    layout in_l = { sz_6gb, data_types::f32, format::bfyx };

    auto config = get_test_default_config(*engine);
    ov::intel_gpu::ImplementationDesc reduce_impl = {format::bfyx, "", impl_types::any};
    if (engine->get_device_info().supports_immad) {
        reduce_impl.impl_type = impl_types::onednn;
    } else {
        reduce_impl.impl_type = impl_types::ocl;
    }
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"reduce", reduce_impl}}));

    topology topology(input_layout("input", in_l),
                      reduce("reduce", input_info("input"), reduce_mode::mean, {2}, true));
    network network(*engine, topology, config);
    auto input = network.get_output_memory("input");
    {
        mem_lock<float, mem_lock_type::write> l(input, get_test_stream());
        const size_t test_size = ov::shape_size(sz_6gb);
        for (size_t i = 0; i < test_size; i++) {
            l[i] = static_cast<float>(i) / s0;
        }
    }

    network.set_input_data("input", input);
    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "reduce");

    auto output_memory = outputs.at("reduce").get_memory();
    auto output_layout = output_memory->get_layout();
    cldnn::mem_lock<float, mem_lock_type::read> output_ptr(output_memory, get_test_stream());

    ASSERT_EQ(output_layout.format, format::bfyx);
    ASSERT_EQ(output_layout.get_linear_size(), s0);

    // ensure that single 6GB buffer is allocated
    ASSERT_EQ(engine->get_max_used_device_memory(), peak_mem_usage);

    size_t sum = s1 * (s1 - 1) / 2;
    float mean = static_cast<float>(sum) / static_cast<float>(s1);
    for (size_t i = 0; i < s0; i++) {
        ASSERT_NEAR(mean, output_ptr[i], 1.0f);
    }
}
