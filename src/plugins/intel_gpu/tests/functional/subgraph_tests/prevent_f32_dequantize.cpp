// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace {

// GPU_OPTIMIZE_DATA is an internal property, so Core::compile_model cannot set it through AnyMap.
class ScopedOptimizeData {
public:
    explicit ScopedOptimizeData(bool enabled) {
        const char* value = std::getenv("OV_GPU_OPTIMIZE_DATA");
        if (value != nullptr) {
            was_set = true;
            previous_value = value;
        }
        OPENVINO_ASSERT(set_value(enabled ? "1" : "0") == 0, "Failed to set OV_GPU_OPTIMIZE_DATA");
    }

    ~ScopedOptimizeData() {
        EXPECT_EQ(was_set ? set_value(previous_value.c_str()) : unset_value(), 0) << "Failed to restore OV_GPU_OPTIMIZE_DATA";
    }

    ScopedOptimizeData(const ScopedOptimizeData&) = delete;
    ScopedOptimizeData& operator=(const ScopedOptimizeData&) = delete;

private:
    static int set_value(const char* value) {
#ifdef _WIN32
        return _putenv_s("OV_GPU_OPTIMIZE_DATA", value);
#else
        return setenv("OV_GPU_OPTIMIZE_DATA", value, 1);
#endif
    }

    static int unset_value() {
#ifdef _WIN32
        return _putenv_s("OV_GPU_OPTIMIZE_DATA", "");
#else
        return unsetenv("OV_GPU_OPTIMIZE_DATA");
#endif
    }

    bool was_set = false;
    std::string previous_value;
};

std::string runtime_info(const std::shared_ptr<ov::Node>& node, const std::string& key) {
    const auto& info = node->get_rt_info();
    const auto it = info.find(key);
    return it == info.end() ? std::string{} : it->second.as<std::string>();
}

bool has_original_name(const std::shared_ptr<ov::Node>& node, const std::string& expected) {
    std::istringstream names(runtime_info(node, ov::exec_model_info::ORIGINAL_NAMES));
    std::string name;
    while (std::getline(names, name, ',')) {
        if (name == expected) {
            return true;
        }
    }
    return false;
}

std::vector<std::shared_ptr<ov::Node>> find_runtime_nodes(const std::shared_ptr<const ov::Model>& model,
                                                          const std::string& layer_type,
                                                          const std::string& original_name = {}) {
    std::vector<std::shared_ptr<ov::Node>> nodes;
    for (const auto& node : model->get_ordered_ops()) {
        if (runtime_info(node, ov::exec_model_info::LAYER_TYPE) == layer_type && (original_name.empty() || has_original_name(node, original_name))) {
            nodes.push_back(node);
        }
    }
    return nodes;
}

std::shared_ptr<ov::Node> find_runtime_node(const std::shared_ptr<const ov::Model>& model,
                                            const std::string& layer_type,
                                            const std::string& original_name = {}) {
    const auto nodes = find_runtime_nodes(model, layer_type, original_name);
    EXPECT_EQ(nodes.size(), 1u) << "Expected one " << layer_type << " node with original name '" << original_name << "'";
    return nodes.size() == 1 ? nodes.front() : nullptr;
}

std::string describe_runtime_model(const std::shared_ptr<const ov::Model>& model) {
    std::ostringstream description;
    description << "Runtime graph:\n";
    for (const auto& node : model->get_ordered_ops()) {
        description << node->get_friendly_name() << " [" << runtime_info(node, ov::exec_model_info::LAYER_TYPE)
                    << "] originals=" << runtime_info(node, ov::exec_model_info::ORIGINAL_NAMES) << " inputs:";
        for (const auto& input : node->input_values()) {
            description << " " << input.get_node()->get_friendly_name() << ":" << input.get_element_type();
        }
        description << " outputs:";
        for (const auto& output : node->outputs()) {
            description << " " << output.get_element_type();
        }
        description << "\n";
    }
    return description.str();
}

std::shared_ptr<ov::Node> make_quantized_input(const ov::Shape& shape, const std::string& name, ov::ParameterVector& parameters) {
    const auto type = ov::element::f16;
    auto input = std::make_shared<ov::op::v0::Parameter>(type, shape);
    input->set_friendly_name(name);
    parameters.push_back(input);
    // The quantization step (1/128) and both endpoints are exactly representable in f16.
    auto fq = ov::test::utils::make_fake_quantize(input, type, 256ul, {}, {0.0f}, {255.0f / 128}, {0.0f}, {255.0f / 128});
    fq->set_friendly_name(name + "_fq");
    return fq;
}

class PreventF32DequantizeTestBase : public ov::test::SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        configuration[ov::hint::inference_precision.name()] = ov::element::f16.get_type_name();
    }

    void compile_model() override {
        const ScopedOptimizeData scoped_optimize_data(optimize_data);
        ov::test::SubgraphBaseStaticTest::compile_model();
    }

    void generate_inputs(const std::vector<ov::Shape>& target_input_static_shapes) override {
        inputs.clear();
        const auto& parameters = function->get_parameters();
        ASSERT_EQ(parameters.size(), target_input_static_shapes.size());
        for (size_t input_index = 0; input_index < parameters.size(); ++input_index) {
            ov::Tensor tensor(ov::element::f16, target_input_static_shapes[input_index]);
            auto* data = tensor.data<ov::float16>();
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                // Independent, deterministic permutations of the quantization grid.
                // For the Add inputs: x[i] = (i % 256)/128, y[i] = ((17*i + 29) % 256)/128.
                const auto quantized_value = (i * (16 * input_index + 1) + 29 * input_index) % 256;
                data[i] = ov::float16(static_cast<float>(quantized_value) / 128.0f);
            }
            inputs.emplace(parameters[input_index], tensor);
        }
    }

    bool optimize_data = true;
};

class PreventF32DequantizeAddBase : public PreventF32DequantizeTestBase {
protected:
    enum class Consumer { MVN, ReshapeWithMVN, ReshapeNonMVN };

    void set_up_model(Consumer consumer_type) {
        PreventF32DequantizeTestBase::SetUp();
        consumer = consumer_type;
        ov::ParameterVector parameters;
        const ov::Shape shape{1, 4, 16, 16};
        auto x = make_quantized_input(shape, "input_x", parameters);
        auto y = make_quantized_input(shape, "input_y", parameters);

        // Two non-constant inputs produce the TypeRelaxed Add(f32) -> Multiply(f16) LPT pattern.
        auto add = std::make_shared<ov::op::v1::Add>(x, y);
        add->set_friendly_name("inner_add");
        std::shared_ptr<ov::Node> output = add;
        if (consumer != Consumer::MVN) {
            auto target_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1024});
            output = std::make_shared<ov::op::v1::Reshape>(output, target_shape, false);
            output->set_friendly_name("reshape");
        }
        if (consumer == Consumer::ReshapeNonMVN) {
            output = std::make_shared<ov::op::v0::Relu>(output);
            output->set_friendly_name("relu");
        } else {
            const std::vector<int64_t> axes_values = consumer == Consumer::MVN ? std::vector<int64_t>{2, 3} : std::vector<int64_t>{1};
            auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes_values.size()}, axes_values);
            output = std::make_shared<ov::op::v6::MVN>(output, axes, true, 1e-5f, ov::op::MVNEpsMode::INSIDE_SQRT);
            output->set_friendly_name("mvn");
        }
        auto result = std::make_shared<ov::op::v0::Result>(output);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, parameters, "PreventF32DequantizeAdd");
    }

    void validate() override {
        ov::test::SubgraphBaseStaticTest::validate();
        const auto runtime_model = compiledModel.get_runtime_model();
        ASSERT_NE(runtime_model, nullptr);
        SCOPED_TRACE(describe_runtime_model(runtime_model));
        SCOPED_TRACE(optimize_data ? "optimize_data=true" : "optimize_data=false");

        const auto add = find_runtime_node(runtime_model, "Eltwise", "inner_add");
        ASSERT_NE(add, nullptr);
        // runtimePrecision can be f32 even when the fused Add produces f16.
        EXPECT_EQ(add->get_output_element_type(0), optimize_data ? ov::element::f16 : ov::element::f32);

        std::shared_ptr<ov::Node> boundary;
        if (consumer != Consumer::MVN) {
            // In the Relu case, later passes move Reshape after Relu and transfer its friendly name.
            boundary = find_runtime_node(runtime_model, "Reshape", consumer == Consumer::ReshapeWithMVN ? "reshape" : "");
            ASSERT_NE(boundary, nullptr);
        }
        if (consumer != Consumer::ReshapeNonMVN) {
            const auto mvn = find_runtime_node(runtime_model, "MVN", "mvn");
            ASSERT_NE(mvn, nullptr);
            ASSERT_EQ(mvn->get_input_size(), 1u);
            EXPECT_EQ(mvn->get_input_element_type(0), ov::element::f16);
            EXPECT_EQ(mvn->get_output_element_type(0), ov::element::f16);
            if (consumer == Consumer::MVN) {
                boundary = mvn;
            } else {
                EXPECT_EQ(mvn->input_value(0).get_node_shared_ptr(), boundary);
            }
        }
        ASSERT_NE(boundary, nullptr);
        ASSERT_EQ(boundary->get_input_size(), 1u);
        EXPECT_EQ(boundary->get_input_element_type(0), ov::element::f16);
        EXPECT_EQ(boundary->get_output_element_type(0), ov::element::f16);

        if (optimize_data) {
            // The scale is fused into Add, whose f16 output feeds MVN or Reshape.
            EXPECT_EQ(boundary->input_value(0).get_node_shared_ptr(), add);
        } else {
            auto scale_input = add;
            if (consumer == Consumer::ReshapeNonMVN) {
                // Negative control: dequantization moves through Relu, which then operates in f32.
                const auto relu = find_runtime_node(runtime_model, "Activation");
                ASSERT_NE(relu, nullptr);
                ASSERT_EQ(relu->get_input_size(), 1u);
                EXPECT_EQ(relu->input_value(0).get_node_shared_ptr(), add);
                EXPECT_EQ(relu->get_input_element_type(0), ov::element::f32);
                EXPECT_EQ(relu->get_output_element_type(0), ov::element::f32);
                scale_input = relu;
            }
            // With fusion disabled, check the standalone f32 -> f16 scale at the boundary.
            const auto scale = boundary->input_value(0).get_node_shared_ptr();
            ASSERT_EQ(runtime_info(scale, ov::exec_model_info::LAYER_TYPE), "Eltwise");
            ASSERT_EQ(scale->get_input_size(), 1u);
            EXPECT_EQ(scale->input_value(0).get_node_shared_ptr(), scale_input);
            EXPECT_EQ(scale->get_input_element_type(0), ov::element::f32);
            EXPECT_EQ(scale->get_output_element_type(0), ov::element::f16);
        }
    }

private:
    Consumer consumer = Consumer::MVN;
};

class PreventF32DequantizeMVN : public PreventF32DequantizeAddBase {
protected:
    void SetUp() override {
        set_up_model(Consumer::MVN);
    }
};

TEST_F(PreventF32DequantizeMVN, smoke_GPU_PreventF32DequantizeMVN) {
    run();
}

TEST_F(PreventF32DequantizeMVN, smoke_GPU_PreventF32DequantizeMVN_WithoutFusion) {
    optimize_data = false;
    run();
}

class PreventF32DequantizeReshapeWithMVN : public PreventF32DequantizeAddBase {
protected:
    void SetUp() override {
        set_up_model(Consumer::ReshapeWithMVN);
    }
};

TEST_F(PreventF32DequantizeReshapeWithMVN, smoke_GPU_PreventF32DequantizeReshapeWithMVN) {
    run();
}

TEST_F(PreventF32DequantizeReshapeWithMVN, smoke_GPU_PreventF32DequantizeReshapeWithMVN_WithoutFusion) {
    optimize_data = false;
    run();
}

class PreventF32DequantizeReshapeNonMVN : public PreventF32DequantizeAddBase {
protected:
    void SetUp() override {
        set_up_model(Consumer::ReshapeNonMVN);
    }
};

TEST_F(PreventF32DequantizeReshapeNonMVN, smoke_GPU_PreventF32DequantizeReshapeNonMVN) {
    run();
}

TEST_F(PreventF32DequantizeReshapeNonMVN, smoke_GPU_PreventF32DequantizeReshapeNonMVN_WithoutFusion) {
    optimize_data = false;
    run();
}

class PreventF32DequantizeVariadicSplitBase : public PreventF32DequantizeTestBase {
protected:
    void set_up_model(bool with_absorber, size_t concat_count, ov::element::Type expected_split_type) {
        PreventF32DequantizeTestBase::SetUp();
        has_absorber = with_absorber;
        split_type = expected_split_type;
        const auto type = ov::element::f16;
        ov::ParameterVector parameters;
        auto fq = make_quantized_input({1, 64}, "source", parameters);
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto split_lengths = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {32, 32});
        auto split = std::make_shared<ov::op::v1::VariadicSplit>(fq, axis, split_lengths);
        split->set_friendly_name("split");

        ov::Output<ov::Node> branch0 = split->output(0);
        ov::Shape current_shape{1, 32};
        for (size_t i = 0; i < concat_count; ++i) {
            // Alternating axes prevent CommonOptimizations from flattening the Concat chain.
            const auto concat_axis = i % 2;
            auto extra_shape = current_shape;
            extra_shape[concat_axis] = 1;
            auto extra = make_quantized_input(extra_shape, "concat_input_" + std::to_string(i), parameters);
            auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{branch0, extra}, static_cast<int64_t>(concat_axis));
            concat->set_friendly_name("budget_concat_" + std::to_string(i));
            branch0 = concat->output(0);
            ++current_shape[concat_axis];
        }

        if (has_absorber) {
            auto weights = ov::op::v0::Constant::create(type, ov::Shape{current_shape[1], 16}, {0.5f});
            auto weights_fq = ov::test::utils::make_fake_quantize(weights, type, 256ul, {}, {-1.0f}, {127.0f / 128}, {-1.0f}, {127.0f / 128});
            auto matmul = std::make_shared<ov::op::v0::MatMul>(branch0, weights_fq);
            matmul->set_friendly_name("matmul");
            branch0 = matmul->output(0);
        } else {
            auto bias = ov::op::v0::Constant::create(type, ov::Shape{}, {1.0f});
            auto add = std::make_shared<ov::op::v1::Add>(branch0, bias);
            add->set_friendly_name("branch0_add");
            branch0 = add->output(0);
        }

        // This non-absorbing branch consumes one visit in the split callback's BFS budget.
        auto bias = ov::op::v0::Constant::create(type, ov::Shape{}, {2.0f});
        auto branch1 = std::make_shared<ov::op::v1::Add>(split->output(1), bias);
        branch1->set_friendly_name("branch1_add");
        auto result0 = std::make_shared<ov::op::v0::Result>(branch0);
        auto result1 = std::make_shared<ov::op::v0::Result>(branch1);
        function = std::make_shared<ov::Model>(ov::ResultVector{result0, result1}, parameters, "PreventF32DequantizeSplit");
    }

    void validate() override {
        ov::test::SubgraphBaseStaticTest::validate();
        const auto runtime_model = compiledModel.get_runtime_model();
        ASSERT_NE(runtime_model, nullptr);
        SCOPED_TRACE(describe_runtime_model(runtime_model));

        const auto source_fq = find_runtime_node(runtime_model, "Quantize", "source_fq");
        ASSERT_NE(source_fq, nullptr);
        ASSERT_GE(source_fq->get_output_size(), 1u);
        for (const auto& output : source_fq->outputs()) {
            EXPECT_EQ(output.get_element_type(), split_type);
        }

        // VariadicSplit is lowered to two Crop nodes. Check both outputs and their producer.
        const auto crops = find_runtime_nodes(runtime_model, "Crop", "split");
        ASSERT_EQ(crops.size(), 2u);
        for (const auto& crop : crops) {
            SCOPED_TRACE(crop->get_friendly_name());
            ASSERT_EQ(crop->get_input_size(), 1u);
            EXPECT_EQ(crop->input_value(0).get_node_shared_ptr(), source_fq);
            EXPECT_EQ(crop->get_input_element_type(0), split_type);
            EXPECT_EQ(crop->get_output_element_type(0), split_type);
        }

        if (has_absorber) {
            const auto matmul = find_runtime_node(runtime_model, "FullyConnected", "matmul");
            ASSERT_NE(matmul, nullptr);
            ASSERT_GE(matmul->get_input_size(), 1u);
            EXPECT_EQ(matmul->get_input_element_type(0), split_type);
        }
    }

private:
    bool has_absorber = false;
    ov::element::Type split_type = ov::element::dynamic;
};

class PreventF32DequantizeVariadicSplitNoAbsorbingConsumer : public PreventF32DequantizeVariadicSplitBase {
protected:
    void SetUp() override {
        set_up_model(false, 0, ov::element::f16);
    }
};

TEST_F(PreventF32DequantizeVariadicSplitNoAbsorbingConsumer, smoke_GPU_VariadicSplitNoAbsorber) {
    run();
}

class PreventF32DequantizeVariadicSplitWithAbsorbingConsumer : public PreventF32DequantizeVariadicSplitBase {
protected:
    void SetUp() override {
        set_up_model(true, 0, ov::element::u8);
    }
};

TEST_F(PreventF32DequantizeVariadicSplitWithAbsorbingConsumer, smoke_GPU_VariadicSplitWithAbsorber) {
    run();
}

class PreventF32DequantizeVariadicSplitWithinBudget : public PreventF32DequantizeVariadicSplitBase {
protected:
    void SetUp() override {
        // 30 Concats + the other branch's Add + MatMul fit exactly in the 32-visit budget.
        set_up_model(true, 30, ov::element::u8);
    }
};

TEST_F(PreventF32DequantizeVariadicSplitWithinBudget, smoke_GPU_VariadicSplitWithinBudget) {
    run();
}

class PreventF32DequantizeVariadicSplitBudgetExhaustion : public PreventF32DequantizeVariadicSplitBase {
protected:
    void SetUp() override {
        // 31 Concats + the other branch's Add exhaust the budget before MatMul is visited.
        set_up_model(true, 31, ov::element::f16);
        // When the budget is exhausted, MatMul executes in FP16 without quantization.
        // Summing 47 FP16 products accumulates minor ULP differences (~0.26% / 4 ULPs)
        // between CPU reference (accumulated in FP32) and discrete GPU (accumulated in FP16).
        rel_threshold = 0.01;
    }
};

TEST_F(PreventF32DequantizeVariadicSplitBudgetExhaustion, smoke_GPU_VariadicSplitBudgetExhaustion) {
    run();
}

}  // namespace
