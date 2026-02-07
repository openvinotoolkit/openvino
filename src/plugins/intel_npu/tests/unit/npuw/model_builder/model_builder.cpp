// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "model_builder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

#include "openvino/op/assign.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset11.hpp"

namespace ov {
namespace test {
namespace npuw {

// ============================================================================
// Simple test models (backward compatibility)
// ============================================================================

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_one_op() {
    auto param = std::make_shared<ov::opset11::Parameter>(ov::element::i64, ov::PartialShape{1, 3, 2, 2});
    param->set_friendly_name("input");
    auto const_value = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1, 1, 1, 1}, {1});
    const_value->set_friendly_name("const_val");
    auto add = std::make_shared<ov::opset11::Add>(param, const_value);
    add->set_friendly_name("add");
    auto result = std::make_shared<ov::opset11::Result>(add);
    result->set_friendly_name("res");
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param});
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_without_repeated_blocks() {
    std::shared_ptr<ov::op::v0::Parameter> input =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1, 40});
    m_nodes.push_back(input);
    set_name(input);

    std::shared_ptr<ov::Node> res = get_block(input);

    auto result = std::make_shared<ov::op::v0::Result>(res);
    m_nodes.push_back(result);
    set_name(result);

    ov::ParameterVector params = {input};
    ov::ResultVector results = {result};

    return std::make_shared<ov::Model>(results, params);
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_repeated_blocks(std::size_t repetitions) {
    return get_model_with_repeated_blocks_and_results(repetitions, {});
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_repeated_blocks() {
    return get_model_with_repeated_blocks(10);
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_repeated_blocks_and_results(
    std::size_t repetitions,
    const std::vector<std::size_t>& block_indices) {
    // Generate head
    std::shared_ptr<ov::op::v0::Parameter> input =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 1, 40});
    m_nodes.push_back(input);
    set_name(input);

    std::vector<std::shared_ptr<ov::Node>> head(7, nullptr);
    head[0] = std::make_shared<ov::op::v1::Add>(input, input);
    head[1] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int>{2});
    head[2] = std::make_shared<ov::op::v1::Divide>(head[0], head[1], true);
    head[3] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 4, 10});
    head[4] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int>{1, 1, 40});
    head[5] = std::make_shared<ov::op::v1::Reshape>(head[2], head[3], false);
    head[6] = std::make_shared<ov::op::v1::Reshape>(head[5], head[4], false);

    for (const auto& h : head) {
        m_nodes.push_back(h);
        set_name(h);
    }

    // Generate repeated blocks
    std::shared_ptr<ov::Node> output = get_block(head[6]);
    std::vector<std::shared_ptr<ov::Node>> block_outputs;
    block_outputs.push_back(output);

    for (size_t i = 0; i < repetitions - 1; ++i) {
        output = get_block(output);
        block_outputs.push_back(output);
    }

    // Generate tail
    std::vector<std::shared_ptr<ov::Node>> tail(6, nullptr);
    tail[0] = std::make_shared<ov::op::v0::Concat>(block_outputs, -1);
    tail[1] = std::make_shared<ov::op::v0::Constant>(ov::element::i32,
                                                     ov::Shape{3},
                                                     std::vector<int>{1, 40, int(repetitions)});
    tail[2] = std::make_shared<ov::op::v1::Reshape>(tail[0], tail[1], false);
    tail[3] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1});
    tail[4] = std::make_shared<ov::op::v1::Multiply>(tail[2], tail[3]);
    tail[5] = std::make_shared<ov::op::v1::Add>(tail[4], tail[4]);

    for (const auto& t : tail) {
        m_nodes.push_back(t);
        set_name(t);
    }

    // Create Results
    ov::ResultVector results;

    // Add Results for specified blocks
    for (size_t idx : block_indices) {
        if (idx < block_outputs.size()) {
            auto result = std::make_shared<ov::op::v0::Result>(block_outputs[idx]);
            m_nodes.push_back(result);
            set_name(result);
            results.push_back(result);
        }
    }

    // Always add final tail Result
    auto final_result = std::make_shared<ov::op::v0::Result>(tail[5]);
    m_nodes.push_back(final_result);
    set_name(final_result);
    results.push_back(final_result);

    ov::ParameterVector params = {input};

    return std::make_shared<ov::Model>(results, params);
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_repeated_blocks_and_parameters(
    std::size_t repetitions,
    const std::vector<std::size_t>& block_indices) {
    if (repetitions == 0) {
        repetitions = 1;
    }

    auto input = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{1, 1, 8});
    m_nodes.push_back(input);

    ov::ParameterVector params = {input};

    std::vector<std::size_t> sorted_indices = block_indices;
    std::sort(sorted_indices.begin(), sorted_indices.end());
    sorted_indices.erase(std::unique(sorted_indices.begin(), sorted_indices.end()), sorted_indices.end());

    std::unordered_map<std::size_t, std::shared_ptr<ov::opset11::Parameter>> block_params;
    for (std::size_t idx : sorted_indices) {
        if (idx >= repetitions) {
            continue;
        }

        auto param = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{1, 1, 8});
        m_nodes.push_back(param);
        block_params.emplace(idx, param);
        params.push_back(param);
    }

    auto scale_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1}, {1.f});
    auto bias_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1, 8}, std::vector<float>(8, 0.5f));
    auto head_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1, 8}, std::vector<float>(8, 1.f));
    m_nodes.push_back(scale_const);
    m_nodes.push_back(bias_const);
    m_nodes.push_back(head_const);

    auto head_add = std::make_shared<ov::opset11::Add>(input, head_const);
    auto head_relu = std::make_shared<ov::opset11::Relu>(head_add);
    m_nodes.push_back(head_add);
    m_nodes.push_back(head_relu);

    ov::Output<ov::Node> current = head_relu;

    for (std::size_t i = 0; i < repetitions; ++i) {
        auto it = block_params.find(i);
        ov::Output<ov::Node> rhs = (it != block_params.end()) ? it->second : current;

        auto add = std::make_shared<ov::opset11::Add>(current, rhs);
        m_nodes.push_back(add);

        auto mul = std::make_shared<ov::opset11::Multiply>(add, scale_const);
        m_nodes.push_back(mul);

        auto relu = std::make_shared<ov::opset11::Relu>(mul);
        m_nodes.push_back(relu);

        auto add_bias = std::make_shared<ov::opset11::Add>(relu, bias_const);
        m_nodes.push_back(add_bias);

        current = add_bias;
    }

    auto tail_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1, 1, 8}, std::vector<float>(8, 2.f));
    m_nodes.push_back(tail_const);

    auto tail_mul = std::make_shared<ov::opset11::Multiply>(current, tail_const);
    auto tail_add = std::make_shared<ov::opset11::Add>(tail_mul, tail_const);
    m_nodes.push_back(tail_mul);
    m_nodes.push_back(tail_add);

    auto result = std::make_shared<ov::opset11::Result>(tail_add);
    m_nodes.push_back(result);

    return std::make_shared<ov::Model>(ov::ResultVector{result}, params);
}

std::shared_ptr<ov::Model> ModelBuilder::get_model_with_multi_output_repeating_blocks(
    std::size_t repetitions,
    bool last_block_has_direct_result) {
    if (repetitions == 0) {
        repetitions = 1;
    }

    auto input = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{1, 1, 8});
    m_nodes.push_back(input);
    set_name(input);

    auto add_const = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1}, {1.f});
    auto k_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {8});
    auto seed_indices = ov::opset11::Constant::create(ov::element::i32, ov::Shape{1, 1, 8}, {0, 1, 2, 3, 4, 5, 6, 7});
    auto tail_scale = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1}, {0.5f});
    auto tail_bias = ov::opset11::Constant::create(ov::element::f32, ov::Shape{1}, {2.f});

    for (const auto& c : {add_const, k_const, seed_indices, tail_scale, tail_bias}) {
        m_nodes.push_back(c);
        set_name(c);
    }

    ov::Output<ov::Node> current_values = input;
    ov::Output<ov::Node> current_indices = seed_indices;

    for (std::size_t i = 0; i < repetitions; ++i) {
        auto indices_as_float = std::make_shared<ov::opset11::Convert>(current_indices, ov::element::f32);
        m_nodes.push_back(indices_as_float);
        set_name(indices_as_float);

        auto mixed = std::make_shared<ov::opset11::Add>(current_values, indices_as_float);
        m_nodes.push_back(mixed);
        set_name(mixed);

        auto shifted = std::make_shared<ov::opset11::Add>(mixed, add_const);
        m_nodes.push_back(shifted);
        set_name(shifted);

        auto topk = std::make_shared<ov::opset11::TopK>(shifted,
                                                        k_const,
                                                        -1,
                                                        ov::op::TopKMode::MAX,
                                                        ov::op::TopKSortType::SORT_VALUES,
                                                        ov::element::i32);
        m_nodes.push_back(topk);
        set_name(topk);

        current_values = topk->output(0);
        current_indices = topk->output(1);
    }

    auto tail_indices_as_float = std::make_shared<ov::opset11::Convert>(current_indices, ov::element::f32);
    m_nodes.push_back(tail_indices_as_float);
    set_name(tail_indices_as_float);

    auto tail_mixed = std::make_shared<ov::opset11::Add>(current_values, tail_indices_as_float);
    m_nodes.push_back(tail_mixed);
    set_name(tail_mixed);

    auto tail_mul = std::make_shared<ov::opset11::Multiply>(tail_mixed, tail_scale);
    m_nodes.push_back(tail_mul);
    set_name(tail_mul);

    auto tail_add = std::make_shared<ov::opset11::Add>(tail_mul, tail_bias);
    m_nodes.push_back(tail_add);
    set_name(tail_add);

    ov::ResultVector results;
    auto tail_result = std::make_shared<ov::opset11::Result>(tail_add);
    m_nodes.push_back(tail_result);
    set_name(tail_result);
    results.push_back(tail_result);

    if (last_block_has_direct_result) {
        auto direct_result = std::make_shared<ov::opset11::Result>(current_values);
        m_nodes.push_back(direct_result);
        set_name(direct_result);
        results.push_back(direct_result);
    }

    ov::ParameterVector params = {input};

    return std::make_shared<ov::Model>(results, params);
}

std::shared_ptr<ov::Node> ModelBuilder::get_block(const std::shared_ptr<ov::Node>& input) {
    std::vector<std::shared_ptr<ov::Node>> model_c(18, nullptr);
    model_c[0] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{0, 2, 1, 3});
    model_c[1] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{1});
    model_c[2] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{0});
    model_c[3] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{2});
    model_c[4] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{0});
    model_c[5] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 1, 1});
    model_c[6] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{1});
    model_c[7] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int>{0});
    model_c[8] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 1, 1});
    model_c[9] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{1, 1, 1, 2});
    model_c[10] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{1, 1, 1, 1});
    model_c[11] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{4}, std::vector<int>{1, 1, 1, 2});
    model_c[12] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1, 1});
    model_c[13] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1, 1});
    model_c[14] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1, 1, 1, 1});
    model_c[15] = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{40, 40});
    model_c[16] = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int>{1, 1, 4, 10});
    model_c[17] = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{3}, std::vector<int>{1, 1, 40});

    for (const auto& c : model_c) {
        m_nodes.push_back(c);
        set_name(c);
    }

    std::vector<std::shared_ptr<ov::Node>> convert(3, nullptr);
    convert[0] = std::make_shared<ov::op::v0::Convert>(model_c[15], ov::element::f16);
    convert[1] = std::make_shared<ov::op::v0::Convert>(convert[0], ov::element::i32);
    convert[2] = std::make_shared<ov::op::v0::Convert>(model_c[12], ov::element::i32);

    for (const auto& c : convert) {
        m_nodes.push_back(c);
        set_name(c);
    }

    std::vector<std::shared_ptr<ov::Node>> op(16, nullptr);
    op[0] = std::make_shared<ov::op::v0::MatMul>(input, convert[1], false, true);
    op[1] = std::make_shared<ov::op::v1::Reshape>(op[0], model_c[16], false);
    op[2] = std::make_shared<ov::op::v1::Transpose>(op[1], model_c[0]);
    op[3] = std::make_shared<ov::op::v0::ShapeOf>(op[2]);
    op[4] = std::make_shared<ov::op::v1::Gather>(op[3], model_c[1], model_c[2]);
    op[5] = std::make_shared<ov::op::v1::Divide>(op[4], model_c[3], true);
    op[6] = std::make_shared<ov::op::v0::Floor>(op[5]);
    op[7] = std::make_shared<ov::op::v3::ScatterUpdate>(model_c[5], model_c[6], op[6], model_c[7]);
    op[8] = std::make_shared<ov::op::v1::StridedSlice>(op[2],
                                                       model_c[8],
                                                       op[7],
                                                       model_c[9],
                                                       std::vector<int64_t>{1, 1, 1, 1},
                                                       std::vector<int64_t>{1, 1, 1, 1});
    op[9] = std::make_shared<ov::op::v1::StridedSlice>(op[2],
                                                       op[7],
                                                       model_c[10],
                                                       model_c[11],
                                                       std::vector<int64_t>{1, 1, 1, 1},
                                                       std::vector<int64_t>{1, 1, 1, 1});
    op[10] = std::make_shared<ov::op::v1::Multiply>(op[9], convert[2]);
    op[11] = std::make_shared<ov::op::v0::Concat>(std::vector<std::shared_ptr<ov::Node>>{op[10], op[8]}, -1);
    op[12] = std::make_shared<ov::op::v1::Multiply>(model_c[13], op[11]);
    op[13] = std::make_shared<ov::op::v1::Multiply>(model_c[14], op[2]);
    op[14] = std::make_shared<ov::op::v1::Add>(op[13], op[12]);
    op[15] = std::make_shared<ov::op::v1::Reshape>(op[14], model_c[17], false);

    for (const auto& o : op) {
        m_nodes.push_back(o);
        set_name(o);
    }

    return op[15];
}

void ModelBuilder::set_name(const std::shared_ptr<ov::Node>& node) {
    node->set_friendly_name("node_" + std::to_string(m_name_idx++));
}

// ============================================================================
// Builder Interface
// ============================================================================

std::shared_ptr<ov::op::v0::Parameter> ModelBuilder::parameter(ov::element::Type type,
                                                               const ov::PartialShape& shape,
                                                               const std::string& name) {
    auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
    param->set_friendly_name(name);
    param->output(0).set_names({name});
    m_nodes.push_back(param);
    m_parameters.push_back(param);
    return param;
}

std::shared_ptr<ov::op::v0::Result> ModelBuilder::result(const ov::Output<ov::Node>& output, const std::string& name) {
    auto res = std::make_shared<ov::op::v0::Result>(output);
    res->set_friendly_name(name);
    res->output(0).set_names({name});
    m_nodes.push_back(res);
    m_results.push_back(res);
    return res;
}

std::shared_ptr<ov::Model> ModelBuilder::build(const std::string& name) {
    return std::make_shared<ov::Model>(ov::ResultVector(m_results.begin(), m_results.end()),
                                       ov::ParameterVector(m_parameters.begin(), m_parameters.end()),
                                       name);
}

void ModelBuilder::clear() {
    m_nodes.clear();
    m_parameters.clear();
    m_results.clear();
    m_name_idx = 0;
}

void ModelBuilder::track(const std::shared_ptr<ov::Node>& node) {
    m_nodes.push_back(node);
}

// ============================================================================
// Weight Compression Helper
// ============================================================================

ov::Output<ov::Node> ModelBuilder::make_weight(const std::string& name,
                                               size_t rows,
                                               size_t cols,
                                               WeightFormat format,
                                               ov::element::Type compute_precision) {
    switch (format) {
    case WeightFormat::FP32: {
        auto weight = ov::opset11::Constant::create(compute_precision,
                                                    ov::Shape{rows, cols},
                                                    std::vector<float>(rows * cols, 0.01f));
        weight->set_friendly_name(name);
        m_nodes.push_back(weight);
        return weight->output(0);
    }
    case WeightFormat::FP16: {
        // f16 weight -> Convert to compute precision
        auto weight = ov::opset11::Constant::create(ov::element::f16,
                                                    ov::Shape{rows, cols},
                                                    std::vector<float>(rows * cols, 0.01f));
        weight->set_friendly_name(name);
        m_nodes.push_back(weight);

        auto convert = std::make_shared<ov::opset11::Convert>(weight, compute_precision);
        convert->set_friendly_name(name + "_convert");
        m_nodes.push_back(convert);
        return convert->output(0);
    }
    case WeightFormat::INT8: {
        // i8 -> Convert(f16) -> Multiply(f16 scale) [-> Convert(compute) if needed]
        // Matches DCOFF SymmNoZP pattern for i8 weights
        auto weight =
            ov::opset11::Constant::create(ov::element::i8, ov::Shape{rows, cols}, std::vector<int8_t>(rows * cols, 1));
        weight->set_friendly_name(name);
        m_nodes.push_back(weight);

        auto convert = std::make_shared<ov::opset11::Convert>(weight, ov::element::f16);
        convert->set_friendly_name(name + "_convert");
        m_nodes.push_back(convert);

        auto scale =
            ov::opset11::Constant::create(ov::element::f16, ov::Shape{rows, 1}, std::vector<float>(rows, 0.01f));
        scale->set_friendly_name(name + "_scale");
        m_nodes.push_back(scale);

        auto scaled = std::make_shared<ov::opset11::Multiply>(convert, scale);
        scaled->set_friendly_name(name + "_decompress");
        m_nodes.push_back(scaled);

        if (compute_precision != ov::element::f16) {
            auto to_compute = std::make_shared<ov::opset11::Convert>(scaled, compute_precision);
            to_compute->set_friendly_name(name + "_to_compute");
            m_nodes.push_back(to_compute);
            return to_compute->output(0);
        }
        return scaled->output(0);
    }
    case WeightFormat::INT4: {
        // i4 -> Convert(f16) -> Multiply(f16 scale) [-> Convert(compute) if needed]
        // Matches DCOFF SymmNoZP pattern for i4 weights
        auto weight =
            ov::opset11::Constant::create(ov::element::i4, ov::Shape{rows, cols}, std::vector<int8_t>(rows * cols, 1));
        weight->set_friendly_name(name);
        m_nodes.push_back(weight);

        auto convert = std::make_shared<ov::opset11::Convert>(weight, ov::element::f16);
        convert->set_friendly_name(name + "_convert");
        m_nodes.push_back(convert);

        auto scale =
            ov::opset11::Constant::create(ov::element::f16, ov::Shape{rows, 1}, std::vector<float>(rows, 0.01f));
        scale->set_friendly_name(name + "_scale");
        m_nodes.push_back(scale);

        auto scaled = std::make_shared<ov::opset11::Multiply>(convert, scale);
        scaled->set_friendly_name(name + "_decompress");
        m_nodes.push_back(scaled);

        if (compute_precision != ov::element::f16) {
            auto to_compute = std::make_shared<ov::opset11::Convert>(scaled, compute_precision);
            to_compute->set_friendly_name(name + "_to_compute");
            m_nodes.push_back(to_compute);
            return to_compute->output(0);
        }
        return scaled->output(0);
    }
    default:
        return make_weight(name, rows, cols, WeightFormat::FP32, compute_precision);
    }
}

// ============================================================================
// Atomic Building Blocks
// ============================================================================

ov::Output<ov::Node> ModelBuilder::make_linear(const ov::Output<ov::Node>& input,
                                               size_t out_features,
                                               const std::string& name,
                                               ov::element::Type precision,
                                               bool add_bias,
                                               WeightFormat weight_format) {
    const auto& shape = input.get_partial_shape();
    const size_t in_features = shape.rank().is_static() && shape[shape.rank().get_length() - 1].is_static()
                                   ? static_cast<size_t>(shape[shape.rank().get_length() - 1].get_length())
                                   : 64;

    auto weight_output = make_weight(name + ".weight", out_features, in_features, weight_format, precision);

    auto matmul = std::make_shared<ov::opset11::MatMul>(input, weight_output, false, true);
    matmul->set_friendly_name(name);
    m_nodes.push_back(matmul);

    if (add_bias) {
        auto bias =
            ov::opset11::Constant::create(precision, ov::Shape{out_features}, std::vector<float>(out_features, 0.0f));
        bias->set_friendly_name(name + ".bias");
        m_nodes.push_back(bias);

        auto add = std::make_shared<ov::opset11::Add>(matmul, bias);
        add->set_friendly_name(name + "_bias_add");
        m_nodes.push_back(add);
        return add->output(0);
    }

    return matmul->output(0);
}

ov::Output<ov::Node> ModelBuilder::make_layer_norm(const ov::Output<ov::Node>& input,
                                                   size_t hidden_size,
                                                   const std::string& name,
                                                   ov::element::Type precision,
                                                   float eps) {
    auto weight =
        ov::opset11::Constant::create(precision, ov::Shape{hidden_size}, std::vector<float>(hidden_size, 1.0f));
    weight->set_friendly_name(name + ".weight");
    m_nodes.push_back(weight);

    auto bias = ov::opset11::Constant::create(precision, ov::Shape{hidden_size}, std::vector<float>(hidden_size, 0.0f));
    bias->set_friendly_name(name + ".bias");
    m_nodes.push_back(bias);

    auto axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    m_nodes.push_back(axes);

    auto mvn = std::make_shared<ov::opset11::MVN>(input, axes, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT);
    mvn->set_friendly_name(name + "_mvn");
    m_nodes.push_back(mvn);

    auto mul = std::make_shared<ov::opset11::Multiply>(mvn, weight);
    m_nodes.push_back(mul);

    auto add = std::make_shared<ov::opset11::Add>(mul, bias);
    add->set_friendly_name(name);
    m_nodes.push_back(add);

    return add->output(0);
}

ov::Output<ov::Node> ModelBuilder::make_rms_norm(const ov::Output<ov::Node>& input,
                                                 size_t hidden_size,
                                                 const std::string& name,
                                                 ov::element::Type precision,
                                                 float eps) {
    auto weight =
        ov::opset11::Constant::create(precision, ov::Shape{hidden_size}, std::vector<float>(hidden_size, 1.0f));
    weight->set_friendly_name(name + ".weight");
    m_nodes.push_back(weight);

    auto squared = std::make_shared<ov::opset11::Multiply>(input, input);
    m_nodes.push_back(squared);

    auto axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    m_nodes.push_back(axes);

    auto mean = std::make_shared<ov::opset11::ReduceMean>(squared, axes, true);
    m_nodes.push_back(mean);

    auto eps_const = ov::opset11::Constant::create(precision, ov::Shape{}, {eps});
    m_nodes.push_back(eps_const);

    auto mean_eps = std::make_shared<ov::opset11::Add>(mean, eps_const);
    m_nodes.push_back(mean_eps);

    auto rsqrt = std::make_shared<ov::opset11::Sqrt>(mean_eps);
    m_nodes.push_back(rsqrt);

    auto normalized = std::make_shared<ov::opset11::Divide>(input, rsqrt);
    m_nodes.push_back(normalized);

    auto scaled = std::make_shared<ov::opset11::Multiply>(normalized, weight);
    scaled->set_friendly_name(name);
    m_nodes.push_back(scaled);

    return scaled->output(0);
}

// ============================================================================
// Dispatchers
// ============================================================================

ov::Output<ov::Node> ModelBuilder::make_norm(const ov::Output<ov::Node>& input,
                                             size_t hidden_size,
                                             const std::string& name,
                                             NormType type,
                                             ov::element::Type precision,
                                             float eps) {
    switch (type) {
    case NormType::LAYER_NORM:
        return make_layer_norm(input, hidden_size, name, precision, eps);
    case NormType::RMS_NORM:
        return make_rms_norm(input, hidden_size, name, precision, eps);
    default:
        return make_layer_norm(input, hidden_size, name, precision, eps);
    }
}

ov::Output<ov::Node> ModelBuilder::make_ffn(const ov::Output<ov::Node>& input,
                                            size_t hidden_size,
                                            size_t intermediate_size,
                                            const std::string& name,
                                            FFNType type,
                                            ov::element::Type precision,
                                            WeightFormat weight_format) {
    switch (type) {
    case FFNType::SWIGLU:
        return make_swiglu_ffn(input, hidden_size, intermediate_size, name, precision, weight_format);
    case FFNType::GELU:
        return make_gelu_ffn(input, hidden_size, intermediate_size, name, precision, weight_format);
    default:
        return make_swiglu_ffn(input, hidden_size, intermediate_size, name, precision, weight_format);
    }
}

// ============================================================================
// Attention Components
// ============================================================================

ov::Output<ov::Node> ModelBuilder::make_multihead_reshape(const ov::Output<ov::Node>& input,
                                                          size_t num_heads,
                                                          size_t head_dim,
                                                          const std::string& name) {
    auto shape = ov::opset11::Constant::create(
        ov::element::i64,
        ov::Shape{4},
        std::vector<int64_t>{0, -1, static_cast<int64_t>(num_heads), static_cast<int64_t>(head_dim)});
    m_nodes.push_back(shape);

    auto reshape = std::make_shared<ov::opset11::Reshape>(input, shape, true);
    reshape->set_friendly_name(name);
    m_nodes.push_back(reshape);

    return reshape->output(0);
}

ov::Output<ov::Node> ModelBuilder::make_attention_transpose(const ov::Output<ov::Node>& input,
                                                            const std::string& name,
                                                            bool reverse) {
    std::vector<int64_t> order = reverse ? std::vector<int64_t>{0, 2, 1, 3} : std::vector<int64_t>{0, 2, 1, 3};
    auto order_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{4}, order);
    m_nodes.push_back(order_const);

    auto transpose = std::make_shared<ov::opset11::Transpose>(input, order_const);
    transpose->set_friendly_name(name);
    m_nodes.push_back(transpose);

    return transpose->output(0);
}

ov::Output<ov::Node> ModelBuilder::make_repeat_kv(const ov::Output<ov::Node>& kv,
                                                  size_t num_heads,
                                                  size_t num_kv_heads,
                                                  const std::string& name) {
    if (num_heads == num_kv_heads) {
        return kv;
    }

    const size_t n_rep = num_heads / num_kv_heads;

    // Input: [batch, kv_heads, seq, head_dim]
    // Build broadcast target shape from kv (Concat output, BEFORE Unsqueeze).
    // ShapeOf must be on the Concat output to match NPUW's AttentionBroadcast
    // regularization pattern: Concat → ShapeOf → Gather → Concat(shape) → Broadcast(Unsqueeze, shape)
    auto shape_of_kv = std::make_shared<ov::opset11::ShapeOf>(kv, ov::element::i64);
    m_nodes.push_back(shape_of_kv);

    auto gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    m_nodes.push_back(gather_axis);

    // Gather [batch, kv_heads] as 2 elements from shape
    auto idx_01 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    m_nodes.push_back(idx_01);
    auto batch_kv_heads = std::make_shared<ov::opset11::Gather>(shape_of_kv, idx_01, gather_axis);
    m_nodes.push_back(batch_kv_heads);

    auto n_rep_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(n_rep)});
    m_nodes.push_back(n_rep_const);

    // Gather [seq, head_dim] as 2 elements from shape
    auto idx_23 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
    m_nodes.push_back(idx_23);
    auto seq_head_dim = std::make_shared<ov::opset11::Gather>(shape_of_kv, idx_23, gather_axis);
    m_nodes.push_back(seq_head_dim);

    // target_shape = [batch, kv_heads, n_rep, seq, head_dim]
    auto broadcast_shape =
        std::make_shared<ov::opset11::Concat>(ov::OutputVector{batch_kv_heads, n_rep_const, seq_head_dim}, 0);
    broadcast_shape->set_friendly_name(name + "_broadcast_shape");
    m_nodes.push_back(broadcast_shape);

    // Unsqueeze to: [batch, kv_heads, 1, seq, head_dim]
    auto unsqueeze_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    m_nodes.push_back(unsqueeze_axis);

    auto unsqueezed = std::make_shared<ov::opset11::Unsqueeze>(kv, unsqueeze_axis);
    unsqueezed->set_friendly_name(name + "_unsqueeze");
    m_nodes.push_back(unsqueezed);

    auto broadcasted =
        std::make_shared<ov::op::v3::Broadcast>(unsqueezed, broadcast_shape, ov::op::BroadcastType::BIDIRECTIONAL);
    broadcasted->set_friendly_name(name + "_broadcast");
    m_nodes.push_back(broadcasted);

    // Reshape to: [batch, num_heads, seq, head_dim]
    // Use special_zero=true so 0 means "copy from input" for the batch dimension
    auto new_shape = ov::opset11::Constant::create(
        ov::element::i64,
        ov::Shape{4},
        std::vector<int64_t>{0,
                             static_cast<int64_t>(num_heads),
                             -1,
                             static_cast<int64_t>(kv.get_partial_shape()[3].get_length())});
    new_shape->set_friendly_name(name + "_shape");
    m_nodes.push_back(new_shape);

    auto reshaped = std::make_shared<ov::opset11::Reshape>(broadcasted, new_shape, true);
    reshaped->set_friendly_name(name);
    m_nodes.push_back(reshaped);

    return reshaped->output(0);
}

ModelBuilder::KVCacheResult ModelBuilder::make_kv_cache_concat(const ov::Output<ov::Node>& current_kv,
                                                               const ov::Output<ov::Node>& batch_source,
                                                               const ov::Output<ov::Node>& beam_idx,
                                                               size_t num_heads,
                                                               size_t head_dim,
                                                               const std::string& name,
                                                               ov::element::Type precision) {
    // Create variable for stateful KV cache
    auto var_shape = ov::PartialShape{-1, static_cast<int64_t>(num_heads), -1, static_cast<int64_t>(head_dim)};
    auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{var_shape, precision, name});

    // Build init shape dynamically: [batch, num_heads, 0, head_dim]
    // Extract batch dimension from batch_source (e.g. input_ids) using ShapeOf + Gather
    // batch_source shape is typically [batch, seq] for input_ids
    auto shape_of = std::make_shared<ov::opset11::ShapeOf>(batch_source, ov::element::i64);
    shape_of->set_friendly_name(name + "_shapeof");
    m_nodes.push_back(shape_of);

    auto zero_idx = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    m_nodes.push_back(zero_idx);
    auto gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    m_nodes.push_back(gather_axis);

    auto batch_dim = std::make_shared<ov::opset11::Gather>(shape_of, zero_idx, gather_axis);
    batch_dim->set_friendly_name(name + "_batch_dim");
    m_nodes.push_back(batch_dim);

    // Create constants for other dimensions
    auto num_heads_const = ov::opset11::Constant::create(ov::element::i64,
                                                         ov::Shape{1},
                                                         std::vector<int64_t>{static_cast<int64_t>(num_heads)});
    m_nodes.push_back(num_heads_const);
    auto zero_seq = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    m_nodes.push_back(zero_seq);
    auto head_dim_const = ov::opset11::Constant::create(ov::element::i64,
                                                        ov::Shape{1},
                                                        std::vector<int64_t>{static_cast<int64_t>(head_dim)});
    m_nodes.push_back(head_dim_const);

    // Concat to build shape: [batch, num_heads, 0, head_dim]
    auto init_shape = std::make_shared<ov::opset11::Concat>(ov::OutputVector{batch_dim->output(0),
                                                                             num_heads_const->output(0),
                                                                             zero_seq->output(0),
                                                                             head_dim_const->output(0)},
                                                            0);
    init_shape->set_friendly_name(name + "_init_shape");
    m_nodes.push_back(init_shape);

    // Broadcast a scalar 0.0 to the init shape
    auto zero_scalar = ov::opset11::Constant::create(precision, ov::Shape{}, std::vector<float>{0.0f});
    m_nodes.push_back(zero_scalar);

    auto init_value = std::make_shared<ov::opset11::Broadcast>(zero_scalar, init_shape);
    init_value->set_friendly_name(name + "_init");
    m_nodes.push_back(init_value);

    // ReadValue reads the cached state (using init_value on first run)
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(init_value, variable);
    read_value->set_friendly_name(name + "_read");
    m_nodes.push_back(read_value);

    // Gather by beam_idx for beam search reordering along batch axis (axis=0)
    // This is required by StatefulToStateless transformation which traces
    // beam_idx -> Gather -> ReadValue to find KV cache variables
    auto beam_gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0});
    m_nodes.push_back(beam_gather_axis);

    auto beam_gather = std::make_shared<ov::opset11::Gather>(read_value, beam_idx, beam_gather_axis);
    beam_gather->set_friendly_name(name + "_beam_gather");
    m_nodes.push_back(beam_gather);

    // Concat past cache with current KV along sequence dimension
    auto concat = std::make_shared<ov::opset11::Concat>(ov::OutputVector{beam_gather->output(0), current_kv}, 2);
    concat->set_friendly_name(name + "_concat");
    m_nodes.push_back(concat);

    // Assign writes the updated cache back to the variable
    auto assign = std::make_shared<ov::op::v6::Assign>(concat, variable);
    assign->set_friendly_name(name + "_assign");
    m_nodes.push_back(assign);

    KVCacheResult result;
    result.concatenated = concat->output(0);
    result.assign = assign;
    return result;
}

ov::Output<ov::Node> ModelBuilder::make_attention(const ov::Output<ov::Node>& q,
                                                  const ov::Output<ov::Node>& k,
                                                  const ov::Output<ov::Node>& v,
                                                  size_t head_dim,
                                                  const std::string& name,
                                                  ov::element::Type precision,
                                                  const ov::Output<ov::Node>& attention_mask) {
    // Use native ScaledDotProductAttention op (v13) which is required by
    // SDPAToPagedAttention transformation and compatible with all backends.
    // Input shapes: Q,K,V = [batch, heads, seq, head_dim], mask = [batch, 1, seq_len, total_seq]

    std::shared_ptr<ov::op::v13::ScaledDotProductAttention> sdpa;
    if (attention_mask.get_node()) {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, attention_mask, false);
    } else {
        sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q, k, v, false);
    }
    sdpa->set_friendly_name(name + ".sdpa");
    m_nodes.push_back(sdpa);

    return sdpa->output(0);
}

// ============================================================================
// Rotary Position Embedding (RoPE)
// ============================================================================

ov::Output<ov::Node> ModelBuilder::make_rope(const ov::Output<ov::Node>& input,
                                             const ov::Output<ov::Node>& position_ids,
                                             size_t head_dim,
                                             size_t max_position,
                                             const std::string& name,
                                             RoPEType type,
                                             ov::element::Type precision) {
    switch (type) {
    case RoPEType::HALF_ROTATION:
        return make_half_rotation_rope(input, position_ids, head_dim, max_position, name, precision);
    case RoPEType::INTERLEAVED:
        return make_interleaved_rope(input, position_ids, head_dim, max_position, name, precision);
    default:
        return make_half_rotation_rope(input, position_ids, head_dim, max_position, name, precision);
    }
}

ov::Output<ov::Node> ModelBuilder::make_half_rotation_rope(const ov::Output<ov::Node>& input,
                                                           const ov::Output<ov::Node>& position_ids,
                                                           size_t head_dim,
                                                           size_t max_position,
                                                           const std::string& name,
                                                           ov::element::Type precision) {
    // input: [batch, seq, heads, head_dim]
    // position_ids: [batch, seq]

    // Cos/sin embedding tables: [max_position, head_dim]
    auto cos_table = ov::opset11::Constant::create(precision,
                                                   ov::Shape{max_position, head_dim},
                                                   std::vector<float>(max_position * head_dim, 0.5f));
    cos_table->set_friendly_name(name + ".cos_table");
    m_nodes.push_back(cos_table);

    auto sin_table = ov::opset11::Constant::create(precision,
                                                   ov::Shape{max_position, head_dim},
                                                   std::vector<float>(max_position * head_dim, 0.5f));
    sin_table->set_friendly_name(name + ".sin_table");
    m_nodes.push_back(sin_table);

    // Gather cos/sin by position: [batch, seq, head_dim]
    auto gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    m_nodes.push_back(gather_axis);

    auto cos_embed = std::make_shared<ov::opset11::Gather>(cos_table, position_ids, gather_axis, 0);
    cos_embed->set_friendly_name(name + "_cos_gather");
    m_nodes.push_back(cos_embed);

    auto sin_embed = std::make_shared<ov::opset11::Gather>(sin_table, position_ids, gather_axis, 0);
    sin_embed->set_friendly_name(name + "_sin_gather");
    m_nodes.push_back(sin_embed);

    // Unsqueeze to [batch, seq, 1, head_dim] for broadcasting with [batch, seq, heads, head_dim]
    auto unsqueeze_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    m_nodes.push_back(unsqueeze_axis);

    auto cos_unsqueezed = std::make_shared<ov::opset11::Unsqueeze>(cos_embed, unsqueeze_axis);
    cos_unsqueezed->set_friendly_name(name + "_cos_unsqueeze");
    m_nodes.push_back(cos_unsqueezed);

    auto sin_unsqueezed = std::make_shared<ov::opset11::Unsqueeze>(sin_embed, unsqueeze_axis);
    sin_unsqueezed->set_friendly_name(name + "_sin_unsqueeze");
    m_nodes.push_back(sin_unsqueezed);

    // Split input in half: x1 = input[..., :half], x2 = input[..., half:]
    const int64_t half = static_cast<int64_t>(head_dim / 2);
    auto zero = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto half_const = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {half});
    auto head_dim_const =
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(head_dim)});
    auto last_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    m_nodes.push_back(zero);
    m_nodes.push_back(half_const);
    m_nodes.push_back(head_dim_const);
    m_nodes.push_back(last_axis);
    m_nodes.push_back(step);

    auto x1 = std::make_shared<ov::op::v8::Slice>(input, zero, half_const, step, last_axis);
    x1->set_friendly_name(name + "_x1");
    m_nodes.push_back(x1);

    auto x2 = std::make_shared<ov::op::v8::Slice>(input, half_const, head_dim_const, step, last_axis);
    x2->set_friendly_name(name + "_x2");
    m_nodes.push_back(x2);

    // Rotate: [-x2, x1]
    auto neg_one = ov::opset11::Constant::create(precision, ov::Shape{}, {-1.0f});
    m_nodes.push_back(neg_one);

    auto neg_x2 = std::make_shared<ov::opset11::Multiply>(x2, neg_one);
    neg_x2->set_friendly_name(name + "_neg_x2");
    m_nodes.push_back(neg_x2);

    auto rotated = std::make_shared<ov::opset11::Concat>(ov::OutputVector{neg_x2, x1}, -1);
    rotated->set_friendly_name(name + "_rotated");
    m_nodes.push_back(rotated);

    // output = input * cos + rotated * sin
    auto input_cos = std::make_shared<ov::opset11::Multiply>(input, cos_unsqueezed);
    input_cos->set_friendly_name(name + "_input_cos");
    m_nodes.push_back(input_cos);

    auto rotated_sin = std::make_shared<ov::opset11::Multiply>(rotated, sin_unsqueezed);
    rotated_sin->set_friendly_name(name + "_rotated_sin");
    m_nodes.push_back(rotated_sin);

    auto output = std::make_shared<ov::opset11::Add>(input_cos, rotated_sin);
    output->set_friendly_name(name);
    m_nodes.push_back(output);

    return output->output(0);
}

ov::Output<ov::Node> ModelBuilder::make_interleaved_rope(const ov::Output<ov::Node>& input,
                                                         const ov::Output<ov::Node>& position_ids,
                                                         size_t head_dim,
                                                         size_t max_position,
                                                         const std::string& name,
                                                         ov::element::Type precision) {
    // input: [batch, seq, heads, head_dim]
    // position_ids: [batch, seq]

    // Cos/sin embedding tables: [max_position, head_dim]
    auto cos_table = ov::opset11::Constant::create(precision,
                                                   ov::Shape{max_position, head_dim},
                                                   std::vector<float>(max_position * head_dim, 0.5f));
    cos_table->set_friendly_name(name + ".cos_table");
    m_nodes.push_back(cos_table);

    auto sin_table = ov::opset11::Constant::create(precision,
                                                   ov::Shape{max_position, head_dim},
                                                   std::vector<float>(max_position * head_dim, 0.5f));
    sin_table->set_friendly_name(name + ".sin_table");
    m_nodes.push_back(sin_table);

    // Gather cos/sin by position: [batch, seq, head_dim]
    auto gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    m_nodes.push_back(gather_axis);

    auto cos_embed = std::make_shared<ov::opset11::Gather>(cos_table, position_ids, gather_axis, 0);
    cos_embed->set_friendly_name(name + "_cos_gather");
    m_nodes.push_back(cos_embed);

    auto sin_embed = std::make_shared<ov::opset11::Gather>(sin_table, position_ids, gather_axis, 0);
    sin_embed->set_friendly_name(name + "_sin_gather");
    m_nodes.push_back(sin_embed);

    // Unsqueeze to [batch, seq, 1, head_dim]
    auto unsqueeze_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    m_nodes.push_back(unsqueeze_axis);

    auto cos_unsqueezed = std::make_shared<ov::opset11::Unsqueeze>(cos_embed, unsqueeze_axis);
    cos_unsqueezed->set_friendly_name(name + "_cos_unsqueeze");
    m_nodes.push_back(cos_unsqueezed);

    auto sin_unsqueezed = std::make_shared<ov::opset11::Unsqueeze>(sin_embed, unsqueeze_axis);
    sin_unsqueezed->set_friendly_name(name + "_sin_unsqueeze");
    m_nodes.push_back(sin_unsqueezed);

    // Interleaved rotation:
    // Reshape [batch, seq, heads, head_dim] -> [batch, seq, heads, head_dim/2, 2]
    const int64_t half_dim = static_cast<int64_t>(head_dim / 2);
    auto reshape_5d =
        ov::opset11::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{0, 0, 0, half_dim, 2});
    m_nodes.push_back(reshape_5d);

    auto reshaped = std::make_shared<ov::opset11::Reshape>(input, reshape_5d, true);
    reshaped->set_friendly_name(name + "_reshape_5d");
    m_nodes.push_back(reshaped);

    // Get element 0 (even) and element 1 (odd) from last dim
    auto zero = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto one = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto two = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto last_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    m_nodes.push_back(zero);
    m_nodes.push_back(one);
    m_nodes.push_back(two);
    m_nodes.push_back(step);
    m_nodes.push_back(last_axis);

    // x_even = reshaped[..., 0:1], x_odd = reshaped[..., 1:2]
    auto x_even = std::make_shared<ov::op::v8::Slice>(reshaped, zero, one, step, last_axis);
    x_even->set_friendly_name(name + "_x_even");
    m_nodes.push_back(x_even);

    auto x_odd = std::make_shared<ov::op::v8::Slice>(reshaped, one, two, step, last_axis);
    x_odd->set_friendly_name(name + "_x_odd");
    m_nodes.push_back(x_odd);

    // Rotate: [-x_odd, x_even]
    auto neg_one = ov::opset11::Constant::create(precision, ov::Shape{}, {-1.0f});
    m_nodes.push_back(neg_one);

    auto neg_x_odd = std::make_shared<ov::opset11::Multiply>(x_odd, neg_one);
    neg_x_odd->set_friendly_name(name + "_neg_x_odd");
    m_nodes.push_back(neg_x_odd);

    // Stack [-x_odd, x_even] along last dim: [batch, seq, heads, head_dim/2, 2]
    auto rotated_pairs = std::make_shared<ov::opset11::Concat>(ov::OutputVector{neg_x_odd, x_even}, -1);
    rotated_pairs->set_friendly_name(name + "_rotated_pairs");
    m_nodes.push_back(rotated_pairs);

    // Reshape back: [batch, seq, heads, head_dim]
    auto reshape_4d = ov::opset11::Constant::create(ov::element::i64,
                                                    ov::Shape{4},
                                                    std::vector<int64_t>{0, 0, 0, static_cast<int64_t>(head_dim)});
    m_nodes.push_back(reshape_4d);

    auto rotated = std::make_shared<ov::opset11::Reshape>(rotated_pairs, reshape_4d, true);
    rotated->set_friendly_name(name + "_rotated");
    m_nodes.push_back(rotated);

    // output = input * cos + rotated * sin
    auto input_cos = std::make_shared<ov::opset11::Multiply>(input, cos_unsqueezed);
    input_cos->set_friendly_name(name + "_input_cos");
    m_nodes.push_back(input_cos);

    auto rotated_sin = std::make_shared<ov::opset11::Multiply>(rotated, sin_unsqueezed);
    rotated_sin->set_friendly_name(name + "_rotated_sin");
    m_nodes.push_back(rotated_sin);

    auto output = std::make_shared<ov::opset11::Add>(input_cos, rotated_sin);
    output->set_friendly_name(name);
    m_nodes.push_back(output);

    return output->output(0);
}

// ============================================================================
// FFN Implementations
// ============================================================================

ov::Output<ov::Node> ModelBuilder::make_swiglu_ffn(const ov::Output<ov::Node>& input,
                                                   size_t hidden_size,
                                                   size_t intermediate_size,
                                                   const std::string& name,
                                                   ov::element::Type precision,
                                                   WeightFormat weight_format) {
    auto gate = make_linear(input, intermediate_size, name + ".gate_proj", precision, false, weight_format);
    auto up = make_linear(input, intermediate_size, name + ".up_proj", precision, false, weight_format);

    auto sigmoid = std::make_shared<ov::opset11::Sigmoid>(gate);
    m_nodes.push_back(sigmoid);

    auto silu = std::make_shared<ov::opset11::Multiply>(gate, sigmoid);
    silu->set_friendly_name(name + "_silu");
    m_nodes.push_back(silu);

    auto gate_up = std::make_shared<ov::opset11::Multiply>(silu, up);
    gate_up->set_friendly_name(name + "_gate_up");
    m_nodes.push_back(gate_up);

    auto down = make_linear(gate_up, hidden_size, name + ".down_proj", precision, false, weight_format);

    return down;
}

ov::Output<ov::Node> ModelBuilder::make_gelu_ffn(const ov::Output<ov::Node>& input,
                                                 size_t hidden_size,
                                                 size_t intermediate_size,
                                                 const std::string& name,
                                                 ov::element::Type precision,
                                                 WeightFormat weight_format) {
    auto up = make_linear(input, intermediate_size, name + ".up_proj", precision, false, weight_format);

    auto gelu = std::make_shared<ov::opset11::Gelu>(up);
    gelu->set_friendly_name(name + "_gelu");
    m_nodes.push_back(gelu);

    auto down = make_linear(gelu, hidden_size, name + ".down_proj", precision, false, weight_format);

    return down;
}

// ============================================================================
// Composite Blocks
// ============================================================================

LayerResult ModelBuilder::make_attention_block(const ov::Output<ov::Node>& input,
                                               const LLMConfig& config,
                                               size_t layer_idx,
                                               const std::string& prefix,
                                               const ov::Output<ov::Node>& position_ids,
                                               const ov::Output<ov::Node>& batch_source,
                                               const ov::Output<ov::Node>& beam_idx,
                                               const ov::Output<ov::Node>& attention_mask) {
    const size_t H = config.hidden_size;
    const size_t num_heads = config.num_heads;
    const size_t kv_heads = config.get_kv_heads();
    const size_t head_dim = config.head_dim;
    const auto prec = config.precision;
    const auto wf = config.weight_format;

    // Pre-attention norm
    auto normed = make_norm(input, H, prefix + "input_layernorm", config.norm_type, prec);

    // Q, K, V projections
    auto q = make_linear(normed, num_heads * head_dim, prefix + "self_attn.q_proj", prec, false, wf);
    auto k = make_linear(normed, kv_heads * head_dim, prefix + "self_attn.k_proj", prec, false, wf);
    auto v = make_linear(normed, kv_heads * head_dim, prefix + "self_attn.v_proj", prec, false, wf);

    // Reshape for multi-head: [batch, seq, heads, head_dim]
    auto q_reshaped = make_multihead_reshape(q, num_heads, head_dim, prefix + "q_reshape");
    auto k_reshaped = make_multihead_reshape(k, kv_heads, head_dim, prefix + "k_reshape");
    auto v_reshaped = make_multihead_reshape(v, kv_heads, head_dim, prefix + "v_reshape");

    // Apply RoPE to Q and K (before transpose, in [batch, seq, heads, head_dim] format)
    ov::Output<ov::Node> q_for_trans = q_reshaped;
    ov::Output<ov::Node> k_for_trans = k_reshaped;
    if (config.use_position_ids && position_ids.get_node() != nullptr) {
        q_for_trans = make_rope(q_reshaped,
                                position_ids,
                                head_dim,
                                config.context_len,
                                prefix + "q_rope",
                                config.rope_type,
                                prec);
        k_for_trans = make_rope(k_reshaped,
                                position_ids,
                                head_dim,
                                config.context_len,
                                prefix + "k_rope",
                                config.rope_type,
                                prec);
    }

    // Transpose: [batch, seq, heads, dim] -> [batch, heads, seq, dim]
    auto q_trans = make_attention_transpose(q_for_trans, prefix + "q_transpose");
    auto k_trans = make_attention_transpose(k_for_trans, prefix + "k_transpose");
    auto v_trans = make_attention_transpose(v_reshaped, prefix + "v_transpose");

    // KV cache (if enabled)
    std::vector<std::shared_ptr<ov::Node>> sinks;
    ov::Output<ov::Node> k_for_attn = k_trans;
    ov::Output<ov::Node> v_for_attn = v_trans;

    if (config.use_kv_cache) {
        // Variable IDs follow the naming convention expected by StatefulToStateless:
        //   "past_key_values.N.keypresent.N.key" (input name + output name concatenated)
        // This ensures parameter names in the converted stateless model are "past_key_values.N.key"
        // (without "input_restored." prefix), which is required by NPUW's HFA/Pyramid attention detection.
        auto layer_str = std::to_string(layer_idx);
        auto k_cache = make_kv_cache_concat(k_trans,
                                            batch_source,
                                            beam_idx,
                                            kv_heads,
                                            head_dim,
                                            "past_key_values." + layer_str + ".key" + "present." + layer_str + ".key",
                                            prec);
        auto v_cache =
            make_kv_cache_concat(v_trans,
                                 batch_source,
                                 beam_idx,
                                 kv_heads,
                                 head_dim,
                                 "past_key_values." + layer_str + ".value" + "present." + layer_str + ".value",
                                 prec);

        sinks.push_back(k_cache.assign);
        sinks.push_back(v_cache.assign);
        k_for_attn = k_cache.concatenated;
        v_for_attn = v_cache.concatenated;
    }

    // For GQA: repeat K/V heads to match Q head count
    auto k_expanded = make_repeat_kv(k_for_attn, num_heads, kv_heads, prefix + "k_repeat");
    auto v_expanded = make_repeat_kv(v_for_attn, num_heads, kv_heads, prefix + "v_repeat");

    // SDPA (attention_mask is expected to be pre-transformed 4D float mask)
    auto attn_output = make_attention(q_trans, k_expanded, v_expanded, head_dim, prefix + "attn", prec, attention_mask);

    // Transpose back and reshape
    auto attn_trans = make_attention_transpose(attn_output, prefix + "attn_out_transpose", true);

    auto reshape_shape = ov::opset11::Constant::create(ov::element::i64,
                                                       ov::Shape{3},
                                                       std::vector<int64_t>{0, -1, static_cast<int64_t>(H)});
    m_nodes.push_back(reshape_shape);

    auto attn_reshaped = std::make_shared<ov::opset11::Reshape>(attn_trans, reshape_shape, true);
    attn_reshaped->set_friendly_name(prefix + "attn_reshape");
    m_nodes.push_back(attn_reshaped);

    // Output projection
    auto o_proj = make_linear(attn_reshaped->output(0), H, prefix + "self_attn.o_proj", prec, false, wf);

    // Residual connection
    auto residual = std::make_shared<ov::opset11::Add>(input, o_proj);
    residual->set_friendly_name(prefix + "attn_residual");
    m_nodes.push_back(residual);

    return {residual->output(0), sinks};
}

ov::Output<ov::Node> ModelBuilder::make_ffn_block(const ov::Output<ov::Node>& input,
                                                  const LLMConfig& config,
                                                  const std::string& prefix) {
    auto normed =
        make_norm(input, config.hidden_size, prefix + "post_attention_layernorm", config.norm_type, config.precision);

    auto ffn_out = make_ffn(normed,
                            config.hidden_size,
                            config.intermediate_size,
                            prefix + "mlp",
                            config.ffn_type,
                            config.precision,
                            config.weight_format);

    auto residual = std::make_shared<ov::opset11::Add>(input, ffn_out);
    residual->set_friendly_name(prefix + "ffn_residual");
    m_nodes.push_back(residual);

    return residual->output(0);
}

LayerResult ModelBuilder::make_decoder_layer(const ov::Output<ov::Node>& input,
                                             const LLMConfig& config,
                                             size_t layer_idx,
                                             const ov::Output<ov::Node>& position_ids,
                                             const ov::Output<ov::Node>& batch_source,
                                             const ov::Output<ov::Node>& beam_idx,
                                             const ov::Output<ov::Node>& attention_mask) {
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";

    auto attn_result =
        make_attention_block(input, config, layer_idx, prefix, position_ids, batch_source, beam_idx, attention_mask);
    auto ffn_out = make_ffn_block(attn_result.output, config, prefix);

    return {ffn_out, attn_result.sinks};
}

ov::Output<ov::Node> ModelBuilder::make_embedding(const ov::Output<ov::Node>& input_ids,
                                                  size_t vocab_size,
                                                  size_t hidden_size,
                                                  const std::string& name,
                                                  ov::element::Type precision) {
    auto weight = ov::opset11::Constant::create(precision,
                                                ov::Shape{vocab_size, hidden_size},
                                                std::vector<float>(vocab_size * hidden_size, 0.01f));
    weight->set_friendly_name(name + ".weight");
    m_nodes.push_back(weight);

    auto axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
    m_nodes.push_back(axis);

    auto gather = std::make_shared<ov::opset11::Gather>(weight, input_ids, axis, 0);
    gather->set_friendly_name(name);
    m_nodes.push_back(gather);

    return gather->output(0);
}

ov::Output<ov::Node> ModelBuilder::make_lm_head(const ov::Output<ov::Node>& hidden_states,
                                                size_t hidden_size,
                                                size_t vocab_size,
                                                const std::string& name,
                                                ov::element::Type precision,
                                                WeightFormat weight_format) {
    return make_linear(hidden_states, vocab_size, name, precision, false, weight_format);
}

// ============================================================================
// LLM Model Builder (convenience wrapper using building blocks)
// ============================================================================

std::shared_ptr<ov::Model> ModelBuilder::build_llm(const LLMConfig& config) {
    clear();

    const auto prec = config.precision;

    // ===== LLM Inputs =====
    auto input_ids = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "input_ids");
    auto attention_mask = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "attention_mask");

    ov::Output<ov::Node> position_ids_output;
    if (config.use_position_ids) {
        auto position_ids_param = parameter(ov::element::i64, ov::PartialShape{-1, -1}, "position_ids");
        position_ids_output = position_ids_param->output(0);
    }

    // beam_idx is required for stateful models used with LLMPipeline
    auto beam_idx = parameter(ov::element::i32, ov::PartialShape{-1}, "beam_idx");

    // ===== HEAD: Token Embedding =====
    auto hidden_states =
        make_embedding(input_ids->output(0), config.vocab_size, config.hidden_size, "model.embed_tokens", prec);

    // ===== Attention mask: create proper 4D causal + padding mask =====
    // Convert [batch, total_seq] i64 mask to [batch, 1, seq_len, total_seq] float additive mask.
    // Uses causal (lower triangular) mask combined with padding mask,
    // producing the 4D shape required by NPUW HFA tiled attention.
    // All layers share the same pre-transformed mask node (required for NPUW repeating block detection)
    ov::Output<ov::Node> sdpa_mask;
    {
        // --- Padding mask component: [batch, total_seq] -> [batch, 1, 1, total_seq] ---
        auto mask_float = std::make_shared<ov::opset11::Convert>(attention_mask->output(0), prec);
        mask_float->set_friendly_name("model.mask_convert");
        m_nodes.push_back(mask_float);

        auto one_const = ov::opset11::Constant::create(prec, ov::Shape{}, {1.0f});
        m_nodes.push_back(one_const);
        auto inv_mask = std::make_shared<ov::opset11::Subtract>(one_const, mask_float);
        inv_mask->set_friendly_name("model.mask_invert");
        m_nodes.push_back(inv_mask);

        auto neg_inf = ov::opset11::Constant::create(prec, ov::Shape{}, {-10000.0f});
        m_nodes.push_back(neg_inf);
        auto padding_mask = std::make_shared<ov::opset11::Multiply>(inv_mask, neg_inf);
        padding_mask->set_friendly_name("model.padding_mask");
        m_nodes.push_back(padding_mask);

        auto pad_shape =
            ov::opset11::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 1, 1, -1});
        m_nodes.push_back(pad_shape);
        auto padding_4d = std::make_shared<ov::opset11::Reshape>(padding_mask, pad_shape, true);
        padding_4d->set_friendly_name("model.padding_mask_4d");
        m_nodes.push_back(padding_4d);

        // --- Causal mask component: [1, 1, seq_len, total_seq] ---
        // Extract dynamic dimensions
        auto ids_shape = std::make_shared<ov::opset11::ShapeOf>(input_ids, ov::element::i64);
        ids_shape->set_friendly_name("model.ids_shape");
        m_nodes.push_back(ids_shape);

        auto mask_shape_node = std::make_shared<ov::opset11::ShapeOf>(attention_mask, ov::element::i64);
        mask_shape_node->set_friendly_name("model.mask_shape");
        m_nodes.push_back(mask_shape_node);

        // Scalar index for Gather (scalar indices -> scalar output)
        auto idx1 = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto gather_axis = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
        m_nodes.push_back(idx1);
        m_nodes.push_back(gather_axis);

        auto seq_len_s = std::make_shared<ov::opset11::Gather>(ids_shape, idx1, gather_axis);
        seq_len_s->set_friendly_name("model.seq_len");
        m_nodes.push_back(seq_len_s);

        auto total_seq_s = std::make_shared<ov::opset11::Gather>(mask_shape_node, idx1, gather_axis);
        total_seq_s->set_friendly_name("model.total_seq");
        m_nodes.push_back(total_seq_s);

        // offset = total_seq - seq_len (= past sequence length)
        auto offset = std::make_shared<ov::opset11::Subtract>(total_seq_s, seq_len_s);
        offset->set_friendly_name("model.causal_offset");
        m_nodes.push_back(offset);

        // kv_range = [0, 1, ..., total_seq-1]
        auto range_start = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto range_step = ov::opset11::Constant::create(ov::element::i64, ov::Shape{}, {1});
        m_nodes.push_back(range_start);
        m_nodes.push_back(range_step);

        auto kv_range = std::make_shared<ov::op::v4::Range>(range_start, total_seq_s, range_step, ov::element::i64);
        kv_range->set_friendly_name("model.kv_range");
        m_nodes.push_back(kv_range);

        // q_range = [0, 1, ..., seq_len-1], then add offset for absolute positions
        auto q_range = std::make_shared<ov::op::v4::Range>(range_start, seq_len_s, range_step, ov::element::i64);
        q_range->set_friendly_name("model.q_range");
        m_nodes.push_back(q_range);

        auto q_abs = std::make_shared<ov::opset11::Add>(q_range, offset);
        q_abs->set_friendly_name("model.q_abs_positions");
        m_nodes.push_back(q_abs);

        // Reshape for broadcasting comparison:
        // q_abs: [seq_len] -> [seq_len, 1]
        // kv_range: [total_seq] -> [1, total_seq]
        auto axis_last = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto axis_first = ov::opset11::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        m_nodes.push_back(axis_last);
        m_nodes.push_back(axis_first);

        auto q_col = std::make_shared<ov::opset11::Unsqueeze>(q_abs, axis_last);
        q_col->set_friendly_name("model.q_col");
        m_nodes.push_back(q_col);

        auto kv_row = std::make_shared<ov::opset11::Unsqueeze>(kv_range, axis_first);
        kv_row->set_friendly_name("model.kv_row");
        m_nodes.push_back(kv_row);

        // causal_bool[i,j] = (kv_range[j] <= q_abs[i]) -> [seq_len, total_seq]
        auto causal_bool = std::make_shared<ov::op::v1::LessEqual>(kv_row, q_col);
        causal_bool->set_friendly_name("model.causal_bool");
        m_nodes.push_back(causal_bool);

        // Convert to additive float mask: True -> 0.0, False -> -10000.0
        auto select_true = ov::opset11::Constant::create(prec, ov::Shape{}, {0.0f});
        auto select_false = ov::opset11::Constant::create(prec, ov::Shape{}, {-10000.0f});
        m_nodes.push_back(select_true);
        m_nodes.push_back(select_false);

        auto causal_float = std::make_shared<ov::op::v1::Select>(causal_bool, select_true, select_false);
        causal_float->set_friendly_name("model.causal_mask");
        m_nodes.push_back(causal_float);

        // [seq_len, total_seq] -> [1, 1, seq_len, total_seq]
        auto unsqueeze_axes = ov::opset11::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
        m_nodes.push_back(unsqueeze_axes);

        auto causal_4d = std::make_shared<ov::opset11::Unsqueeze>(causal_float, unsqueeze_axes);
        causal_4d->set_friendly_name("model.causal_mask_4d");
        m_nodes.push_back(causal_4d);

        // --- Combine: padding [batch, 1, 1, total_seq] + causal [1, 1, seq_len, total_seq] ---
        // Broadcasts to [batch, 1, seq_len, total_seq]
        auto combined = std::make_shared<ov::opset11::Add>(padding_4d, causal_4d);
        combined->set_friendly_name("model.mask_4d");
        m_nodes.push_back(combined);

        sdpa_mask = combined->output(0);
    }

    // ===== MIDDLE: Decoder Layers =====
    ov::Output<ov::Node> current = hidden_states;
    ov::SinkVector all_sinks;

    for (size_t layer = 0; layer < config.num_layers; ++layer) {
        auto layer_result = make_decoder_layer(current,
                                               config,
                                               layer,
                                               position_ids_output,
                                               input_ids->output(0),
                                               beam_idx->output(0),
                                               sdpa_mask);
        current = layer_result.output;

        for (auto& sink : layer_result.sinks) {
            all_sinks.push_back(std::dynamic_pointer_cast<ov::op::Sink>(sink));
        }
    }

    // ===== TAIL: Final Norm + LM Head =====
    auto final_norm = make_norm(current, config.hidden_size, "model.norm", config.norm_type, prec);
    auto logits =
        make_lm_head(final_norm, config.hidden_size, config.vocab_size, "lm_head", prec, config.lm_head_format);

    // ===== Build Model =====
    result(logits, "logits");

    // Create stateful model with sinks
    auto model = std::make_shared<ov::Model>(m_results, all_sinks, m_parameters, "llm_test_model");
    return model;
}

}  // namespace npuw
}  // namespace test
}  // namespace ov
