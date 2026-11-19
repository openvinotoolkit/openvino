// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/src/classes/gated_residual.hpp"

#include <algorithm>
#include <cstdint>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gated_residual.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

using Mode = ov::op::internal::GatedResidual::Mode;

struct QuantRange {
    int32_t qmin;
    int32_t qmax;
};

QuantRange quant_range_for(const ov::element::Type& t) {
    if (t == ov::element::u8) {
        return {0, 255};
    }
    if (t == ov::element::i8) {
        return {-128, 127};
    }
    if (t == ov::element::u4) {
        return {0, 15};
    }
    if (t == ov::element::i4) {
        return {-8, 7};
    }
    OPENVINO_THROW("Unsupported compressed weight element type: ", t);
}

void write_quantized(ov::Tensor& tensor, const std::vector<int32_t>& values) {
    const auto et = tensor.get_element_type();
    if (et == ov::element::u8) {
        auto* p = tensor.data<uint8_t>();
        for (size_t i = 0; i < values.size(); i++) {
            p[i] = static_cast<uint8_t>(values[i]);
        }
    } else if (et == ov::element::i8) {
        auto* p = tensor.data<int8_t>();
        for (size_t i = 0; i < values.size(); i++) {
            p[i] = static_cast<int8_t>(values[i]);
        }
    } else if (et == ov::element::u4) {
        auto iter = ov::element::iterator<ov::element::u4>(tensor.data());
        for (size_t i = 0; i < values.size(); i++, ++iter) {
            *iter = static_cast<uint8_t>(values[i]);
        }
    } else if (et == ov::element::i4) {
        auto iter = ov::element::iterator<ov::element::i4>(tensor.data());
        for (size_t i = 0; i < values.size(); i++, ++iter) {
            *iter = static_cast<int8_t>(values[i]);
        }
    } else {
        OPENVINO_THROW("Unsupported compressed weight element type: ", et);
    }
}

// Stores `value` at `index`, rounded to the tensor's float type, so the reference and the fused graph agree.
void write_float(ov::Tensor& tensor, size_t index, float value) {
    const auto et = tensor.get_element_type();
    if (et == ov::element::f32) {
        tensor.data<float>()[index] = value;
    } else if (et == ov::element::f16) {
        tensor.data<ov::float16>()[index] = ov::float16(value);
    } else if (et == ov::element::bf16) {
        tensor.data<ov::bfloat16>()[index] = ov::bfloat16(value);
    } else {
        OPENVINO_THROW("Unsupported decompression parameter type: ", et);
    }
}

// Decompression subgraph of a compressed projection weight [N, K] in the NNCF layout the fusion matches: Constant [N,
// G, K/G] -> Convert -> Subtract(Convert(zp [N, G, 1])) -> Multiply(scale [N, G, 1]) -> Reshape [N, K]. Scale and zero
// point are rounded so the reference and the fused graph agree.
ov::Output<ov::Node> make_compressed_weight(size_t n,
                                            size_t k,
                                            size_t groups,
                                            const ov::test::GatedResidualWeightCompression& spec,
                                            const ov::element::Type& data_type,
                                            std::mt19937& rng) {
    using namespace ov::op;
    const auto range = quant_range_for(spec.weights_precision);
    const size_t group_width = k / groups;
    std::uniform_int_distribution<int32_t> q_dist(range.qmin, range.qmax);
    std::uniform_real_distribution<float> scale_dist(0.001F, 0.01F);

    ov::Tensor weight(spec.weights_precision, ov::Shape{n, groups, group_width});
    std::vector<int32_t> q_values(n * k);
    for (auto& q : q_values) {
        q = q_dist(rng);
    }
    write_quantized(weight, q_values);

    ov::Tensor scale(spec.scale_precision, ov::Shape{n, groups, 1});
    for (size_t i = 0; i < n * groups; i++) {
        write_float(scale, i, scale_dist(rng));
    }

    auto weights_const = std::make_shared<v0::Constant>(weight);
    ov::Output<ov::Node> decompressed = std::make_shared<v0::Convert>(weights_const, data_type);
    if (spec.has_zero_points) {
        ov::Tensor zp(spec.weights_precision, ov::Shape{n, groups, 1});
        std::vector<int32_t> zp_values(n * groups);
        for (auto& z : zp_values) {
            z = q_dist(rng);
        }
        write_quantized(zp, zp_values);
        auto zp_const = std::make_shared<v0::Constant>(zp);
        auto zp_f = std::make_shared<v0::Convert>(zp_const, data_type);
        decompressed = std::make_shared<v1::Subtract>(decompressed, zp_f);
    }
    // NNCF stores the scale in its own float type and converts it to the compute type.
    ov::Output<ov::Node> scale_value = std::make_shared<v0::Constant>(scale);
    if (scale.get_element_type() != data_type) {
        scale_value = std::make_shared<v0::Convert>(scale_value, data_type);
    }
    decompressed = std::make_shared<v1::Multiply>(decompressed, scale_value);
    auto target = v0::Constant::create(ov::element::i64,
                                       ov::Shape{2},
                                       std::vector<int64_t>{static_cast<int64_t>(n), static_cast<int64_t>(k)});
    return std::make_shared<v1::Reshape>(decompressed, target, false);
}

// The projection weight [N, K] (transpose_b): a plain constant, or the decompression subgraph of a compressed one.
ov::Output<ov::Node> make_weight(size_t n,
                                 size_t k,
                                 const ov::element::Type& data_type,
                                 const ov::test::GatedResidualWeightCompression& compression,
                                 std::mt19937& rng) {
    if (compression.compressed) {
        const size_t groups = compression.group_size == 0 ? 1 : k / compression.group_size;
        return make_compressed_weight(n, k, groups, compression, data_type, rng);
    }
    ov::Tensor t(data_type, ov::Shape{n, k});
    std::uniform_real_distribution<float> dist(-0.1F, 0.1F);
    std::mt19937 local(static_cast<uint32_t>(n * 131 + k));
    for (size_t i = 0; i < n * k; i++) {
        write_float(t, i, dist(local));
    }
    return std::make_shared<ov::op::v0::Constant>(t);
}

std::shared_ptr<ov::op::v0::Constant> scalar(const ov::element::Type& et, float v) {
    return ov::op::v0::Constant::create(et, ov::Shape{}, {v});
}

// Reshape keeping `leading_rank` dims (0 = copy), then `tail`.
std::shared_ptr<ov::op::v1::Reshape> reshape_keep_leading(const ov::Output<ov::Node>& src,
                                                          size_t leading_rank,
                                                          const std::vector<int64_t>& tail) {
    std::vector<int64_t> target(leading_rank, 0);
    target.insert(target.end(), tail.begin(), tail.end());
    auto shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{target.size()}, target);
    return std::make_shared<ov::op::v1::Reshape>(src, shape, true);
}

// The checkpoint's decomposed sub-graphs, run as the reference. The device under test runs the same graph after the
// common fusion pass.
std::shared_ptr<ov::Model> make_reference_model(Mode mode,
                                                const ov::PartialShape& leading_shape,
                                                size_t hc_count,
                                                size_t hidden_size,
                                                size_t lowrank,
                                                const ov::element::Type& precision,
                                                bool produce_write_gate,
                                                bool norm_gain_plus_one,
                                                float epsilon,
                                                const ov::test::GatedResidualWeightCompression& compression) {
    using namespace ov::op;
    const size_t cd = hc_count * hidden_size;
    const size_t leading_rank = leading_shape.size();
    const auto rows_shape = [&](size_t last) {
        auto ps = leading_shape;
        ps.push_back(ov::Dimension(static_cast<int64_t>(last)));
        return ps;
    };
    const auto c = static_cast<int64_t>(hc_count);
    const auto d = static_cast<int64_t>(hidden_size);
    const auto cd_i = static_cast<int64_t>(cd);

    std::mt19937 rng(12345);
    if (mode == Mode::WRITE) {
        auto residual = std::make_shared<v0::Parameter>(precision, rows_shape(cd));
        residual->set_friendly_name("residual");
        auto y = std::make_shared<v0::Parameter>(precision, rows_shape(hidden_size));
        y->set_friendly_name("sublayer_y");
        auto gate = std::make_shared<v0::Parameter>(precision, rows_shape(hc_count));
        gate->set_friendly_name("write_gate");

        auto s3 = std::make_shared<v0::Unsqueeze>(gate, v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
        auto y3 = std::make_shared<v0::Unsqueeze>(y, v0::Constant::create(ov::element::i64, ov::Shape{1}, {-2}));
        // The exporter's form: the Multiply is flattened and added to the flat residual.
        auto scaled_flat = reshape_keep_leading(std::make_shared<v1::Multiply>(s3, y3), leading_rank, {cd_i});
        auto out = std::make_shared<v1::Add>(residual, scaled_flat);
        return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<v0::Result>(out)},
                                           ov::ParameterVector{residual, y, gate});
    }

    auto residual = std::make_shared<v0::Parameter>(precision, rows_shape(cd));
    residual->set_friendly_name("residual");
    auto gain = std::make_shared<v0::Parameter>(precision, ov::Shape{cd});
    gain->set_friendly_name("norm_weight");

    // Per-branch RMSNorm, decomposed. The gain already holds the `1 + w` fold when norm_gain_plus_one is set.
    auto r3 = reshape_keep_leading(residual, leading_rank, {c, d});
    auto sq = std::make_shared<v1::Power>(r3, scalar(precision, 2.0F));
    auto mean = std::make_shared<v1::ReduceMean>(sq, v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);
    auto biased = std::make_shared<v1::Add>(mean, scalar(precision, epsilon));
    auto rms = std::make_shared<v0::Sqrt>(biased);
    auto inv_rms = std::make_shared<v1::Divide>(scalar(precision, 1.0F), rms);
    auto normalized = std::make_shared<v1::Multiply>(r3, inv_rms);
    auto normalized_flat = reshape_keep_leading(normalized, leading_rank, {cd_i});
    ov::Output<ov::Node> gain_value = gain;
    if (norm_gain_plus_one) {
        gain_value = std::make_shared<v1::Add>(gain, scalar(precision, 1.0F));
    }
    auto rbar = std::make_shared<v1::Multiply>(normalized_flat, gain_value);

    // Down projection with the 1/C scale, SiLU, up projection, sigmoid gate, weighted branch mix and its 1/C.
    auto low = std::make_shared<v0::MatMul>(rbar, make_weight(lowrank, cd, precision, compression, rng), false, true);
    auto low_scaled = std::make_shared<v1::Multiply>(low, scalar(precision, 1.0F / static_cast<float>(hc_count)));
    auto act = std::make_shared<v4::Swish>(low_scaled);
    auto up = std::make_shared<v0::MatMul>(act, make_weight(cd, lowrank, precision, compression, rng), false, true);
    auto read_gate = std::make_shared<v0::Sigmoid>(up);
    auto weighted = std::make_shared<v1::Multiply>(read_gate, rbar);
    auto weighted3 = reshape_keep_leading(weighted, leading_rank, {c, d});
    auto reduced =
        std::make_shared<v1::ReduceSum>(weighted3, v0::Constant::create(ov::element::i64, ov::Shape{1}, {-2}), false);
    auto mixed = std::make_shared<v1::Multiply>(reduced, scalar(precision, 1.0F / static_cast<float>(hc_count)));

    ov::ResultVector results{std::make_shared<v0::Result>(mixed)};
    ov::ParameterVector params{residual, gain};
    if (produce_write_gate) {
        // Write gate: 2 * sigmoid(W_inject * Rbar / C), one scalar per branch.
        auto inject =
            std::make_shared<v0::MatMul>(rbar, make_weight(hc_count, cd, precision, compression, rng), false, true);
        auto inject_scaled =
            std::make_shared<v1::Multiply>(inject, scalar(precision, 1.0F / static_cast<float>(hc_count)));
        auto gate_out =
            std::make_shared<v1::Multiply>(std::make_shared<v0::Sigmoid>(inject_scaled), scalar(precision, 2.0F));
        results.push_back(std::make_shared<v0::Result>(gate_out));
    }
    return std::make_shared<ov::Model>(results, params);
}

}  // namespace

namespace ov::test {

std::string GatedResidualLayerCPUTest::getTestCaseName(const testing::TestParamInfo<GatedResidualLayerParams>& obj) {
    const auto& [mode, shape, produce_write_gate, norm_gain_plus_one, input_shape, additional_config, device] =
        obj.param;
    const auto& [hc_count, hidden_size, lowrank, precision, compression] = shape;
    std::ostringstream result;
    result << "Mode=" << static_cast<int>(mode);
    result << "_C=" << hc_count << "_D=" << hidden_size << "_L=" << lowrank;
    result << "_WG=" << produce_write_gate;
    result << "_GainP1=" << norm_gain_plus_one;
    result << "_IS=" << ov::test::utils::partialShape2str({input_shape.first}) << "_TS=";
    for (const auto& target : input_shape.second) {
        result << ov::test::utils::vec2str(target);
    }
    result << "_Prc=" << precision;
    if (compression.compressed) {
        result << "_Wcompressed=" << compression.weights_precision;
        result << "_ZP=" << compression.has_zero_points;
        result << "_GS=" << compression.group_size;
        result << "_SP=" << compression.scale_precision;
    }
    for (const auto& [name, value] : additional_config) {
        result << "_" << name << "=" << value.as<std::string>();
    }
    result << "_Target=" << device;
    return result.str();
}

void GatedResidualLayerCPUTest::SetUp() {
    const auto& [mode, shape, produce_write_gate, norm_gain_plus_one, input_shape, additional_config, device] =
        GetParam();
    targetDevice = device;
    configuration.insert(additional_config.begin(), additional_config.end());
    m_mode = mode;
    m_hc_count = shape.hc_count;
    m_hidden_size = shape.hidden_size;
    m_lowrank = shape.lowrank;
    m_produce_write_gate = produce_write_gate;
    m_norm_gain_plus_one = norm_gain_plus_one;
    m_precision = shape.precision;
    m_compression = shape.compression;

    // Every case uses dynamic leading dims, so the node takes the dynamic-shape path (shape inference + prepareParams
    // per inference); `input_shape.second` gives the static targets.
    const auto& leading_ps = input_shape.first;
    const auto& leading_targets = input_shape.second;
    OPENVINO_ASSERT(leading_ps.is_dynamic() && !leading_targets.empty(),
                    "GatedResidual tests must use a dynamic input shape with at least one target shape");

    // The reference is the decomposed sub-graph. The plugin fuses it into one GatedResidual node (WRITE runs as
    // Eltwise).
    functionRefs = make_reference_model(mode,
                                        leading_ps,
                                        m_hc_count,
                                        m_hidden_size,
                                        m_lowrank,
                                        m_precision,
                                        produce_write_gate,
                                        norm_gain_plus_one,
                                        m_epsilon,
                                        m_compression);
    function = functionRefs->clone();

    std::vector<InputShape> input_shapes;
    input_shapes.reserve(function->get_parameters().size());
    for (const auto& param : function->get_parameters()) {
        const auto& ps = param->get_partial_shape();
        std::vector<ov::Shape> targets;
        if (ps.is_static()) {
            targets.assign(leading_targets.size(), ps.to_shape());
        } else {
            const auto last = static_cast<size_t>(ps[ps.size() - 1].get_length());
            for (auto target : leading_targets) {
                target.push_back(last);
                targets.push_back(std::move(target));
            }
        }
        input_shapes.push_back(InputShape{ps, targets});
    }
    init_input_shapes(input_shapes);
}

void GatedResidualLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    const auto& params = function->get_parameters();
    int seed = 1;

    const auto fill = [&](const std::shared_ptr<ov::op::v0::Parameter>& param,
                          const ov::Shape& shape,
                          float lo,
                          float hi) {
        inputs[param] =
            ov::test::utils::create_and_fill_tensor_real_distribution(param->get_element_type(), shape, lo, hi, seed++);
    };

    if (m_mode == Mode::WRITE) {
        // residual, sublayer_y, write_gate
        for (size_t i = 0; i < 3; ++i) {
            fill(params[i], targetInputStaticShapes[i], -1.0F, 1.0F);
        }
        return;
    }

    // Residual and norm weight are the only model inputs; the projection weights are constants of the graph.
    fill(params[0], targetInputStaticShapes[0], -1.0F, 1.0F);  // residual
    fill(params[1], targetInputStaticShapes[1], -0.1F, 0.1F);  // norm weight
}

void GatedResidualLayerCPUTest::compare(const std::vector<ov::Tensor>& expected,
                                        const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    // The error follows the coarser of the model and the inference precision (the f32 reference runs at the model's).
    const auto inference = configuration.find(ov::hint::inference_precision.name());
    const auto inference_precision =
        inference == configuration.end() ? ov::element::f32 : inference->second.as<ov::element::Type>();
    if (m_precision == ov::element::bf16 || inference_precision == ov::element::bf16) {
        abs_threshold = 1e-2F;
        rel_threshold = 2e-2F;
    } else if (m_precision == ov::element::f16 || inference_precision == ov::element::f16) {
        abs_threshold = 2e-3F;
        rel_threshold = 1e-2F;
    } else {
        abs_threshold = 1e-4F;
        rel_threshold = 1e-4F;
    }
    if (m_compression.compressed) {
        // Decompression rounding dominates the error for quantized weights, more so for 4-bit types; dynamic
        // quantization (on by default) adds int8 activation error.
        const auto dq_group_size = configuration.find(ov::hint::dynamic_quantization_group_size.name());
        const bool dyn_quant = dq_group_size == configuration.end() || dq_group_size->second.as<uint64_t>() != 0;
        const bool narrow = dyn_quant || m_compression.weights_precision == ov::element::u4 ||
                            m_compression.weights_precision == ov::element::i4;
        abs_threshold = std::max(abs_threshold, narrow ? 0.2 : 0.1);
        rel_threshold = std::max(rel_threshold, narrow ? 0.2 : 0.1);
    }
    for (size_t i = 0; i < expected.size(); i++) {
        ov::test::utils::compare(expected[i], actual[i], abs_threshold, rel_threshold);
    }
}

TEST_P(GatedResidualLayerCPUTest, Inference) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    const auto runtime_model = compiledModel.get_runtime_model();
    if (std::get<0>(GetParam()) == Mode::WRITE) {
        // The Multiply and the residual Add are one Eltwise after GatedResidualWriteSinkReshape; the node does not run
        // WRITE.
        CheckNumberOfNodesWithType(runtime_model, {"GatedResidual"}, 0);
        CheckNumberOfNodesWithType(runtime_model, {"Eltwise"}, 1);
    } else {
        CheckNumberOfNodesWithType(runtime_model, {"GatedResidual"}, 1);
    }
}

}  // namespace ov::test
