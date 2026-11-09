// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/gated_residual.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_iterator.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gated_residual.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "openvino/runtime/tensor.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

namespace {

using Mode = ov::op::internal::GatedResidual::Mode;
using ReadInputIdx = ov::op::internal::GatedResidual::ReadInputIdx;
using WriteInputIdx = ov::op::internal::GatedResidual::WriteInputIdx;

float sigmoid(float x) {
    return 1.0F / (1.0F + std::exp(-x));
}

float silu(float x) {
    return x * sigmoid(x);
}

// Upconverts an f32/bf16/f16-backed tensor into a float vector.
std::vector<float> tensor_to_f32(const ov::Tensor& t) {
    std::vector<float> out(t.get_size());
    const auto et = t.get_element_type();
    if (et == ov::element::f32) {
        const auto* p = t.data<const float>();
        std::copy(p, p + out.size(), out.begin());
    } else if (et == ov::element::bf16) {
        const auto* p = t.data<const ov::bfloat16>();
        std::transform(p, p + out.size(), out.begin(), [](ov::bfloat16 v) {
            return static_cast<float>(v);
        });
    } else if (et == ov::element::f16) {
        const auto* p = t.data<const ov::float16>();
        std::transform(p, p + out.size(), out.begin(), [](ov::float16 v) {
            return static_cast<float>(v);
        });
    } else {
        OPENVINO_THROW("Unsupported element type for tensor_to_f32: ", et);
    }
    return out;
}

// Downcasts a float vector into a f32/bf16/f16-backed tensor's native storage.
void store_as(const std::vector<float>& src, ov::Tensor& dst) {
    const auto et = dst.get_element_type();
    if (et == ov::element::f32) {
        std::copy(src.begin(), src.end(), dst.data<float>());
    } else if (et == ov::element::bf16) {
        std::transform(src.begin(), src.end(), dst.data<ov::bfloat16>(), [](float v) {
            return static_cast<ov::bfloat16>(v);
        });
    } else if (et == ov::element::f16) {
        std::transform(src.begin(), src.end(), dst.data<ov::float16>(), [](float v) {
            return static_cast<ov::float16>(v);
        });
    } else {
        OPENVINO_THROW("Unsupported element type for store_as: ", et);
    }
}

// dst[rows,N] = src[rows,K] @ weight[N,K]^T, matching the CPU kernel's projection convention.
void matmul_nt(const float* src, size_t rows, size_t k, const float* weight, size_t n, float* dst) {
    for (size_t row = 0; row < rows; row++) {
        for (size_t nn = 0; nn < n; nn++) {
            double acc = 0.0;
            const float* w_row = weight + nn * k;
            const float* s_row = src + row * k;
            for (size_t kk = 0; kk < k; kk++) {
                acc += static_cast<double>(s_row[kk]) * w_row[kk];
            }
            dst[row * n + nn] = static_cast<float>(acc);
        }
    }
}

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

void write_quantized_weight(ov::Tensor& tensor, const std::vector<int32_t>& values) {
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

struct QuantizedWeight {
    ov::Tensor weight;
    ov::Tensor scale;                // [G, N], floating point
    ov::Tensor zero_point;           // [G, N], the weights' type
    std::vector<float> dequantized;  // row-major [N, K]
};

void write_scale(ov::Tensor& tensor, size_t index, float value) {
    const auto et = tensor.get_element_type();
    if (et == ov::element::f32) {
        tensor.data<float>()[index] = value;
    } else if (et == ov::element::f16) {
        tensor.data<ov::float16>()[index] = ov::float16(value);
    } else if (et == ov::element::bf16) {
        tensor.data<ov::bfloat16>()[index] = ov::bfloat16(value);
    } else {
        OPENVINO_THROW("Unsupported scale element type: ", et);
    }
}

float read_scale(const ov::Tensor& tensor, size_t index) {
    const auto et = tensor.get_element_type();
    if (et == ov::element::f32) {
        return tensor.data<const float>()[index];
    }
    if (et == ov::element::f16) {
        return static_cast<float>(tensor.data<const ov::float16>()[index]);
    }
    if (et == ov::element::bf16) {
        return static_cast<float>(tensor.data<const ov::bfloat16>()[index]);
    }
    OPENVINO_THROW("Unsupported scale element type: ", et);
}

// Builds a [N,K] quantized weight plus its [G,N] scale / zero-point tensors (G = K / group size, N contiguous) and the
// corresponding dequantized float matrix (`dequantized = (quantized - zero_point) * scale`). The reference uses the
// scale as rounded to its storage type, i.e. exactly what the node reads.
QuantizedWeight make_quantized_weight(size_t n,
                                      size_t k,
                                      const ov::test::GatedResidualWeightCompression& spec,
                                      std::mt19937& rng) {
    const auto range = quant_range_for(spec.weights_precision);
    const size_t group_width = spec.group_size == 0 ? k : spec.group_size;
    const size_t groups = k / group_width;

    QuantizedWeight out;
    out.weight = ov::Tensor(spec.weights_precision, ov::Shape{n, k});
    out.scale = ov::Tensor(spec.scale_precision, ov::Shape{groups, n});
    if (spec.has_zero_points) {
        out.zero_point = ov::Tensor(spec.weights_precision, ov::Shape{groups, n});
    }
    out.dequantized.resize(n * k);

    std::uniform_int_distribution<int32_t> q_dist(range.qmin, range.qmax);
    std::uniform_real_distribution<float> scale_dist(0.001F, 0.01F);

    std::vector<int32_t> q_values(n * k);
    std::vector<int32_t> zp_values(spec.has_zero_points ? groups * n : 0);

    for (size_t row = 0; row < n; row++) {
        for (size_t g = 0; g < groups; g++) {
            write_scale(out.scale, g * n + row, scale_dist(rng));
            const float scale = read_scale(out.scale, g * n + row);
            const int32_t zp = spec.has_zero_points ? q_dist(rng) : 0;
            if (spec.has_zero_points) {
                zp_values[g * n + row] = zp;
            }
            for (size_t kk = 0; kk < group_width; kk++) {
                const size_t idx = g * group_width + kk;
                const int32_t q = q_dist(rng);
                q_values[row * k + idx] = q;
                out.dequantized[row * k + idx] = static_cast<float>(q - zp) * scale;
            }
        }
    }
    write_quantized_weight(out.weight, q_values);
    if (spec.has_zero_points) {
        write_quantized_weight(out.zero_point, zp_values);
    }
    return out;
}

}  // namespace

namespace ov::test {

std::string GatedResidualLayerCPUTest::getTestCaseName(const testing::TestParamInfo<GatedResidualLayerParams>& obj) {
    const auto& [mode, shape, produce_write_gate, norm_gain_plus_one, input_shape, device] = obj.param;
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
    result << "_Target=" << device;
    return result.str();
}

void GatedResidualLayerCPUTest::SetUp() {
    const auto& [mode, shape, produce_write_gate, norm_gain_plus_one, input_shape, device] = GetParam();
    // Plain variables, not structured bindings: the lambdas below capture them and a structured binding cannot be
    // captured before C++20 on older compilers.
    const size_t hc_count = shape.hc_count;
    const size_t hidden_size = shape.hidden_size;
    const size_t lowrank = shape.lowrank;
    const ov::element::Type precision = shape.precision;
    const GatedResidualWeightCompression& compression = shape.compression;
    targetDevice = device;
    m_mode = mode;
    m_hc_count = hc_count;
    m_hidden_size = hidden_size;
    m_lowrank = lowrank;
    m_produce_write_gate = produce_write_gate;
    m_norm_gain_plus_one = norm_gain_plus_one;
    m_precision = precision;
    m_compression = compression;

    const size_t cd = hc_count * hidden_size;
    // Every case runs with dynamic leading [B, T] dims on the data-path inputs (residual, sublayer_y,
    // write_gate), so the node always takes the dynamic-shape path (shape inference + prepareParams per
    // inference); only weights, scales and norm_weight keep static shapes. The static targets of each
    // inference come from `input_shape.second`.
    const auto& leading_ps = input_shape.first;
    const auto& leading_targets = input_shape.second;
    OPENVINO_ASSERT(leading_ps.is_dynamic() && !leading_targets.empty(),
                    "GatedResidual tests must use a dynamic input shape with at least one target shape");
    const auto rows_shape = [&](size_t last) {
        auto ps = leading_ps;
        ps.push_back(ov::Dimension(static_cast<int64_t>(last)));
        return ps;
    };
    using ov::op::v0::Constant;
    using ov::op::v0::Parameter;
    ov::ParameterVector params;
    std::shared_ptr<ov::Node> gr;

    if (mode == Mode::WRITE) {
        auto residual = std::make_shared<Parameter>(precision, rows_shape(cd));
        auto y = std::make_shared<Parameter>(precision, rows_shape(hidden_size));
        auto gate = std::make_shared<Parameter>(precision, rows_shape(hc_count));
        params = {residual, y, gate};
        gr = std::make_shared<ov::op::internal::GatedResidual>(ov::OutputVector{residual, y, gate},
                                                               mode,
                                                               hc_count,
                                                               hidden_size,
                                                               lowrank,
                                                               m_epsilon,
                                                               norm_gain_plus_one,
                                                               false);
    } else {
        auto residual = std::make_shared<Parameter>(precision, rows_shape(cd));
        auto norm_weight = std::make_shared<Parameter>(precision, ov::Shape{cd});
        ov::OutputVector args{residual, norm_weight};
        params = {residual, norm_weight};

        // Compressed weights are baked in as frozen Constants holding real quantized data (matching
        // how deployed quantized models represent them -- see the production Qwen IR's
        // Const(i8)->Convert->Multiply decompression subgraph), rather than runtime Parameters.
        // disable_constant_folding + enable_keep_const_precision protect their native i4/u4/i8/u8
        // element type from the CPU plugin's blanket ConvertPrecision pass, which otherwise silently
        // promotes i4->i8/u4->u8 for any Constant not marked this way. The matching scale/zero-point
        // tensors are graph Constants too, with the exact values generated here so the dequantization math
        // matches what the CPU node will compute.
        std::mt19937 rng(12345);
        ov::Output<ov::Node> w_down;
        ov::Output<ov::Node> w_up;
        ov::Output<ov::Node> w_inject;
        // Decompression scales/zero-points are graph Constants too (they are the decompression constants of the
        // constant weights), so the executor prepares their oneDNN views once.
        ov::Tensor down_scale_tensor;
        ov::Tensor down_zp_tensor;
        ov::Tensor up_scale_tensor;
        ov::Tensor up_zp_tensor;
        ov::Tensor inject_scale_tensor;
        ov::Tensor inject_zp_tensor;

        // Projection weights are graph Constants in every case, as in deployed models (they are the fused GEMM
        // weights), so the executor may prepack them once. Their f32 values are kept for the reference.
        int weight_seed = 100;
        const auto make_weight = [&](const ov::Shape& shape, std::vector<float>& f32_values) {
            const auto tensor =
                ov::test::utils::create_and_fill_tensor_real_distribution(precision, shape, -0.1F, 0.1F, weight_seed++);
            f32_values = tensor_to_f32(tensor);
            return std::make_shared<Constant>(tensor);
        };

        if (!compression.compressed) {
            w_down = make_weight(ov::Shape{lowrank, cd}, m_down_weight_f32);
            w_up = make_weight(ov::Shape{cd, lowrank}, m_up_weight_f32);
        } else {
            auto qw_down = make_quantized_weight(lowrank, cd, compression, rng);
            auto w_down_const = std::make_shared<Constant>(qw_down.weight);
            ov::pass::disable_constant_folding(w_down_const);
            ov::enable_keep_const_precision(w_down_const);
            w_down = w_down_const;
            m_down_weight_f32 = std::move(qw_down.dequantized);
            down_scale_tensor = qw_down.scale;
            if (compression.has_zero_points) {
                down_zp_tensor = qw_down.zero_point;
            }

            auto qw_up = make_quantized_weight(cd, lowrank, compression, rng);
            auto w_up_const = std::make_shared<Constant>(qw_up.weight);
            ov::pass::disable_constant_folding(w_up_const);
            ov::enable_keep_const_precision(w_up_const);
            w_up = w_up_const;
            m_up_weight_f32 = std::move(qw_up.dequantized);
            up_scale_tensor = qw_up.scale;
            if (compression.has_zero_points) {
                up_zp_tensor = qw_up.zero_point;
            }
        }
        args.push_back(w_down);
        args.push_back(w_up);

        // Compressed form: scales (then zero points, when present) of w_down and w_up follow the weights, and the
        // write-gate inputs (w_inject, its scale and zero point) come last; see GatedResidual::ReadLayout. Scale
        // and zero-point Constants are graph constants too; zero points have the weights' type, which must survive
        // the CPU plugin's precision conversion like the weights do.
        const auto add_decompression_input = [&](const ov::Tensor& tensor) {
            auto node = std::make_shared<Constant>(tensor);
            if (tensor.get_element_type() == compression.weights_precision) {
                ov::pass::disable_constant_folding(node);
                ov::enable_keep_const_precision(node);
            }
            args.push_back(node);
        };
        if (compression.compressed) {
            add_decompression_input(down_scale_tensor);
            add_decompression_input(up_scale_tensor);
            if (compression.has_zero_points) {
                add_decompression_input(down_zp_tensor);
                add_decompression_input(up_zp_tensor);
            }
        }

        if (produce_write_gate) {
            if (!compression.compressed) {
                w_inject = make_weight(ov::Shape{hc_count, cd}, m_inject_weight_f32);
            } else {
                auto qw_inject = make_quantized_weight(hc_count, cd, compression, rng);
                auto w_inject_const = std::make_shared<Constant>(qw_inject.weight);
                ov::pass::disable_constant_folding(w_inject_const);
                ov::enable_keep_const_precision(w_inject_const);
                w_inject = w_inject_const;
                m_inject_weight_f32 = std::move(qw_inject.dequantized);
                inject_scale_tensor = qw_inject.scale;
                if (compression.has_zero_points) {
                    inject_zp_tensor = qw_inject.zero_point;
                }
            }
            args.push_back(w_inject);
            if (compression.compressed) {
                add_decompression_input(inject_scale_tensor);
                if (compression.has_zero_points) {
                    add_decompression_input(inject_zp_tensor);
                }
            }
        }

        gr = std::make_shared<ov::op::internal::GatedResidual>(args,
                                                               mode,
                                                               hc_count,
                                                               hidden_size,
                                                               lowrank,
                                                               m_epsilon,
                                                               norm_gain_plus_one,
                                                               produce_write_gate);
    }

    std::vector<InputShape> input_shapes;
    input_shapes.reserve(params.size());
    for (const auto& param : params) {
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

    ov::ResultVector results;
    for (const auto& out : gr->outputs()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(out));
    }
    function = std::make_shared<ov::Model>(results, params);
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
        fill(params[WriteInputIdx::RESIDUAL], targetInputStaticShapes[WriteInputIdx::RESIDUAL], -1.0F, 1.0F);
        fill(params[WriteInputIdx::SUBLAYER_Y], targetInputStaticShapes[WriteInputIdx::SUBLAYER_Y], -1.0F, 1.0F);
        fill(params[WriteInputIdx::WRITE_GATE], targetInputStaticShapes[WriteInputIdx::WRITE_GATE], -1.0F, 1.0F);
        return;
    }

    // RESIDUAL/NORM_WEIGHT are always the first two model parameters -- compressed weight Constants
    // (when present) aren't model inputs, see SetUp().
    fill(params[ReadInputIdx::RESIDUAL], targetInputStaticShapes[ReadInputIdx::RESIDUAL], -1.0F, 1.0F);
    fill(params[ReadInputIdx::NORM_WEIGHT], targetInputStaticShapes[ReadInputIdx::NORM_WEIGHT], -0.1F, 0.1F);

    if (!m_compression.compressed) {
        // The projection weights are graph Constants (see SetUp()): residual and norm_weight are the only inputs.
        return;
    }

    // Compressed weights and their decompression scales/zero-points are graph Constants (see SetUp()):
    // residual and norm_weight are the only inputs.
}

std::vector<ov::Tensor> GatedResidualLayerCPUTest::calculate_refs() {
    const auto& params = function->get_parameters();
    const size_t c = m_hc_count;
    const size_t d = m_hidden_size;
    const size_t l = m_lowrank;
    const size_t cd = c * d;

    if (m_mode == Mode::WRITE) {
        const auto residual = tensor_to_f32(inputs.at(params[WriteInputIdx::RESIDUAL]));
        const auto y = tensor_to_f32(inputs.at(params[WriteInputIdx::SUBLAYER_Y]));
        const auto gate = tensor_to_f32(inputs.at(params[WriteInputIdx::WRITE_GATE]));
        const size_t rows = residual.size() / cd;

        std::vector<float> out(rows * cd);
        for (size_t row = 0; row < rows; row++) {
            for (size_t cc = 0; cc < c; cc++) {
                const float g = gate[row * c + cc];
                for (size_t dd = 0; dd < d; dd++) {
                    const size_t idx = row * cd + cc * d + dd;
                    out[idx] = residual[idx] + g * y[row * d + dd];
                }
            }
        }

        ov::Tensor out_tensor(m_precision, inputs.at(params[WriteInputIdx::RESIDUAL]).get_shape());
        store_as(out, out_tensor);
        return {out_tensor};
    }

    const auto residual = tensor_to_f32(inputs.at(params[ReadInputIdx::RESIDUAL]));
    const auto norm_weight = tensor_to_f32(inputs.at(params[ReadInputIdx::NORM_WEIGHT]));
    const size_t rows = residual.size() / cd;

    std::vector<float> rbar(rows * cd);
    for (size_t row = 0; row < rows; row++) {
        for (size_t cc = 0; cc < c; cc++) {
            const float* branch_in = residual.data() + row * cd + cc * d;
            double sumsq = 0.0;
            for (size_t dd = 0; dd < d; dd++) {
                sumsq += static_cast<double>(branch_in[dd]) * branch_in[dd];
            }
            const float inv_rms = 1.0F / std::sqrt(static_cast<float>(sumsq / d) + m_epsilon);
            float* branch_out = rbar.data() + row * cd + cc * d;
            for (size_t dd = 0; dd < d; dd++) {
                float gain = norm_weight[cc * d + dd];
                if (m_norm_gain_plus_one) {
                    gain += 1.0F;
                }
                branch_out[dd] = branch_in[dd] * inv_rms * gain;
            }
        }
    }

    const float inv_c = 1.0F / static_cast<float>(c);

    std::vector<float> t(rows * l);
    matmul_nt(rbar.data(), rows, cd, m_down_weight_f32.data(), l, t.data());
    for (auto& v : t) {
        v = silu(v * inv_c);
    }

    std::vector<float> logit(rows * cd);
    matmul_nt(t.data(), rows, l, m_up_weight_f32.data(), cd, logit.data());

    std::vector<float> mixed(rows * d);
    for (size_t row = 0; row < rows; row++) {
        for (size_t dd = 0; dd < d; dd++) {
            double acc = 0.0;
            for (size_t cc = 0; cc < c; cc++) {
                const size_t idx = cc * d + dd;
                acc += static_cast<double>(sigmoid(logit[row * cd + idx])) * rbar[row * cd + idx];
            }
            mixed[row * d + dd] = static_cast<float>(acc * inv_c);
        }
    }

    std::vector<ov::Tensor> result;
    // Output shapes follow the (possibly dynamic) residual input: [B, T, D] and [B, T, C].
    const auto residual_shape = inputs.at(params[ReadInputIdx::RESIDUAL]).get_shape();
    ov::Shape mixed_shape = residual_shape;
    mixed_shape.back() = d;
    ov::Tensor mixed_tensor(m_precision, mixed_shape);
    store_as(mixed, mixed_tensor);
    result.push_back(mixed_tensor);

    if (m_produce_write_gate) {
        std::vector<float> wg_logit(rows * c);
        matmul_nt(rbar.data(), rows, cd, m_inject_weight_f32.data(), c, wg_logit.data());
        std::vector<float> write_gate(rows * c);
        for (size_t i = 0; i < write_gate.size(); i++) {
            write_gate[i] = 2.0F * sigmoid(wg_logit[i] * inv_c);
        }
        ov::Shape wg_shape = residual_shape;
        wg_shape.back() = c;
        ov::Tensor wg_tensor(m_precision, wg_shape);
        store_as(write_gate, wg_tensor);
        result.push_back(wg_tensor);
    }
    return result;
}

void GatedResidualLayerCPUTest::compare(const std::vector<ov::Tensor>& expected,
                                        const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    if (m_precision == ov::element::bf16) {
        abs_threshold = 1e-2F;
        rel_threshold = 2e-2F;
    } else if (m_precision == ov::element::f16) {
        abs_threshold = 2e-3F;
        rel_threshold = 1e-2F;
    } else {
        abs_threshold = 1e-4F;
        rel_threshold = 1e-4F;
    }
    if (m_compression.compressed) {
        // Decompression rounding dominates the error for quantized weights; loosen thresholds
        // accordingly, more so for 4-bit types (mirrors GroupedMatMulCompressedLayerTest).
        const bool narrow =
            m_compression.weights_precision == ov::element::u4 || m_compression.weights_precision == ov::element::i4;
        abs_threshold = std::max(abs_threshold, narrow ? 0.2 : 0.1);
        rel_threshold = std::max(rel_threshold, narrow ? 0.2 : 0.1);
    }
    for (size_t i = 0; i < expected.size(); i++) {
        ov::test::utils::compare(expected[i], actual[i], abs_threshold, rel_threshold);
    }
}

TEST_P(GatedResidualLayerCPUTest, Inference) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    // bf16 / f16 cases need the native instructions (the executor has no upconvert path for them).
    const auto precision = std::get<1>(GetParam()).precision;
    if (precision == ov::element::bf16 && !ov::with_cpu_x86_bfloat16()) {
        GTEST_SKIP() << "bf16 requires avx512_core_bf16";
    }
    if (precision == ov::element::f16 && !ov::with_cpu_x86_avx512_core_fp16()) {
        GTEST_SKIP() << "f16 requires avx512_core_fp16";
    }
    run();
    const auto runtime_model = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(runtime_model, {"GatedResidual"}, 1);
}

}  // namespace ov::test
