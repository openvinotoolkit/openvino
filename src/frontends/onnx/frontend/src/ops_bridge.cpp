// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ops_bridge.hpp"

#include <functional>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>

#include "core/attribute.hpp"
#include "op/abs.hpp"
#include "op/acos.hpp"
#include "op/acosh.hpp"
#include "op/adaptive_avg_pooling2d.hpp"
#include "op/add.hpp"
#include "op/affine.hpp"
#include "op/and.hpp"
#include "op/argmax.hpp"
#include "op/argmin.hpp"
#include "op/asin.hpp"
#include "op/asinh.hpp"
#include "op/atan.hpp"
#include "op/atanh.hpp"
#include "op/aten.hpp"
#include "op/average_pool.hpp"
#include "op/batch_norm.hpp"
#include "op/bitshift.hpp"
#include "op/bitwise_and.hpp"
#include "op/bitwise_not.hpp"
#include "op/bitwise_or.hpp"
#include "op/bitwise_xor.hpp"
#include "op/blackmanwindow.hpp"
#include "op/cast.hpp"
#include "op/cast_like.hpp"
#include "op/ceil.hpp"
#include "op/celu.hpp"
#include "op/clip.hpp"
#include "op/com.microsoft/attention.hpp"
#include "op/com.microsoft/bias_gelu.hpp"
#include "op/com.microsoft/embed_layer_normalization.hpp"
#include "op/com.microsoft/fused_conv.hpp"
#include "op/com.microsoft/fusedgemm.hpp"
#include "op/com.microsoft/pad.hpp"
#include "op/com.microsoft/skip_layer_normalization.hpp"
#include "op/compress.hpp"
#include "op/concat.hpp"
#include "op/constant.hpp"
#include "op/constant_fill.hpp"
#include "op/constant_of_shape.hpp"
#include "op/conv.hpp"
#include "op/conv_integer.hpp"
#include "op/conv_transpose.hpp"
#include "op/cos.hpp"
#include "op/cosh.hpp"
#include "op/crop.hpp"
#include "op/cum_sum.hpp"
#include "op/depth_to_space.hpp"
#include "op/dequantize_linear.hpp"
#include "op/dft.hpp"
#include "op/div.hpp"
#include "op/dropout.hpp"
#include "op/dynamic_quantize_linear.hpp"
#include "op/einsum.hpp"
#include "op/elu.hpp"
#include "op/equal.hpp"
#include "op/erf.hpp"
#include "op/exp.hpp"
#include "op/expand.hpp"
#include "op/eye_like.hpp"
#include "op/flatten.hpp"
#include "op/floor.hpp"
#include "op/gather.hpp"
#include "op/gather_elements.hpp"
#include "op/gather_nd.hpp"
#include "op/gelu.hpp"
#include "op/gemm.hpp"
#include "op/global_average_pool.hpp"
#include "op/global_max_pool.hpp"
#include "op/greater.hpp"
#include "op/greater_or_equal.hpp"
#include "op/grid_sample.hpp"
#include "op/group_normalization.hpp"
#include "op/gru.hpp"
#include "op/hammingwindow.hpp"
#include "op/hannwindow.hpp"
#include "op/hard_sigmoid.hpp"
#include "op/hard_swish.hpp"
#include "op/hardmax.hpp"
#include "op/identity.hpp"
#include "op/if.hpp"
#include "op/image_scaler.hpp"
#include "op/instance_norm.hpp"
#include "op/is_finite.hpp"
#include "op/is_inf.hpp"
#include "op/is_nan.hpp"
#include "op/layer_normalization.hpp"
#include "op/leaky_relu.hpp"
#include "op/less.hpp"
#include "op/less_or_equal.hpp"
#include "op/log.hpp"
#include "op/log_softmax.hpp"
#include "op/loop.hpp"
#include "op/lp_norm.hpp"
#include "op/lp_pool.hpp"
#include "op/lrn.hpp"
#include "op/lstm.hpp"
#include "op/matmul.hpp"
#include "op/matmul_integer.hpp"
#include "op/max.hpp"
#include "op/max_pool.hpp"
#include "op/max_roi_pool.hpp"
#include "op/mean.hpp"
#include "op/mean_variance_normalization.hpp"
#include "op/min.hpp"
#include "op/mish.hpp"
#include "op/mod.hpp"
#include "op/mul.hpp"
#include "op/neg.hpp"
#include "op/nms_rotated.hpp"
#include "op/non_max_suppression.hpp"
#include "op/non_zero.hpp"
#include "op/not.hpp"
#include "op/onehot.hpp"
#include "op/or.hpp"
#include "op/org.openvinotoolkit/deformable_conv_2d.hpp"
#include "op/org.openvinotoolkit/detection_output.hpp"
#include "op/org.openvinotoolkit/experimental_detectron/detection_output.hpp"
#include "op/org.openvinotoolkit/experimental_detectron/generate_proposals_single_image.hpp"
#include "op/org.openvinotoolkit/experimental_detectron/prior_grid_generator.hpp"
#include "op/org.openvinotoolkit/experimental_detectron/roi_feature_extractor.hpp"
#include "op/org.openvinotoolkit/experimental_detectron/topk_rios.hpp"
#include "op/org.openvinotoolkit/fake_quantize.hpp"
#include "op/org.openvinotoolkit/generate_proposals.hpp"
#include "op/org.openvinotoolkit/group_norm.hpp"
#include "op/org.openvinotoolkit/normalize.hpp"
#include "op/org.openvinotoolkit/prior_box.hpp"
#include "op/org.openvinotoolkit/swish.hpp"
#include "op/pad.hpp"
#include "op/pow.hpp"
#include "op/prelu.hpp"
#include "op/qlinear_conv.hpp"
#include "op/qlinear_matmul.hpp"
#include "op/quantize_linear.hpp"
#include "op/random_normal.hpp"
#include "op/random_normal_like.hpp"
#include "op/random_uniform.hpp"
#include "op/random_uniform_like.hpp"
#include "op/range.hpp"
#include "op/reciprocal.hpp"
#include "op/reduce.hpp"
#include "op/relu.hpp"
#include "op/reshape.hpp"
#include "op/resize.hpp"
#include "op/reverse_sequence.hpp"
#include "op/rnn.hpp"
#include "op/roi_align.hpp"
#include "op/round.hpp"
#include "op/scan.hpp"
#include "op/scatter_elements.hpp"
#include "op/scatter_nd.hpp"
#include "op/selu.hpp"
#include "op/shape.hpp"
#include "op/shrink.hpp"
#include "op/sigmoid.hpp"
#include "op/sign.hpp"
#include "op/sin.hpp"
#include "op/sinh.hpp"
#include "op/size.hpp"
#include "op/slice.hpp"
#include "op/softmax.hpp"
#include "op/softplus.hpp"
#include "op/softsign.hpp"
#include "op/space_to_depth.hpp"
#include "op/split.hpp"
#include "op/sqrt.hpp"
#include "op/squeeze.hpp"
#include "op/stft.hpp"
#include "op/sub.hpp"
#include "op/sum.hpp"
#include "op/tan.hpp"
#include "op/tanh.hpp"
#include "op/thresholded_relu.hpp"
#include "op/tile.hpp"
#include "op/topk.hpp"
#include "op/transpose.hpp"
#include "op/trilu.hpp"
#include "op/unique.hpp"
#include "op/unsqueeze.hpp"
#include "op/upsample.hpp"
#include "op/where.hpp"
#include "op/xor.hpp"
#include "openvino/util/log.hpp"

using namespace ov::frontend::onnx;

namespace ov {
namespace frontend {
namespace onnx {

const char* OPENVINO_ONNX_DOMAIN = "org.openvinotoolkit";

namespace {
template <typename Container = std::map<int64_t, Operator>>
typename Container::const_iterator find(int64_t version, const Container& map) {
    // Get the latest version.
    if (version == -1) {
        return map.empty() ? std::end(map) : --std::end(map);
    }
    while (version > 0) {
        const auto it = map.find(version--);
        if (it != std::end(map)) {
            return it;
        }
    }
    return std::end(map);
}
}  // namespace

void OperatorsBridge::register_operator_in_custom_domain(std::string name,
                                                         VersionRange range,
                                                         Operator fn,
                                                         std::string domain,
                                                         std::string warning_mes) {
    for (int version = range.m_since; version <= range.m_until; ++version) {
        register_operator(name, version, domain, fn);
    }
    if (!warning_mes.empty()) {
        OPENVINO_WARN << "Operator: " << name << " since version: " << range.m_since
                      << " until version: " << range.m_until << " registered with warning: " << warning_mes;
    }
}

void OperatorsBridge::register_operator(std::string name, VersionRange range, Operator fn, std::string warning_mes) {
    register_operator_in_custom_domain(name, range, std::move(fn), "", warning_mes);
}

void OperatorsBridge::register_operator(const std::string& name,
                                        int64_t version,
                                        const std::string& domain,
                                        Operator fn) {
    auto it = m_map[domain][name].find(version);
    if (it == std::end(m_map[domain][name])) {
        m_map[domain][name].emplace(version, std::move(fn));
    } else {
        it->second = std::move(fn);
        OPENVINO_WARN << "Overwriting existing operator: " << (domain.empty() ? "ai.onnx" : domain)
                      << "." + name + ":" + std::to_string(version);
    }
}

void OperatorsBridge::unregister_operator(const std::string& name, int64_t version, const std::string& domain) {
    auto domain_it = m_map.find(domain);
    if (domain_it == m_map.end()) {
        OPENVINO_ERR << "unregister_operator: domain '" + domain + "' was not registered before";
        return;
    }
    auto name_it = domain_it->second.find(name);
    if (name_it == domain_it->second.end()) {
        OPENVINO_ERR << "unregister_operator: operator '" + name + "' was not registered before";
        return;
    }
    auto version_it = name_it->second.find(version);
    if (version_it == name_it->second.end()) {
        OPENVINO_ERR << "unregister_operator: operator '" + name + "' with version " + std::to_string(version) +
                            " was not registered before";
        return;
    }
    m_map[domain][name].erase(version_it);
    if (m_map[domain][name].empty()) {
        m_map[domain].erase(name);
        if (m_map[domain].empty()) {
            m_map.erase(domain);
        }
    }
}

OperatorSet OperatorsBridge::get_operator_set(const std::string& domain, int64_t version) const {
    OperatorSet result;

    const auto dm = m_map.find(domain);
    if (dm == std::end(m_map)) {
        OPENVINO_DEBUG << "Domain '" << domain << "' not recognized by OpenVINO";
        return result;
    }
    if (domain == "" && version > LATEST_SUPPORTED_ONNX_OPSET_VERSION) {
        OPENVINO_WARN << "Currently ONNX operator set version: " << version
                      << " is unsupported. Falling back to: " << LATEST_SUPPORTED_ONNX_OPSET_VERSION;
    }
    for (const auto& op : dm->second) {
        const auto& it = find(version, op.second);
        if (it == std::end(op.second)) {
            OPENVINO_THROW("Unsupported operator version: " + (domain.empty() ? "" : domain + ".") + op.first + ":" +
                           std::to_string(version));
        }
        result.emplace(op.first, it->second);
    }
    return result;
}

bool OperatorsBridge::is_operator_registered(const std::string& name,
                                             int64_t version,
                                             const std::string& domain) const {
    // search for domain
    const auto dm_map = m_map.find(domain);
    if (dm_map == std::end(m_map)) {
        return false;
    }
    // search for name
    const auto op_map = dm_map->second.find(name);
    if (op_map == std::end(dm_map->second)) {
        return false;
    }

    return find(version, op_map->second) != std::end(op_map->second);
}

void OperatorsBridge::overwrite_operator(const std::string& name, const std::string& domain, Operator fn) {
    const auto domain_it = m_map.find(domain);
    if (domain_it != m_map.end()) {
        auto& domain_opset = domain_it->second;
        domain_opset[name].clear();
    }
    register_operator(name, 1, domain, std::move(fn));
}

static const char* const MICROSOFT_DOMAIN = "com.microsoft";
static const char* const PYTORCH_ATEN_DOMAIN = "org.pytorch.aten";
static const char* const MMDEPLOY_DOMAIN = "mmdeploy";

#define REGISTER_OPERATOR(name_, ver_, fn_) \
    m_map[""][name_].emplace(ver_, std::bind(op::set_##ver_::fn_, std::placeholders::_1));

#define REGISTER_OPERATOR_WITH_DOMAIN(domain_, name_, ver_, fn_) \
    m_map[domain_][name_].emplace(ver_, std::bind(op::set_##ver_::fn_, std::placeholders::_1));

OperatorsBridge::OperatorsBridge() {
    register_operator("Abs", VersionRange{1, 5}, op::set_1::abs, "Legacy consumed_inputs is not supported");
    register_operator("Abs", VersionRange::since(6), op::set_6::abs);
    register_operator("Acos", VersionRange::single_version_for_all_opsets(), op::set_7::acos);
    register_operator("Acosh", VersionRange::single_version_for_all_opsets(), op::set_9::acosh);
    register_operator("Add", VersionRange{1, 5}, op::set_1::add, "Legacy consumed_inputs is not supported");
    register_operator("Add", VersionRange::in(6), op::set_6::add);
    register_operator("Add", VersionRange{7, 12}, op::set_7::add);
    register_operator("Add", VersionRange::in(13), op::set_13::add);
    register_operator("Add", VersionRange::since(14), op::set_14::add);
    register_operator("And", VersionRange{1, 6}, op::set_1::logical_and);
    register_operator("And", VersionRange::since(6), op::set_7::logical_and);
    // 101468 - Use the VersionRange-based approach for all operators
    REGISTER_OPERATOR("ArgMin", 1, argmin);
    REGISTER_OPERATOR("ArgMin", 12, argmin);
    REGISTER_OPERATOR("ArgMax", 1, argmax);
    REGISTER_OPERATOR("ArgMax", 12, argmax);
    REGISTER_OPERATOR("Asin", 1, asin);
    REGISTER_OPERATOR("Asinh", 1, asinh);
    REGISTER_OPERATOR("Atan", 1, atan);
    REGISTER_OPERATOR("ATen", 1, aten);
    REGISTER_OPERATOR("Atanh", 1, atanh);
    REGISTER_OPERATOR("AveragePool", 1, average_pool);
    REGISTER_OPERATOR("BatchNormalization", 1, batch_norm);
    REGISTER_OPERATOR("BatchNormalization", 7, batch_norm);
    REGISTER_OPERATOR("BatchNormalization", 14, batch_norm);
    REGISTER_OPERATOR("BitShift", 1, bitshift);
    REGISTER_OPERATOR("BitwiseAnd", 1, bitwise_and);
    REGISTER_OPERATOR("BitwiseNot", 1, bitwise_not);
    REGISTER_OPERATOR("BitwiseOr", 1, bitwise_or);
    REGISTER_OPERATOR("BitwiseXor", 1, bitwise_xor);
    REGISTER_OPERATOR("BlackmanWindow", 1, blackmanwindow);
    REGISTER_OPERATOR("Cast", 1, cast);
    REGISTER_OPERATOR("CastLike", 1, cast_like);
    REGISTER_OPERATOR("Ceil", 1, ceil);
    REGISTER_OPERATOR("Celu", 1, celu);
    REGISTER_OPERATOR("Clip", 1, clip);
    REGISTER_OPERATOR("Clip", 11, clip);
    REGISTER_OPERATOR("Concat", 1, concat);
    REGISTER_OPERATOR("Constant", 1, constant);
    REGISTER_OPERATOR("Constant", 13, constant);
    REGISTER_OPERATOR("ConstantOfShape", 1, constant_of_shape);
    REGISTER_OPERATOR("Conv", 1, conv);
    REGISTER_OPERATOR("ConvInteger", 1, conv_integer);
    REGISTER_OPERATOR("ConvTranspose", 1, conv_transpose);
    REGISTER_OPERATOR("Compress", 1, compress);
    REGISTER_OPERATOR("Cos", 1, cos);
    REGISTER_OPERATOR("Cosh", 1, cosh);
    REGISTER_OPERATOR("ConstantFill", 1, constant_fill);
    REGISTER_OPERATOR("CumSum", 1, cum_sum);
    REGISTER_OPERATOR("DepthToSpace", 1, depth_to_space);
    REGISTER_OPERATOR("DequantizeLinear", 1, dequantize_linear);
    REGISTER_OPERATOR("DequantizeLinear", 13, dequantize_linear);
    REGISTER_OPERATOR("Div", 1, div);
    REGISTER_OPERATOR("Div", 7, div);
    REGISTER_OPERATOR("DFT", 1, dft);
    REGISTER_OPERATOR("Dropout", 1, dropout);
    REGISTER_OPERATOR("Dropout", 7, dropout);
    REGISTER_OPERATOR("Dropout", 12, dropout);
    REGISTER_OPERATOR("DynamicQuantizeLinear", 1, dynamic_quantize_linear);
    REGISTER_OPERATOR("Einsum", 1, einsum);
    REGISTER_OPERATOR("Elu", 1, elu);
    REGISTER_OPERATOR("Equal", 1, equal);
    REGISTER_OPERATOR("Erf", 1, erf);
    REGISTER_OPERATOR("Exp", 1, exp);
    REGISTER_OPERATOR("Expand", 1, expand);
    REGISTER_OPERATOR("EyeLike", 1, eye_like);
    REGISTER_OPERATOR("Flatten", 1, flatten);
    REGISTER_OPERATOR("Floor", 1, floor);
    REGISTER_OPERATOR("Gather", 1, gather);
    REGISTER_OPERATOR("GatherElements", 1, gather_elements);
    REGISTER_OPERATOR("GatherND", 1, gather_nd);
    REGISTER_OPERATOR("Gelu", 1, gelu);
    REGISTER_OPERATOR("Gemm", 1, gemm);
    REGISTER_OPERATOR("Gemm", 6, gemm);
    REGISTER_OPERATOR("GlobalAveragePool", 1, global_average_pool);
    REGISTER_OPERATOR("GlobalLpPool", 1, global_lp_pool);
    REGISTER_OPERATOR("GlobalMaxPool", 1, global_max_pool);
    REGISTER_OPERATOR("Greater", 1, greater);
    REGISTER_OPERATOR("GreaterOrEqual", 1, greater_or_equal);
    REGISTER_OPERATOR("GreaterOrEqual", 16, greater_or_equal);
    REGISTER_OPERATOR("GridSample", 1, grid_sample);
    REGISTER_OPERATOR("GroupNormalization", 1, group_normalization);
    REGISTER_OPERATOR("GRU", 1, gru);
    REGISTER_OPERATOR("HannWindow", 1, hannwindow);
    REGISTER_OPERATOR("HammingWindow", 1, hammingwindow);
    REGISTER_OPERATOR("Hardmax", 1, hardmax);
    REGISTER_OPERATOR("Hardmax", 13, hardmax);
    REGISTER_OPERATOR("HardSigmoid", 1, hard_sigmoid);
    REGISTER_OPERATOR("HardSwish", 1, hard_swish);
    REGISTER_OPERATOR("Identity", 1, identity);
    REGISTER_OPERATOR("If", 1, if_op);
    REGISTER_OPERATOR("ImageScaler", 1, image_scaler);
    REGISTER_OPERATOR("InstanceNormalization", 1, instance_norm);
    REGISTER_OPERATOR("IsFinite", 1, is_finite);
    REGISTER_OPERATOR("IsInf", 1, is_inf);
    REGISTER_OPERATOR("IsNaN", 1, is_nan)
    REGISTER_OPERATOR("LayerNormalization", 1, layer_normalization);
    REGISTER_OPERATOR("LeakyRelu", 1, leaky_relu);
    REGISTER_OPERATOR("Less", 1, less);
    REGISTER_OPERATOR("LessOrEqual", 1, less_or_equal);
    REGISTER_OPERATOR("LessOrEqual", 16, less_or_equal);
    REGISTER_OPERATOR("Log", 1, log);
    REGISTER_OPERATOR("LogSoftmax", 1, log_softmax);
    REGISTER_OPERATOR("LogSoftmax", 13, log_softmax);
    REGISTER_OPERATOR("Loop", 1, loop);
    REGISTER_OPERATOR("LpNormalization", 1, lp_norm);
    REGISTER_OPERATOR("LRN", 1, lrn);
    REGISTER_OPERATOR("LSTM", 1, lstm);
    REGISTER_OPERATOR("MatMulInteger", 1, matmul_integer);
    REGISTER_OPERATOR("MatMul", 1, matmul);
    REGISTER_OPERATOR("MaxPool", 1, max_pool);
    REGISTER_OPERATOR("MaxPool", 8, max_pool);
    REGISTER_OPERATOR("MaxRoiPool", 1, max_roi_pool);
    REGISTER_OPERATOR("Max", 1, max);
    REGISTER_OPERATOR("Max", 8, max);
    REGISTER_OPERATOR("Mean", 1, mean);
    REGISTER_OPERATOR("MeanVarianceNormalization", 1, mean_variance_normalization);
    REGISTER_OPERATOR("MeanVarianceNormalization", 9, mean_variance_normalization);
    REGISTER_OPERATOR("Min", 1, min);
    REGISTER_OPERATOR("Min", 8, min);
    REGISTER_OPERATOR("Mish", 1, mish);
    REGISTER_OPERATOR("Mod", 1, mod);
    REGISTER_OPERATOR("Mul", 1, mul);
    REGISTER_OPERATOR("Mul", 7, mul);
    REGISTER_OPERATOR("Neg", 1, neg);
    REGISTER_OPERATOR("NonMaxSuppression", 1, non_max_suppression);
    REGISTER_OPERATOR("NonZero", 1, non_zero);
    REGISTER_OPERATOR("Not", 1, logical_not);
    REGISTER_OPERATOR("Or", 1, logical_or);
    REGISTER_OPERATOR("OneHot", 1, onehot);
    REGISTER_OPERATOR("Pad", 1, pad);
    REGISTER_OPERATOR("Pad", 11, pad);
    REGISTER_OPERATOR("Pow", 1, pow);
    REGISTER_OPERATOR("PRelu", 1, prelu);
    REGISTER_OPERATOR("QLinearConv", 1, qlinear_conv);
    REGISTER_OPERATOR("QLinearMatMul", 1, qlinear_matmul);
    REGISTER_OPERATOR("QuantizeLinear", 1, quantize_linear);
    REGISTER_OPERATOR("QuantizeLinear", 13, quantize_linear);
    REGISTER_OPERATOR("Range", 1, range);
    REGISTER_OPERATOR("RandomNormal", 1, random_normal);
    REGISTER_OPERATOR("RandomNormalLike", 1, random_normal_like);
    REGISTER_OPERATOR("RandomUniform", 1, random_uniform);
    REGISTER_OPERATOR("RandomUniformLike", 1, random_uniform_like);
    REGISTER_OPERATOR("Reciprocal", 1, reciprocal);
    REGISTER_OPERATOR("ReduceLogSum", 1, reduce_log_sum);
    register_operator("ReduceLogSum", VersionRange{1, 17}, op::set_1::reduce_log_sum);
    register_operator("ReduceLogSum", VersionRange::since(18), op::set_18::reduce_log_sum);
    REGISTER_OPERATOR("ReduceLogSumExp", 1, reduce_log_sum_exp);
    REGISTER_OPERATOR("ReduceL1", 1, reduce_l1);
    REGISTER_OPERATOR("ReduceL2", 1, reduce_l2);
    REGISTER_OPERATOR("ReduceMax", 1, reduce_max);
    REGISTER_OPERATOR("ReduceMax", 13, reduce_max);
    REGISTER_OPERATOR("ReduceMax", 18, reduce_max);
    REGISTER_OPERATOR("ReduceMax", 20, reduce_max);
    REGISTER_OPERATOR("ReduceMean", 1, reduce_mean);
    REGISTER_OPERATOR("ReduceMin", 1, reduce_min);
    REGISTER_OPERATOR("ReduceMin", 13, reduce_min);
    REGISTER_OPERATOR("ReduceMin", 18, reduce_min);
    REGISTER_OPERATOR("ReduceMin", 20, reduce_min);
    REGISTER_OPERATOR("ReduceProd", 1, reduce_prod);
    REGISTER_OPERATOR("ReduceSum", 1, reduce_sum);
    REGISTER_OPERATOR("ReduceSum", 13, reduce_sum);
    REGISTER_OPERATOR("ReduceSumSquare", 1, reduce_sum_square);
    REGISTER_OPERATOR("Relu", 1, relu);
    REGISTER_OPERATOR("Reshape", 1, reshape);
    REGISTER_OPERATOR("Resize", 1, resize);
    REGISTER_OPERATOR("Resize", 11, resize);
    REGISTER_OPERATOR("ReverseSequence", 1, reverse_sequence);
    REGISTER_OPERATOR("RNN", 1, rnn);
    REGISTER_OPERATOR("RoiAlign", 1, roi_align);
    REGISTER_OPERATOR("RoiAlign", 16, roi_align);
    REGISTER_OPERATOR("Round", 1, round);
    REGISTER_OPERATOR("Scan", 1, scan);
    REGISTER_OPERATOR("Scan", 9, scan);
    REGISTER_OPERATOR("ScatterElements", 1, scatter_elements);
    REGISTER_OPERATOR("ScatterND", 1, scatter_nd);
    REGISTER_OPERATOR("Selu", 1, selu);
    REGISTER_OPERATOR("Shape", 1, shape);
    REGISTER_OPERATOR("Shape", 15, shape)
    REGISTER_OPERATOR("Shrink", 1, shrink);
    REGISTER_OPERATOR("Sigmoid", 1, sigmoid);
    REGISTER_OPERATOR("Sign", 1, sign);
    REGISTER_OPERATOR("Sin", 1, sin);
    REGISTER_OPERATOR("Sinh", 1, sinh);
    REGISTER_OPERATOR("Size", 1, size);
    REGISTER_OPERATOR("Slice", 1, slice);
    REGISTER_OPERATOR("Slice", 10, slice);
    REGISTER_OPERATOR("Softmax", 1, softmax);
    REGISTER_OPERATOR("Softmax", 11, softmax);
    REGISTER_OPERATOR("Softmax", 13, softmax);
    REGISTER_OPERATOR("Softplus", 1, softplus);
    REGISTER_OPERATOR("Softsign", 1, softsign);
    REGISTER_OPERATOR("SpaceToDepth", 1, space_to_depth);
    REGISTER_OPERATOR("Split", 1, split);
    REGISTER_OPERATOR("Split", 13, split);
    register_operator("STFT",
                      VersionRange::single_version_for_all_opsets(),
                      op::set_17::stft,
                      "frame_step and frame_length inputs must be constants; signal shape must be static;");
    REGISTER_OPERATOR("Sqrt", 1, sqrt);
    REGISTER_OPERATOR("Squeeze", 1, squeeze);
    REGISTER_OPERATOR("Squeeze", 13, squeeze);
    REGISTER_OPERATOR("Sub", 1, sub);
    REGISTER_OPERATOR("Sub", 7, sub);
    REGISTER_OPERATOR("Sum", 1, sum);
    REGISTER_OPERATOR("Sum", 8, sum);
    REGISTER_OPERATOR("Tan", 1, tan);
    REGISTER_OPERATOR("Tanh", 1, tanh);
    REGISTER_OPERATOR("ThresholdedRelu", 1, thresholded_relu);
    REGISTER_OPERATOR("Tile", 1, tile);
    REGISTER_OPERATOR("TopK", 1, topk);
    REGISTER_OPERATOR("TopK", 10, topk);
    REGISTER_OPERATOR("TopK", 11, topk);
    REGISTER_OPERATOR("Transpose", 1, transpose);
    REGISTER_OPERATOR("Trilu", 1, trilu);
    REGISTER_OPERATOR("Unique", 1, unique);
    REGISTER_OPERATOR("Unsqueeze", 1, unsqueeze);
    REGISTER_OPERATOR("Unsqueeze", 13, unsqueeze);
    REGISTER_OPERATOR("Where", 1, where);
    REGISTER_OPERATOR("Xor", 1, logical_xor);

    // deprecated ops
    REGISTER_OPERATOR("Affine", 1, affine);
    REGISTER_OPERATOR("Crop", 1, crop);
    REGISTER_OPERATOR("Scatter", 1, scatter_elements);
    REGISTER_OPERATOR("Upsample", 1, upsample);
    REGISTER_OPERATOR("Upsample", 7, upsample);
    REGISTER_OPERATOR("Upsample", 9, upsample);

    // custom ops
    REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "DeformableConv2D", 1, deformable_conv_2d);
    REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "DetectionOutput", 1, detection_output);
    REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN,
                                  "ExperimentalDetectronDetectionOutput",
                                  1,
                                  experimental_detectron_detection_output);
    REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN,
                                  "ExperimentalDetectronGenerateProposalsSingleImage",
                                  1,
                                  experimental_detectron_generate_proposals);
    REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "ExperimentalDetectronGroupNorm", 1, group_norm);
    REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN,
                                  "ExperimentalDetectronPriorGridGenerator",
                                  1,
                                  experimental_detectron_prior_grid_generator);
    REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN,
                                  "ExperimentalDetectronROIFeatureExtractor",
                                  1,
                                  experimental_detectron_roi_feature_extractor);
    REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN,
                                  "ExperimentalDetectronTopKROIs",
                                  1,
                                  experimental_detectron_topk_rois);
    REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "FakeQuantize", 1, fake_quantize);
    REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "GenerateProposals", 1, generate_proposals);
    REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "GroupNorm", 1, group_norm);
    REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "Normalize", 1, normalize);
    REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "PriorBox", 1, prior_box);
    REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "PriorBoxClustered", 1, prior_box_clustered);
    REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "Swish", 1, swish);

    REGISTER_OPERATOR_WITH_DOMAIN(MICROSOFT_DOMAIN, "Attention", 1, attention);
    REGISTER_OPERATOR_WITH_DOMAIN(MICROSOFT_DOMAIN, "BiasGelu", 1, bias_gelu);
    REGISTER_OPERATOR_WITH_DOMAIN(MICROSOFT_DOMAIN, "EmbedLayerNormalization", 1, embed_layer_normalization);
    REGISTER_OPERATOR_WITH_DOMAIN(MICROSOFT_DOMAIN, "FusedConv", 1, fused_conv);
    REGISTER_OPERATOR_WITH_DOMAIN(MICROSOFT_DOMAIN, "FusedGemm", 1, fusedgemm);
    REGISTER_OPERATOR_WITH_DOMAIN(MICROSOFT_DOMAIN, "GatherND", 1, gather_nd);
    REGISTER_OPERATOR_WITH_DOMAIN(MICROSOFT_DOMAIN, "SkipLayerNormalization", 1, skip_layer_normalization);
    REGISTER_OPERATOR_WITH_DOMAIN(MICROSOFT_DOMAIN, "Trilu", 1, trilu);

    register_operator_in_custom_domain("DequantizeLinear",
                                       VersionRange::since(1),
                                       op::set_13::dequantize_linear,
                                       "com.microsoft");
    register_operator_in_custom_domain("Gelu", VersionRange::since(1), op::set_1::gelu, "com.microsoft");
    register_operator_in_custom_domain("Pad",
                                       VersionRange::single_version_for_all_opsets(),
                                       op::custom::set_1::pad,
                                       "com.microsoft");
    register_operator_in_custom_domain("QuantizeLinear",
                                       VersionRange::since(1),
                                       op::set_13::quantize_linear,
                                       "com.microsoft");

    REGISTER_OPERATOR_WITH_DOMAIN(PYTORCH_ATEN_DOMAIN, "adaptive_avg_pool2d", 1, adaptive_avg_pooling2d);
    REGISTER_OPERATOR_WITH_DOMAIN(MMDEPLOY_DOMAIN, "NMSRotated", 1, nms_rotated);
}

#undef REGISTER_OPERATOR
#undef REGISTER_OPERATOR_WITH_DOMAIN
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
