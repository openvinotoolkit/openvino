// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>

#include "core/attribute.hpp"
#include "ngraph/log.hpp"
#include "op/abs.hpp"
#include "op/acos.hpp"
#include "op/acosh.hpp"
#include "op/add.hpp"
#include "op/and.hpp"
#include "op/argmax.hpp"
#include "op/argmin.hpp"
#include "op/asin.hpp"
#include "op/asinh.hpp"
#include "op/atan.hpp"
#include "op/atanh.hpp"
#include "op/average_pool.hpp"
#include "op/batch_norm.hpp"
#include "op/bitshift.hpp"
#include "op/cast.hpp"
#include "op/ceil.hpp"
#include "op/clip.hpp"
#include "op/compress.hpp"
#include "op/concat.hpp"
#include "op/constant.hpp"
#include "op/constant_fill.hpp"
#include "op/constant_of_shape.hpp"
#include "op/conv.hpp"
// #include "op/conv_integer.hpp"
#include "op/conv_transpose.hpp"
#include "op/cos.hpp"
#include "op/cosh.hpp"
#include "op/cum_sum.hpp"
#include "op/depth_to_space.hpp"
#include "op/dequantize_linear.hpp"
#include "op/div.hpp"
#include "op/dropout.hpp"
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
#include "op/gemm.hpp"
#include "op/global_average_pool.hpp"
#include "op/global_max_pool.hpp"
#include "op/greater.hpp"
#include "op/gru.hpp"
#include "op/hard_sigmoid.hpp"
#include "op/hardmax.hpp"
#include "op/identity.hpp"
#include "op/image_scaler.hpp"
#include "op/instance_norm.hpp"
#include "op/leaky_relu.hpp"
#include "op/less.hpp"
#include "op/log.hpp"
#include "op/log_softmax.hpp"
#include "op/loop.hpp"
#include "op/lp_norm.hpp"
#include "op/lp_pool.hpp"
#include "op/lrn.hpp"
#include "op/lstm.hpp"
#include "op/matmul.hpp"
#include "op/max.hpp"
#include "op/max_pool.hpp"
#include "op/mean.hpp"
#include "op/mean_variance_normalization.hpp"
#include "op/min.hpp"
#include "op/mod.hpp"
#include "op/mul.hpp"
#include "op/neg.hpp"
#include "op/non_max_suppression.hpp"
#include "op/non_zero.hpp"
#include "op/not.hpp"
#include "op/onehot.hpp"
#include "op/or.hpp"
#include "op/pad.hpp"
#include "op/pow.hpp"
#include "op/prelu.hpp"
// #include "op/quant_conv.hpp"
#include "op/quantize_linear.hpp"
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
#include "op/sub.hpp"
#include "op/sum.hpp"
#include "op/tan.hpp"
#include "op/tanh.hpp"
#include "op/thresholded_relu.hpp"
#include "op/tile.hpp"
#include "op/topk.hpp"
#include "op/transpose.hpp"
#include "op/unsqueeze.hpp"
#include "op/upsample.hpp"
#include "op/where.hpp"
#include "op/xor.hpp"
#include "ops_bridge.hpp"

#include "op/org.openvinotoolkit/deformable_conv_2d.hpp"
#include "op/org.openvinotoolkit/detection_output.hpp"
#include "op/org.openvinotoolkit/experimental_detectron/detection_output.hpp"
#include "op/org.openvinotoolkit/experimental_detectron/generate_proposals_single_image.hpp"
#include "op/org.openvinotoolkit/experimental_detectron/prior_grid_generator.hpp"
#include "op/org.openvinotoolkit/experimental_detectron/roi_feature_extractor.hpp"
#include "op/org.openvinotoolkit/experimental_detectron/topk_rios.hpp"
#include "op/org.openvinotoolkit/fake_quantize.hpp"
#include "op/org.openvinotoolkit/group_norm.hpp"
#include "op/org.openvinotoolkit/normalize.hpp"
#include "op/org.openvinotoolkit/prior_box.hpp"
#include "op/org.openvinotoolkit/swish.hpp"

namespace ngraph
{
    namespace onnx_import
    {
        namespace detail
        {
            const std::map<std::int64_t, Operator>::const_iterator
                find(std::int64_t version, const std::map<std::int64_t, Operator>& map)
            {
                // Get the latest version.
                if (version == -1)
                {
                    return map.empty() ? std::end(map) : --std::end(map);
                }
                while (version > 0)
                {
                    std::map<std::int64_t, Operator>::const_iterator it = map.find(version--);
                    if (it != std::end(map))
                    {
                        return it;
                    }
                }
                return std::end(map);
            }
        } // namespace detail

        void OperatorsBridge::_register_operator(const std::string& name,
                                                 std::int64_t version,
                                                 const std::string& domain,
                                                 Operator fn)
        {
            std::lock_guard<std::mutex> guard(lock);

            auto it = m_map[domain][name].find(version);
            if (it == std::end(m_map[domain][name]))
            {
                m_map[domain][name].emplace(version, std::move(fn));
            }
            else
            {
                it->second = std::move(fn);
                NGRAPH_WARN << "Overwriting existing operator: "
                            << (domain.empty() ? "ai.onnx" : domain)
                            << "." + name + ":" + std::to_string(version);
            }
        }

        void OperatorsBridge::_unregister_operator(const std::string& name,
                                                   std::int64_t version,
                                                   const std::string& domain)
        {
            std::lock_guard<std::mutex> guard(lock);

            auto domain_it = m_map.find(domain);
            if (domain_it == m_map.end())
            {
                NGRAPH_ERR << "unregister_operator: domain '" + domain +
                                  "' was not registered before";
                return;
            }
            auto name_it = domain_it->second.find(name);
            if (name_it == domain_it->second.end())
            {
                NGRAPH_ERR << "unregister_operator: operator '" + name +
                                  "' was not registered before";
                return;
            }
            auto version_it = name_it->second.find(version);
            if (version_it == name_it->second.end())
            {
                NGRAPH_ERR << "unregister_operator: operator '" + name + "' with version " +
                                  std::to_string(version) + " was not registered before";
                return;
            }
            m_map[domain][name].erase(version_it);
            if (m_map[domain][name].empty())
            {
                m_map[domain].erase(name);
                if (m_map[domain].empty())
                {
                    m_map.erase(domain);
                }
            }
        }

        OperatorSet OperatorsBridge::_get_operator_set(const std::string& domain,
                                                       std::int64_t version)
        {
            std::lock_guard<std::mutex> guard(lock);

            OperatorSet result;

            auto dm = m_map.find(domain);
            if (dm == std::end(m_map))
            {
                NGRAPH_DEBUG << "Domain '" << domain << "' not recognized by nGraph";
                return OperatorSet{};
            }
            if (domain == "" && version > OperatorsBridge::LATEST_SUPPORTED_ONNX_OPSET_VERSION)
            {
                NGRAPH_WARN << "Currently ONNX operator set version: " << version
                            << " is unsupported. Falling back to: "
                            << OperatorsBridge::LATEST_SUPPORTED_ONNX_OPSET_VERSION;
            }
            for (const auto& op : dm->second)
            {
                const auto& it = detail::find(version, op.second);
                if (it == std::end(op.second))
                {
                    throw error::UnsupportedVersion{op.first, version, domain};
                }
                result.emplace(op.first, it->second);
            }
            return result;
        }

        bool OperatorsBridge::_is_operator_registered(const std::string& name,
                                                      std::int64_t version,
                                                      const std::string& domain)
        {
            std::lock_guard<std::mutex> guard(lock);
            // search for domain
            auto dm_map = m_map.find(domain);
            if (dm_map == std::end(m_map))
            {
                return false;
            }
            // search for name
            auto op_map = dm_map->second.find(name);
            if (op_map == std::end(dm_map->second))
            {
                return false;
            }

            if (detail::find(version, op_map->second) != std::end(op_map->second))
            {
                return true;
            }
            else
            {
                return false;
            }
        }

#define REGISTER_OPERATOR(name_, ver_, fn_)                                                        \
    m_map[""][name_].emplace(ver_, std::bind(op::set_##ver_::fn_, std::placeholders::_1))

#define REGISTER_OPERATOR_WITH_DOMAIN(domain_, name_, ver_, fn_)                                   \
    m_map[domain_][name_].emplace(ver_, std::bind(op::set_##ver_::fn_, std::placeholders::_1))

        OperatorsBridge::OperatorsBridge()
        {
            REGISTER_OPERATOR("Abs", 1, abs);
            REGISTER_OPERATOR("Acos", 1, acos);
            REGISTER_OPERATOR("Acosh", 1, acosh);
            REGISTER_OPERATOR("Add", 1, add);
            REGISTER_OPERATOR("Add", 7, add);
            REGISTER_OPERATOR("And", 1, logical_and);
            REGISTER_OPERATOR("ArgMin", 1, argmin);
            REGISTER_OPERATOR("ArgMin", 12, argmin);
            REGISTER_OPERATOR("ArgMax", 1, argmax);
            REGISTER_OPERATOR("ArgMax", 12, argmax);
            REGISTER_OPERATOR("Asin", 1, asin);
            REGISTER_OPERATOR("Asinh", 1, asinh);
            REGISTER_OPERATOR("Atan", 1, atan);
            REGISTER_OPERATOR("Atanh", 1, atanh);
            REGISTER_OPERATOR("AveragePool", 1, average_pool);
            REGISTER_OPERATOR("BatchNormalization", 1, batch_norm);
            REGISTER_OPERATOR("BatchNormalization", 7, batch_norm);
            REGISTER_OPERATOR("BitShift", 1, bitshift);
            REGISTER_OPERATOR("Cast", 1, cast);
            REGISTER_OPERATOR("Ceil", 1, ceil);
            REGISTER_OPERATOR("Clip", 1, clip);
            REGISTER_OPERATOR("Clip", 11, clip);
            REGISTER_OPERATOR("Concat", 1, concat);
            REGISTER_OPERATOR("Constant", 1, constant);
            REGISTER_OPERATOR("Constant", 13, constant);
            REGISTER_OPERATOR("ConstantOfShape", 1, constant_of_shape);
            REGISTER_OPERATOR("Conv", 1, conv);
            // REGISTER_OPERATOR("ConvInteger", 1, conv_integer);
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
            REGISTER_OPERATOR("Dropout", 1, dropout);
            REGISTER_OPERATOR("Dropout", 7, dropout);
            REGISTER_OPERATOR("Dropout", 12, dropout);
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
            REGISTER_OPERATOR("Gemm", 1, gemm);
            REGISTER_OPERATOR("Gemm", 6, gemm);
            REGISTER_OPERATOR("GlobalAveragePool", 1, global_average_pool);
            REGISTER_OPERATOR("GlobalLpPool", 1, global_lp_pool);
            REGISTER_OPERATOR("GlobalMaxPool", 1, global_max_pool);
            REGISTER_OPERATOR("Greater", 1, greater);
            REGISTER_OPERATOR("GRU", 1, gru);
            REGISTER_OPERATOR("Hardmax", 1, hardmax);
            REGISTER_OPERATOR("Hardmax", 13, hardmax);
            REGISTER_OPERATOR("HardSigmoid", 1, hard_sigmoid);
            REGISTER_OPERATOR("Identity", 1, identity);
            REGISTER_OPERATOR("ImageScaler", 1, image_scaler);
            REGISTER_OPERATOR("InstanceNormalization", 1, instance_norm);
            REGISTER_OPERATOR("LeakyRelu", 1, leaky_relu);
            REGISTER_OPERATOR("Less", 1, less);
            REGISTER_OPERATOR("Log", 1, log);
            REGISTER_OPERATOR("LogSoftmax", 1, log_softmax);
            REGISTER_OPERATOR("LogSoftmax", 13, log_softmax);
            REGISTER_OPERATOR("Loop", 1, loop);
            REGISTER_OPERATOR("LpNormalization", 1, lp_norm);
            REGISTER_OPERATOR("LRN", 1, lrn);
            REGISTER_OPERATOR("LSTM", 1, lstm);
            REGISTER_OPERATOR("MatMul", 1, matmul);
            REGISTER_OPERATOR("MaxPool", 1, max_pool);
            REGISTER_OPERATOR("Max", 1, max);
            REGISTER_OPERATOR("Max", 8, max);
            REGISTER_OPERATOR("Mean", 1, mean);
            REGISTER_OPERATOR("MeanVarianceNormalization", 1, mean_variance_normalization);
            REGISTER_OPERATOR("MeanVarianceNormalization", 9, mean_variance_normalization);
            REGISTER_OPERATOR("Min", 1, min);
            REGISTER_OPERATOR("Min", 8, min);
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
            // REGISTER_OPERATOR("QLinearConv", 1, quant_conv);
            REGISTER_OPERATOR("QuantizeLinear", 1, quantize_linear);
            REGISTER_OPERATOR("QuantizeLinear", 13, quantize_linear);
            REGISTER_OPERATOR("Range", 1, range);
            REGISTER_OPERATOR("Reciprocal", 1, reciprocal);
            REGISTER_OPERATOR("ReduceLogSum", 1, reduce_log_sum);
            REGISTER_OPERATOR("ReduceLogSumExp", 1, reduce_log_sum_exp);
            REGISTER_OPERATOR("ReduceL1", 1, reduce_l1);
            REGISTER_OPERATOR("ReduceL2", 1, reduce_l2);
            REGISTER_OPERATOR("ReduceMax", 1, reduce_max);
            REGISTER_OPERATOR("ReduceMean", 1, reduce_mean);
            REGISTER_OPERATOR("ReduceMin", 1, reduce_min);
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
            REGISTER_OPERATOR("Round", 1, round);
            REGISTER_OPERATOR("Scatter", 1, scatter_elements);
            REGISTER_OPERATOR("ScatterElements", 1, scatter_elements);
            REGISTER_OPERATOR("ScatterND", 1, scatter_nd);
            REGISTER_OPERATOR("Selu", 1, selu);
            REGISTER_OPERATOR("Shape", 1, shape);
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
            REGISTER_OPERATOR("Unsqueeze", 1, unsqueeze);
            REGISTER_OPERATOR("Unsqueeze", 13, unsqueeze);
            REGISTER_OPERATOR("Upsample", 1, upsample);
            REGISTER_OPERATOR("Upsample", 7, upsample);
            REGISTER_OPERATOR("Upsample", 9, upsample);
            REGISTER_OPERATOR("Where", 1, where);
            REGISTER_OPERATOR("Xor", 1, logical_xor);

            // custom OPs
            REGISTER_OPERATOR_WITH_DOMAIN(
                OPENVINO_ONNX_DOMAIN, "DeformableConv2D", 1, deformable_conv_2d);
            REGISTER_OPERATOR_WITH_DOMAIN(
                OPENVINO_ONNX_DOMAIN, "DetectionOutput", 1, detection_output);
            REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN,
                                          "ExperimentalDetectronDetectionOutput",
                                          1,
                                          experimental_detectron_detection_output);
            REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN,
                                          "ExperimentalDetectronGenerateProposalsSingleImage",
                                          1,
                                          experimental_detectron_generate_proposals);
            REGISTER_OPERATOR_WITH_DOMAIN(
                OPENVINO_ONNX_DOMAIN, "ExperimentalDetectronGroupNorm", 1, group_norm);
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
            REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "GroupNorm", 1, group_norm);
            REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "Normalize", 1, normalize);
            REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "PriorBox", 1, prior_box);
            REGISTER_OPERATOR_WITH_DOMAIN(
                OPENVINO_ONNX_DOMAIN, "PriorBoxClustered", 1, prior_box_clustered);
            REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "Swish", 1, swish);
        }

#undef REGISTER_OPERATOR
#undef REGISTER_OPERATOR_WITH_DOMAIN
    } // namespace onnx_import

} // namespace ngraph
