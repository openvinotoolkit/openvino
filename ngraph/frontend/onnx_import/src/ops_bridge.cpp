//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <functional>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>

#include "ngraph/log.hpp"
#include "onnx_import/core/attribute.hpp"
#include "onnx_import/op/abs.hpp"
#include "onnx_import/op/acos.hpp"
#include "onnx_import/op/acosh.hpp"
#include "onnx_import/op/add.hpp"
#include "onnx_import/op/and.hpp"
#include "onnx_import/op/argmax.hpp"
#include "onnx_import/op/argmin.hpp"
#include "onnx_import/op/asin.hpp"
#include "onnx_import/op/asinh.hpp"
#include "onnx_import/op/atan.hpp"
#include "onnx_import/op/atanh.hpp"
#include "onnx_import/op/average_pool.hpp"
#include "onnx_import/op/batch_norm.hpp"
#include "onnx_import/op/cast.hpp"
#include "onnx_import/op/ceil.hpp"
#include "onnx_import/op/clip.hpp"
#include "onnx_import/op/concat.hpp"
#include "onnx_import/op/constant.hpp"
#include "onnx_import/op/constant_of_shape.hpp"
#include "onnx_import/op/conv.hpp"
// #include "onnx_import/op/conv_integer.hpp"
#include "onnx_import/op/conv_transpose.hpp"
#include "onnx_import/op/cos.hpp"
#include "onnx_import/op/cosh.hpp"
#include "onnx_import/op/cum_sum.hpp"
#include "onnx_import/op/depth_to_space.hpp"
#include "onnx_import/op/dequantize_linear.hpp"
#include "onnx_import/op/div.hpp"
#include "onnx_import/op/dropout.hpp"
#include "onnx_import/op/elu.hpp"
#include "onnx_import/op/equal.hpp"
#include "onnx_import/op/erf.hpp"
#include "onnx_import/op/exp.hpp"
#include "onnx_import/op/expand.hpp"
#include "onnx_import/op/eye_like.hpp"
#include "onnx_import/op/flatten.hpp"
#include "onnx_import/op/floor.hpp"
#include "onnx_import/op/gather.hpp"
// #include "onnx_import/op/gather_nd.hpp"
#include "onnx_import/op/gemm.hpp"
#include "onnx_import/op/global_average_pool.hpp"
#include "onnx_import/op/global_max_pool.hpp"
#include "onnx_import/op/greater.hpp"
#include "onnx_import/op/gru.hpp"
#include "onnx_import/op/hard_sigmoid.hpp"
#include "onnx_import/op/hardmax.hpp"
#include "onnx_import/op/identity.hpp"
#include "onnx_import/op/image_scaler.hpp"
#include "onnx_import/op/instance_norm.hpp"
#include "onnx_import/op/leaky_relu.hpp"
#include "onnx_import/op/less.hpp"
#include "onnx_import/op/log.hpp"
#include "onnx_import/op/log_softmax.hpp"
#include "onnx_import/op/loop.hpp"
#include "onnx_import/op/lp_norm.hpp"
#include "onnx_import/op/lp_pool.hpp"
#include "onnx_import/op/lrn.hpp"
#include "onnx_import/op/lstm.hpp"
#include "onnx_import/op/matmul.hpp"
#include "onnx_import/op/matmul_integer.hpp"
#include "onnx_import/op/max.hpp"
#include "onnx_import/op/max_pool.hpp"
#include "onnx_import/op/mean.hpp"
#include "onnx_import/op/mean_variance_normalization.hpp"
#include "onnx_import/op/min.hpp"
#include "onnx_import/op/mod.hpp"
#include "onnx_import/op/mul.hpp"
#include "onnx_import/op/neg.hpp"
#include "onnx_import/op/non_max_suppression.hpp"
#include "onnx_import/op/non_zero.hpp"
#include "onnx_import/op/not.hpp"
#include "onnx_import/op/onehot.hpp"
#include "onnx_import/op/or.hpp"
#include "onnx_import/op/pad.hpp"
#include "onnx_import/op/pow.hpp"
#include "onnx_import/op/prelu.hpp"
#include "onnx_import/op/qlinear_matmul.hpp"
// #include "onnx_import/op/quant_conv.hpp"
#include "onnx_import/op/quantize_linear.hpp"
#include "onnx_import/op/range.hpp"
#include "onnx_import/op/reciprocal.hpp"
#include "onnx_import/op/reduce.hpp"
#include "onnx_import/op/relu.hpp"
#include "onnx_import/op/reshape.hpp"
#include "onnx_import/op/resize.hpp"
#include "onnx_import/op/reverse_sequence.hpp"
#include "onnx_import/op/rnn.hpp"
#include "onnx_import/op/roi_align.hpp"
#include "onnx_import/op/round.hpp"
#include "onnx_import/op/scatter_elements.hpp"
#include "onnx_import/op/scatter_nd.hpp"
#include "onnx_import/op/selu.hpp"
#include "onnx_import/op/shape.hpp"
#include "onnx_import/op/shrink.hpp"
#include "onnx_import/op/sigmoid.hpp"
#include "onnx_import/op/sign.hpp"
#include "onnx_import/op/sin.hpp"
#include "onnx_import/op/sinh.hpp"
#include "onnx_import/op/size.hpp"
#include "onnx_import/op/slice.hpp"
#include "onnx_import/op/softmax.hpp"
#include "onnx_import/op/softplus.hpp"
#include "onnx_import/op/softsign.hpp"
#include "onnx_import/op/space_to_depth.hpp"
#include "onnx_import/op/split.hpp"
#include "onnx_import/op/sqrt.hpp"
#include "onnx_import/op/squeeze.hpp"
#include "onnx_import/op/sub.hpp"
#include "onnx_import/op/sum.hpp"
#include "onnx_import/op/tan.hpp"
#include "onnx_import/op/tanh.hpp"
#include "onnx_import/op/thresholded_relu.hpp"
#include "onnx_import/op/tile.hpp"
#include "onnx_import/op/topk.hpp"
#include "onnx_import/op/transpose.hpp"
#include "onnx_import/op/unsqueeze.hpp"
#include "onnx_import/op/upsample.hpp"
#include "onnx_import/op/where.hpp"
#include "onnx_import/op/xor.hpp"
#include "onnx_import/ops_bridge.hpp"

#include "onnx_import/op/org.openvinotoolkit/detection_output.hpp"
#include "onnx_import/op/org.openvinotoolkit/fake_quantize.hpp"
#include "onnx_import/op/org.openvinotoolkit/group_norm.hpp"
#include "onnx_import/op/org.openvinotoolkit/normalize.hpp"
#include "onnx_import/op/org.openvinotoolkit/prior_box.hpp"

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
        }

        void OperatorsBridge::_register_operator(const std::string& name,
                                                 std::int64_t version,
                                                 const std::string& domain,
                                                 Operator fn)
        {
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

        OperatorSet OperatorsBridge::_get_operator_set(const std::string& domain,
                                                       std::int64_t version)
        {
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
            REGISTER_OPERATOR("ArgMax", 1, argmax);
            REGISTER_OPERATOR("Asin", 1, asin);
            REGISTER_OPERATOR("Asinh", 1, asinh);
            REGISTER_OPERATOR("Atan", 1, atan);
            REGISTER_OPERATOR("Atanh", 1, atanh);
            REGISTER_OPERATOR("AveragePool", 1, average_pool);
            REGISTER_OPERATOR("BatchNormalization", 1, batch_norm);
            REGISTER_OPERATOR("Cast", 1, cast);
            REGISTER_OPERATOR("Ceil", 1, ceil);
            REGISTER_OPERATOR("Clip", 1, clip);
            REGISTER_OPERATOR("Clip", 11, clip);
            REGISTER_OPERATOR("Concat", 1, concat);
            REGISTER_OPERATOR("Constant", 1, constant);
            REGISTER_OPERATOR("ConstantOfShape", 1, constant_of_shape);
            REGISTER_OPERATOR("Conv", 1, conv);
            // REGISTER_OPERATOR("ConvInteger", 1, conv_integer);
            REGISTER_OPERATOR("ConvTranspose", 1, conv_transpose);
            REGISTER_OPERATOR("Cos", 1, cos);
            REGISTER_OPERATOR("Cosh", 1, cosh);
            REGISTER_OPERATOR("CumSum", 1, cum_sum);
            REGISTER_OPERATOR("DepthToSpace", 1, depth_to_space);
            REGISTER_OPERATOR("DequantizeLinear", 1, dequantize_linear);
            REGISTER_OPERATOR("DequantizeLinear", 13, dequantize_linear);
            REGISTER_OPERATOR("Div", 1, div);
            REGISTER_OPERATOR("Div", 7, div);
            REGISTER_OPERATOR("Dropout", 1, dropout);
            REGISTER_OPERATOR("Elu", 1, elu);
            REGISTER_OPERATOR("Equal", 1, equal);
            REGISTER_OPERATOR("Erf", 1, erf);
            REGISTER_OPERATOR("Exp", 1, exp);
            REGISTER_OPERATOR("Expand", 1, expand);
            REGISTER_OPERATOR("EyeLike", 1, eye_like);
            REGISTER_OPERATOR("Flatten", 1, flatten);
            REGISTER_OPERATOR("Floor", 1, floor);
            REGISTER_OPERATOR("Gather", 1, gather);
            // REGISTER_OPERATOR("GatherND", 1, gather_nd);
            REGISTER_OPERATOR("Gemm", 1, gemm);
            REGISTER_OPERATOR("Gemm", 6, gemm);
            REGISTER_OPERATOR("GlobalAveragePool", 1, global_average_pool);
            REGISTER_OPERATOR("GlobalLpPool", 1, global_lp_pool);
            REGISTER_OPERATOR("GlobalMaxPool", 1, global_max_pool);
            REGISTER_OPERATOR("Greater", 1, greater);
            REGISTER_OPERATOR("GRU", 1, gru);
            REGISTER_OPERATOR("Hardmax", 1, hardmax);
            REGISTER_OPERATOR("HardSigmoid", 1, hard_sigmoid);
            REGISTER_OPERATOR("Identity", 1, identity);
            REGISTER_OPERATOR("ImageScaler", 1, image_scaler);
            REGISTER_OPERATOR("InstanceNormalization", 1, instance_norm);
            REGISTER_OPERATOR("LeakyRelu", 1, leaky_relu);
            REGISTER_OPERATOR("Less", 1, less);
            REGISTER_OPERATOR("Log", 1, log);
            REGISTER_OPERATOR("LogSoftmax", 1, log_softmax);
            // REGISTER_OPERATOR("Loop", 1, loop); // Loop operator disabled for the 2021.1 release
            REGISTER_OPERATOR("LpNormalization", 1, lp_norm);
            REGISTER_OPERATOR("LRN", 1, lrn);
            REGISTER_OPERATOR("LSTM", 1, lstm);
            REGISTER_OPERATOR("MatMul", 1, matmul);
            REGISTER_OPERATOR("MatMulInteger", 1, matmul_integer);
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
            REGISTER_OPERATOR("QLinearMatMul", 1, qlinear_matmul);
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
            REGISTER_OPERATOR("Softplus", 1, softplus);
            REGISTER_OPERATOR("Softsign", 1, softsign);
            REGISTER_OPERATOR("SpaceToDepth", 1, space_to_depth);
            REGISTER_OPERATOR("Split", 1, split);
            REGISTER_OPERATOR("Sqrt", 1, sqrt);
            REGISTER_OPERATOR("Squeeze", 1, squeeze);
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
            REGISTER_OPERATOR("Upsample", 1, upsample);
            REGISTER_OPERATOR("Upsample", 9, upsample);
            REGISTER_OPERATOR("Where", 1, where);
            REGISTER_OPERATOR("Xor", 1, logical_xor);

            // custom OPs
            REGISTER_OPERATOR_WITH_DOMAIN(
                OPENVINO_ONNX_DOMAIN, "DetectionOutput", 1, detection_output);
            REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "FakeQuantize", 1, fake_quantize);
            REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "GroupNorm", 1, group_norm);
            REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "Normalize", 1, normalize);
            REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "PriorBox", 1, prior_box);
        }

#undef REGISTER_OPERATOR
#undef REGISTER_OPERATOR_WITH_DOMAIN
    } // namespace onnx_import

} // namespace ngraph
