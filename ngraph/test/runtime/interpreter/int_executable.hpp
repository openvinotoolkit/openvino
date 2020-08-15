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

#pragma once

#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <ngraph/runtime/host_tensor.hpp>
#include "backend.hpp"
#include "int_backend_visibility.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/reference/abs.hpp"
#include "ngraph/runtime/reference/acos.hpp"
#include "ngraph/runtime/reference/any.hpp"
#include "ngraph/runtime/reference/asin.hpp"
#include "ngraph/runtime/reference/atan.hpp"
#include "ngraph/runtime/reference/atan2.hpp"
#include "ngraph/runtime/reference/avg_pool.hpp"
#include "ngraph/runtime/reference/batch_norm.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"
#include "ngraph/runtime/reference/ceiling.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/runtime/reference/constant.hpp"
#include "ngraph/runtime/reference/convert.hpp"
#include "ngraph/runtime/reference/convolution.hpp"
#include "ngraph/runtime/reference/cos.hpp"
#include "ngraph/runtime/reference/cosh.hpp"
#include "ngraph/runtime/reference/cum_sum.hpp"
#include "ngraph/runtime/reference/dequantize.hpp"
#include "ngraph/runtime/reference/dot.hpp"
#include "ngraph/runtime/reference/elu.hpp"
#include "ngraph/runtime/reference/embedding_bag_offsets_sum.hpp"
#include "ngraph/runtime/reference/embedding_bag_packed_sum.hpp"
#include "ngraph/runtime/reference/embedding_segments_sum.hpp"
#include "ngraph/runtime/reference/erf.hpp"
#include "ngraph/runtime/reference/exp.hpp"
#include "ngraph/runtime/reference/extract_image_patches.hpp"
#include "ngraph/runtime/reference/floor.hpp"
#include "ngraph/runtime/reference/gather.hpp"
#include "ngraph/runtime/reference/gather_nd.hpp"
#include "ngraph/runtime/reference/log.hpp"
#include "ngraph/runtime/reference/lrn.hpp"
#include "ngraph/runtime/reference/matmul.hpp"
#include "ngraph/runtime/reference/max.hpp"
#include "ngraph/runtime/reference/max_pool.hpp"
#include "ngraph/runtime/reference/min.hpp"
#include "ngraph/runtime/reference/negate.hpp"
#include "ngraph/runtime/reference/not.hpp"
#include "ngraph/runtime/reference/one_hot.hpp"
#include "ngraph/runtime/reference/pad.hpp"
#include "ngraph/runtime/reference/product.hpp"
#include "ngraph/runtime/reference/quantize.hpp"
#include "ngraph/runtime/reference/relu.hpp"
#include "ngraph/runtime/reference/replace_slice.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/result.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/reverse_sequence.hpp"
#include "ngraph/runtime/reference/round.hpp"
#include "ngraph/runtime/reference/select.hpp"
#include "ngraph/runtime/reference/sigmoid.hpp"
#include "ngraph/runtime/reference/sign.hpp"
#include "ngraph/runtime/reference/sin.hpp"
#include "ngraph/runtime/reference/sinh.hpp"
#include "ngraph/runtime/reference/softmax.hpp"
#include "ngraph/runtime/reference/sqrt.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/runtime/reference/tan.hpp"
#include "ngraph/runtime/reference/tanh.hpp"
#include "ngraph/runtime/reference/topk.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "op/avg_pool.hpp"
#include "op/convolution.hpp"
#include "op/group_conv.hpp"

#include "reference/detection_output.hpp"
#include "reference/scatter_nd_update.hpp"
#include "reference/scatter_update.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class INTBackend;
            class INTExecutable;

            // This expands the op list in op_tbl.hpp into a list of enumerations that look like
            // this:
            // Abs,
            // Acos,
            // ...
            enum class OP_TYPEID
            {
#define NGRAPH_OP(NAME, NAMESPACE) ID_SUFFIX(NAME),
#include "opset_int_tbl.hpp"
#undef NGRAPH_OP
                UnknownOp
            };
        } // namespace interpreter
    }     // namespace runtime
} // namespace ngraph

class INTERPRETER_BACKEND_API ngraph::runtime::interpreter::INTExecutable : public Executable
{
    friend class INTBackend;

public:
    INTExecutable(const std::shared_ptr<Function>& function,
                  bool enable_performance_collection = false);

    bool call(const std::vector<std::shared_ptr<Tensor>>& outputs,
              const std::vector<std::shared_ptr<Tensor>>& inputs) override;

    void set_nan_check(bool enable);

    std::vector<PerformanceCounter> get_performance_data() const override;

    std::shared_ptr<runtime::Tensor> create_input_tensor(size_t input_index) override;

    std::shared_ptr<runtime::Tensor> create_output_tensor(size_t output_index) override;

    std::vector<std::shared_ptr<runtime::Tensor>>
        create_input_tensor(size_t input_index, size_t pipeline_depth) override;

    std::vector<std::shared_ptr<runtime::Tensor>>
        create_output_tensor(size_t output_index, size_t pipeline_depth) override;

protected:
    std::shared_ptr<ngraph::op::Parameter> get_parameter(size_t index) const;
    std::shared_ptr<ngraph::op::Result> get_result(size_t index) const;
    int get_alignment() const { return 64; }
    bool m_is_compiled = false;
    bool m_nan_check_enabled = false;
    bool m_performance_counters_enabled = false;
    std::shared_ptr<Function> m_function;
    std::unordered_map<std::shared_ptr<const Node>, stopwatch> m_timer_map;
    std::vector<std::shared_ptr<Node>> m_nodes;
    std::set<std::string> m_unsupported_op_name_list;

    static OP_TYPEID get_typeid(const Node& node);

    static void perform_nan_check(const std::vector<std::shared_ptr<HostTensor>>&,
                                  const Node* op = nullptr);

    virtual void generate_calls(const element::Type& type,
                                const Node& op,
                                const std::vector<std::shared_ptr<HostTensor>>& outputs,
                                const std::vector<std::shared_ptr<HostTensor>>& inputs);

    template <typename T>
    void op_engine(const Node& node,
                   const std::vector<std::shared_ptr<HostTensor>>& out,
                   const std::vector<std::shared_ptr<HostTensor>>& args)
    {
// We want to check that every OP_TYPEID enumeration is included in the list.
// These GCC flags enable compile-time checking so that if an enumeration
// is not in the list an error is generated.
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
        switch (get_typeid(node))
        {
        case OP_TYPEID::Abs:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::abs<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Acos:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::acos<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Any:
        {
            const op::Any* any = static_cast<const op::Any*>(&node);
            reference::any(args[0]->get_data_ptr<const char>(),
                           out[0]->get_data_ptr<char>(),
                           node.get_input_shape(0),
                           node.get_output_shape(0),
                           any->get_reduction_axes());
            break;
        }
        case OP_TYPEID::Asin:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::asin<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Atan:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::atan<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Elu:
        {
            const op::Elu* elu_node = static_cast<const op::Elu*>(&node);

            size_t element_count = shape_size(node.get_output_shape(0));
            reference::elu<T>(args[0]->get_data_ptr<const T>(),
                              out[0]->get_data_ptr<T>(),
                              element_count,
                              elu_node->get_alpha());
            break;
        }
        case OP_TYPEID::AvgPool:
        {
            const op::v0::AvgPool* avg_pool = static_cast<const op::v0::AvgPool*>(&node);

            reference::avg_pool<T>(args[0]->get_data_ptr<const T>(),
                                   out[0]->get_data_ptr<T>(),
                                   node.get_input_shape(0),
                                   node.get_output_shape(0),
                                   avg_pool->get_window_shape(),
                                   avg_pool->get_window_movement_strides(),
                                   avg_pool->get_padding_below(),
                                   avg_pool->get_padding_above(),
                                   avg_pool->get_include_padding_in_avg_computation());
            break;
        }
        case OP_TYPEID::BatchNormInference:
        {
            const ngraph::op::BatchNormInference* bn =
                static_cast<const ngraph::op::BatchNormInference*>(&node);
            reference::batch_norm_inference<T>(bn->get_eps_value(),
                                               args[0]->get_data_ptr<const T>(),
                                               args[1]->get_data_ptr<const T>(),
                                               args[2]->get_data_ptr<const T>(),
                                               args[3]->get_data_ptr<const T>(),
                                               args[4]->get_data_ptr<const T>(),
                                               out[0]->get_data_ptr<T>(),
                                               node.get_input_shape(2));
            break;
        }
        case OP_TYPEID::BroadcastLike: break;
        case OP_TYPEID::Ceiling:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::ceiling<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Convert:
        {
            // const op::Convert* c = static_cast<const op::Convert*>(&node);
            element::Type type = node.get_element_type();
            std::stringstream ss;
            size_t element_count = shape_size(node.get_output_shape(0));
            switch (type)
            {
            case element::Type_t::boolean:
                reference::convert_to_bool<T>(
                    args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<char>(), element_count);
                break;
            case element::Type_t::f32:
                reference::convert<T>(
                    args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<float>(), element_count);
                break;
            case element::Type_t::f64:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<double>(),
                                      element_count);
                break;
            case element::Type_t::i8:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<int8_t>(),
                                      element_count);
                break;
            case element::Type_t::i16:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<int16_t>(),
                                      element_count);
                break;
            case element::Type_t::i32:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<int32_t>(),
                                      element_count);
                break;
            case element::Type_t::i64:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<int64_t>(),
                                      element_count);
                break;
            case element::Type_t::u8:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<uint8_t>(),
                                      element_count);
                break;
            case element::Type_t::u16:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<uint16_t>(),
                                      element_count);
                break;
            case element::Type_t::u32:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<uint32_t>(),
                                      element_count);
                break;
            case element::Type_t::u64:
                reference::convert<T>(args[0]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<uint64_t>(),
                                      element_count);
                break;
            case element::Type_t::undefined:
            case element::Type_t::dynamic:
            case element::Type_t::u1:
            case element::Type_t::bf16:
            case element::Type_t::f16:
                ss << "unsupported element type " << type << " op Convert";
                throw std::runtime_error(ss.str());
            }
            break;
        }
        case OP_TYPEID::Convolution:
        {
            const op::v0::Convolution* c = static_cast<const op::v0::Convolution*>(&node);
            reference::convolution<T>(args[0]->get_data_ptr<const T>(),
                                      args[1]->get_data_ptr<const T>(),
                                      out[0]->get_data_ptr<T>(),
                                      node.get_input_shape(0),
                                      node.get_input_shape(1),
                                      node.get_output_shape(0),
                                      c->get_window_movement_strides(),
                                      c->get_window_dilation_strides(),
                                      c->get_padding_below(),
                                      c->get_padding_above(),
                                      c->get_data_dilation_strides());

            break;
        }
        case OP_TYPEID::ConvolutionBackpropData:
        {
            // Note that args[1] and args[0] are switched here from the usual order.
            const op::v0::ConvolutionBackpropData* c =
                static_cast<const op::v0::ConvolutionBackpropData*>(&node);
            reference::convolution_backprop_in<T>(args[1]->get_data_ptr<const T>(),
                                                  args[0]->get_data_ptr<const T>(),
                                                  out[0]->get_data_ptr<T>(),
                                                  c->get_input_shape(1),
                                                  c->get_input_shape(0),
                                                  c->get_data_batch_shape(),
                                                  c->get_data_dilation_strides_forward(),
                                                  c->get_window_dilation_strides_forward(),
                                                  c->compute_backward_delta_out_pad_below(),
                                                  c->compute_backward_delta_out_pad_above(),
                                                  c->get_window_movement_strides_forward());
            break;
        }
        case OP_TYPEID::Cos:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::cos<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Cosh:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::cosh<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::CumSum:
        {
            const op::CumSum* cumsum = static_cast<const op::CumSum*>(&node);
            auto axis_et = node.get_input_element_type(1);
            if (axis_et == element::i32)
            {
                reference::cumsum<T, int32_t>(args[0]->get_data_ptr<const T>(),
                                              args[1]->get_data_ptr<const int32_t>(),
                                              out[0]->get_data_ptr<T>(),
                                              node.get_input_shape(0),
                                              cumsum->is_exclusive(),
                                              cumsum->is_reverse());
            }
            else if (axis_et == element::i64)
            {
                reference::cumsum<T, int64_t>(args[0]->get_data_ptr<const T>(),
                                              args[1]->get_data_ptr<const int64_t>(),
                                              out[0]->get_data_ptr<T>(),
                                              node.get_input_shape(0),
                                              cumsum->is_exclusive(),
                                              cumsum->is_reverse());
            }
            break;
        }
        case OP_TYPEID::Dequantize:
        {
            const op::Dequantize* dequantize = static_cast<const op::Dequantize*>(&node);
            auto type = dequantize->get_element_type();

            if (type == element::f32)
            {
                reference::dequantize<T>(args[0]->get_data_ptr<const T>(),
                                         args[1]->get_data_ptr<const float>(),
                                         args[2]->get_data_ptr<const T>(),
                                         out[0]->get_data_ptr<float>(),
                                         node.get_input_shape(0),
                                         node.get_input_shape(1),
                                         dequantize->get_axes());
            }
            else if (type == element::f64)
            {
                reference::dequantize<T>(args[0]->get_data_ptr<const T>(),
                                         args[1]->get_data_ptr<const double>(),
                                         args[2]->get_data_ptr<const T>(),
                                         out[0]->get_data_ptr<double>(),
                                         node.get_input_shape(0),
                                         node.get_input_shape(1),
                                         dequantize->get_axes());
            }
            else
            {
                std::stringstream ss;
                ss << "unsupported element type " << type << " op Dequantize";
                throw std::runtime_error(ss.str());
            }

            break;
        }
        case OP_TYPEID::Dot:
        {
            const op::Dot* dot = static_cast<const op::Dot*>(&node);

            reference::dot(args[0]->get_data_ptr<const T>(),
                           args[1]->get_data_ptr<const T>(),
                           out[0]->get_data_ptr<T>(),
                           node.get_input_shape(0),
                           node.get_input_shape(1),
                           node.get_output_shape(0),
                           dot->get_reduction_axes_count());
            break;
        }
        case OP_TYPEID::EmbeddingBagOffsetsSum_v3:
        {
            const op::EmbeddingBagOffsetsSum* embed =
                static_cast<const op::EmbeddingBagOffsetsSum*>(&node);
            auto indicesType = embed->input(1).get_element_type();
            size_t indices_num = shape_size(embed->get_input_shape(1));

            if (indicesType == element::u64 || indicesType == element::i64)
            {
                reference::embeddingBagOffsetsSum<T, size_t>(
                    args[0]->get_data_ptr<const T>(),
                    args[1]->get_data_ptr<const size_t>(),
                    args[2]->get_data_ptr<const size_t>(),
                    args.size() > 3 ? args[3]->get_data_ptr<const size_t>() : nullptr,
                    args.size() > 4 ? args[4]->get_data_ptr<const T>() : nullptr,
                    out[0]->get_data_ptr<T>(),
                    indices_num,
                    embed->get_shape());
            }
            else if (indicesType == element::u32 || indicesType == element::i32)
            {
                reference::embeddingBagOffsetsSum<T, unsigned>(
                    args[0]->get_data_ptr<const T>(),
                    args[1]->get_data_ptr<const unsigned>(),
                    args[2]->get_data_ptr<const unsigned>(),
                    args.size() > 3 ? args[3]->get_data_ptr<const unsigned>() : nullptr,
                    args.size() > 4 ? args[4]->get_data_ptr<const T>() : nullptr,
                    out[0]->get_data_ptr<T>(),
                    indices_num,
                    embed->get_shape());
            }
            else
            {
                throw ngraph_error(std::string("Unsupported index type ") +
                                   indicesType.c_type_string() +
                                   std::string(" in EmbeddingBagOffsetsSum"));
            }
            break;
        }
        case OP_TYPEID::EmbeddingBagPackedSum_v3:
        {
            const op::EmbeddingBagPackedSum* embed =
                static_cast<const op::EmbeddingBagPackedSum*>(&node);
            auto indicesType = embed->input(1).get_element_type();

            if (indicesType == element::u64 || indicesType == element::i64)
            {
                reference::embeddingBagPackedSum<T, size_t>(
                    args[0]->get_data_ptr<const T>(),
                    args[1]->get_data_ptr<const size_t>(),
                    args.size() > 2 ? args[2]->get_data_ptr<const T>() : nullptr,
                    out[0]->get_data_ptr<T>(),
                    embed->get_input_shape(1),
                    embed->get_shape());
            }
            else if (indicesType == element::u32 || indicesType == element::i32)
            {
                reference::embeddingBagPackedSum<T, unsigned>(
                    args[0]->get_data_ptr<const T>(),
                    args[1]->get_data_ptr<const unsigned>(),
                    args.size() > 2 ? args[2]->get_data_ptr<const T>() : nullptr,
                    out[0]->get_data_ptr<T>(),
                    embed->get_input_shape(1),
                    embed->get_shape());
            }
            else
            {
                throw ngraph_error(std::string("Unsupported index type ") +
                                   indicesType.c_type_string() +
                                   std::string(" in EmbeddingBagPackedSum"));
            }
            break;
        }
        case OP_TYPEID::EmbeddingSegmentsSum_v3:
        {
            const op::EmbeddingSegmentsSum* embed =
                static_cast<const op::EmbeddingSegmentsSum*>(&node);
            auto indicesType = embed->input(1).get_element_type();
            size_t indices_num = shape_size(embed->get_input_shape(1));

            if (indicesType == element::u64 || indicesType == element::i64)
            {
                reference::embeddingSegmentsSum<T, size_t>(
                    args[0]->get_data_ptr<const T>(),
                    args[1]->get_data_ptr<const size_t>(),
                    args[2]->get_data_ptr<const size_t>(),
                    args.size() > 4 ? args[4]->get_data_ptr<const size_t>() : nullptr,
                    args.size() > 5 ? args[5]->get_data_ptr<const T>() : nullptr,
                    out[0]->get_data_ptr<T>(),
                    embed->get_input_shape(0),
                    embed->get_input_shape(1),
                    embed->get_shape());
            }
            else if (indicesType == element::u32 || indicesType == element::i32)
            {
                reference::embeddingSegmentsSum<T, unsigned>(
                    args[0]->get_data_ptr<const T>(),
                    args[1]->get_data_ptr<const unsigned>(),
                    args[2]->get_data_ptr<const unsigned>(),
                    args.size() > 4 ? args[4]->get_data_ptr<const unsigned>() : nullptr,
                    args.size() > 5 ? args[5]->get_data_ptr<const T>() : nullptr,
                    out[0]->get_data_ptr<T>(),
                    embed->get_input_shape(0),
                    embed->get_input_shape(1),
                    embed->get_shape());
            }
            else
            {
                throw ngraph_error(std::string("Unsupported index type ") +
                                   indicesType.c_type_string() +
                                   std::string(" in EmbeddingSegmentsSum"));
            }
            break;
        }
        case OP_TYPEID::Erf:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::erf<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::ExtractImagePatches_v3:
        {
            const op::ExtractImagePatches* extImgPatches =
                static_cast<const op::ExtractImagePatches*>(&node);
            reference::extractImagePatches<T, size_t>(extImgPatches,
                                                      args[0]->get_data_ptr<const T>(),
                                                      out[0]->get_data_ptr<T>(),
                                                      extImgPatches->get_input_shape(0),
                                                      extImgPatches->get_shape());
            break;
        }
        case OP_TYPEID::Exp:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::exp<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
#ifdef INTERPRETER_USE_HYBRID
        case OP_TYPEID::FunctionCall:
        {
            auto f = static_cast<const runtime::hybrid::op::FunctionCall*>(&node);
            auto backend = f->get_backend();
            auto executable = f->get_executable();

            std::vector<std::shared_ptr<Tensor>> outputs;
            std::vector<std::shared_ptr<Tensor>> inputs;
            for (const std::shared_ptr<HostTensor>& t : out)
            {
                auto backend_tensor = backend->create_tensor(
                    t->get_element_type(), t->get_shape(), t->get_data_ptr());
                outputs.push_back(backend_tensor);
            }
            for (const std::shared_ptr<HostTensor>& t : args)
            {
                auto backend_tensor = backend->create_tensor(
                    t->get_element_type(), t->get_shape(), t->get_data_ptr());
                inputs.push_back(backend_tensor);
            }
            executable->call(outputs, inputs);
            break;
        }
#endif
        case OP_TYPEID::Floor:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::floor<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::GatherND:
        {
            if (node.get_input_element_type(1) == element::i64)
            {
                reference::gather_nd<T, int64_t>(args[0]->get_data_ptr<T>(),
                                                 args[1]->get_data_ptr<int64_t>(),
                                                 out[0]->get_data_ptr<T>(),
                                                 node.get_input_shape(0),
                                                 node.get_input_shape(1),
                                                 node.get_output_shape(0));
            }
            else if (node.get_input_element_type(1) == element::i32)
            {
                reference::gather_nd<T, int32_t>(args[0]->get_data_ptr<T>(),
                                                 args[1]->get_data_ptr<int32_t>(),
                                                 out[0]->get_data_ptr<T>(),
                                                 node.get_input_shape(0),
                                                 node.get_input_shape(1),
                                                 node.get_output_shape(0));
            }
            else
            {
                throw ngraph_error("Unexpected type");
            }
            break;
        }
        case OP_TYPEID::Log:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::log<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::LRN:
        {
            const op::LRN* lrn = static_cast<const op::LRN*>(&node);
            reference::lrn<T>(args[0]->get_data_ptr<const T>(),
                              lrn->get_reduction_axes(),
                              out[0]->get_data_ptr<T>(),
                              node.get_input_shape(0),
                              lrn->get_alpha(),
                              lrn->get_beta(),
                              lrn->get_bias(),
                              lrn->get_nsize());
            break;
        }
        case OP_TYPEID::Negative:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::negate<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::LogicalNot_v1:
        case OP_TYPEID::Not:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::logical_not(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::OneHot:
        {
            const op::OneHot* oh = static_cast<const op::OneHot*>(&node);
            reference::one_hot<T>(args[0]->get_data_ptr<const T>(),
                                  out[0]->get_data_ptr<T>(),
                                  node.get_input_shape(0),
                                  node.get_output_shape(0),
                                  oh->get_one_hot_axis());
            break;
        }
        case OP_TYPEID::Parameter: break;
        case OP_TYPEID::Pad:
        {
            const op::Pad* pad = static_cast<const op::Pad*>(&node);

            reference::pad(args[0]->get_data_ptr<const T>(),
                           args[1]->get_data_ptr<const T>(),
                           out[0]->get_data_ptr<T>(),
                           node.get_input_shape(0),
                           node.get_output_shape(0),
                           pad->get_padding_below(),
                           pad->get_padding_above(),
                           pad->get_pad_mode());
            break;
        }
        case OP_TYPEID::Quantize:
        {
            const op::Quantize* quantize = static_cast<const op::Quantize*>(&node);
            auto type = quantize->get_element_type();

            if (type == element::u8)
            {
                reference::quantize<T>(args[0]->get_data_ptr<const T>(),
                                       args[1]->get_data_ptr<const T>(),
                                       args[2]->get_data_ptr<const uint8_t>(),
                                       out[0]->get_data_ptr<uint8_t>(),
                                       node.get_input_shape(0),
                                       node.get_input_shape(1),
                                       quantize->get_axes(),
                                       quantize->get_round_mode());
            }
            else if (type == element::i8)
            {
                reference::quantize<T>(args[0]->get_data_ptr<const T>(),
                                       args[1]->get_data_ptr<const T>(),
                                       args[2]->get_data_ptr<const int8_t>(),
                                       out[0]->get_data_ptr<int8_t>(),
                                       node.get_input_shape(0),
                                       node.get_input_shape(1),
                                       quantize->get_axes(),
                                       quantize->get_round_mode());
            }
            else if (type == element::i32)
            {
                reference::quantize<T>(args[0]->get_data_ptr<const T>(),
                                       args[1]->get_data_ptr<const T>(),
                                       args[2]->get_data_ptr<const int32_t>(),
                                       out[0]->get_data_ptr<int32_t>(),
                                       node.get_input_shape(0),
                                       node.get_input_shape(1),
                                       quantize->get_axes(),
                                       quantize->get_round_mode());
            }
            else
            {
                std::stringstream ss;
                ss << "unsupported element type " << type << " op Quantize";
                throw std::runtime_error(ss.str());
            }

            break;
        }

        case OP_TYPEID::QuantizedConvolution:
        {
            const op::QuantizedConvolution* qc =
                static_cast<const op::QuantizedConvolution*>(&node);

            auto input_element_type = qc->get_input_element_type(0);
            auto filter_element_type = qc->get_input_element_type(1);
            auto output_element_type = qc->get_output_element_type(0);

            if (input_element_type == element::u8 && filter_element_type == element::i8 &&
                output_element_type == element::i8)
            {
                reference::convolution<uint8_t, int8_t, int8_t, int32_t>(
                    args[0]->get_data_ptr<const uint8_t>(),
                    args[1]->get_data_ptr<const int8_t>(),
                    out[0]->get_data_ptr<int8_t>(),
                    node.get_input_shape(0),
                    node.get_input_shape(1),
                    node.get_output_shape(0),
                    qc->get_window_movement_strides(),
                    qc->get_window_dilation_strides(),
                    qc->get_padding_below(),
                    qc->get_padding_above(),
                    qc->get_data_dilation_strides(),
                    args[2]->get_data_ptr<const float>(),
                    args[3]->get_data_ptr<const uint8_t>(),
                    args[4]->get_data_ptr<const float>(),
                    args[5]->get_data_ptr<const int8_t>(),
                    args[6]->get_data_ptr<const float>(),
                    args[7]->get_data_ptr<const int8_t>());
            }
            else if (input_element_type == element::u8 && filter_element_type == element::u8 &&
                     output_element_type == element::u8)
            {
                reference::convolution<uint8_t, uint8_t, uint8_t, int32_t>(
                    args[0]->get_data_ptr<const uint8_t>(),
                    args[1]->get_data_ptr<const uint8_t>(),
                    out[0]->get_data_ptr<uint8_t>(),
                    node.get_input_shape(0),
                    node.get_input_shape(1),
                    node.get_output_shape(0),
                    qc->get_window_movement_strides(),
                    qc->get_window_dilation_strides(),
                    qc->get_padding_below(),
                    qc->get_padding_above(),
                    qc->get_data_dilation_strides(),
                    args[2]->get_data_ptr<const float>(),
                    args[3]->get_data_ptr<const uint8_t>(),
                    args[4]->get_data_ptr<const float>(),
                    args[5]->get_data_ptr<const uint8_t>(),
                    args[6]->get_data_ptr<const float>(),
                    args[7]->get_data_ptr<const uint8_t>());
            }
            else if (input_element_type == element::u8 && filter_element_type == element::i8 &&
                     output_element_type == element::i32)
            {
                reference::convolution<uint8_t, int8_t, int32_t, int32_t>(
                    args[0]->get_data_ptr<const uint8_t>(),
                    args[1]->get_data_ptr<const int8_t>(),
                    out[0]->get_data_ptr<int32_t>(),
                    node.get_input_shape(0),
                    node.get_input_shape(1),
                    node.get_output_shape(0),
                    qc->get_window_movement_strides(),
                    qc->get_window_dilation_strides(),
                    qc->get_padding_below(),
                    qc->get_padding_above(),
                    qc->get_data_dilation_strides(),
                    args[2]->get_data_ptr<const float>(),
                    args[3]->get_data_ptr<const uint8_t>(),
                    args[4]->get_data_ptr<const float>(),
                    args[5]->get_data_ptr<const int8_t>(),
                    args[6]->get_data_ptr<const float>(),
                    args[7]->get_data_ptr<const int32_t>());
            }
            else if (input_element_type == element::u8 && filter_element_type == element::u8 &&
                     output_element_type == element::i32)
            {
                reference::convolution<uint8_t, uint8_t, int32_t, int32_t>(
                    args[0]->get_data_ptr<const uint8_t>(),
                    args[1]->get_data_ptr<const uint8_t>(),
                    out[0]->get_data_ptr<int32_t>(),
                    node.get_input_shape(0),
                    node.get_input_shape(1),
                    node.get_output_shape(0),
                    qc->get_window_movement_strides(),
                    qc->get_window_dilation_strides(),
                    qc->get_padding_below(),
                    qc->get_padding_above(),
                    qc->get_data_dilation_strides(),
                    args[2]->get_data_ptr<const float>(),
                    args[3]->get_data_ptr<const uint8_t>(),
                    args[4]->get_data_ptr<const float>(),
                    args[5]->get_data_ptr<const uint8_t>(),
                    args[6]->get_data_ptr<const float>(),
                    args[7]->get_data_ptr<const int32_t>());
            }
            else
            {
                std::stringstream ss;
                ss << "unsupported element type";
                throw std::runtime_error(ss.str());
            }

            break;
        }

        case OP_TYPEID::QuantizedDot:
        {
            const op::QuantizedDot* qd = static_cast<const op::QuantizedDot*>(&node);

            auto input0_element_type = qd->get_input_element_type(0);
            auto input1_element_type = qd->get_input_element_type(1);
            auto output_element_type = qd->get_output_element_type(0);

            if (input0_element_type == element::u8 && input1_element_type == element::i8 &&
                output_element_type == element::i8)
            {
                reference::dot<uint8_t, int8_t, int8_t, int32_t>(
                    args[0]->get_data_ptr<const uint8_t>(),
                    args[1]->get_data_ptr<const int8_t>(),
                    out[0]->get_data_ptr<int8_t>(),
                    node.get_input_shape(0),
                    node.get_input_shape(1),
                    node.get_output_shape(0),
                    1,
                    args[2]->get_data_ptr<const float>(),
                    args[3]->get_data_ptr<const uint8_t>(),
                    args[4]->get_data_ptr<const float>(),
                    args[5]->get_data_ptr<const int8_t>(),
                    args[6]->get_data_ptr<const float>(),
                    args[7]->get_data_ptr<const int8_t>());
            }
            else if (input0_element_type == element::u8 && input1_element_type == element::u8 &&
                     output_element_type == element::u8)
            {
                reference::dot<uint8_t, uint8_t, uint8_t, int32_t>(
                    args[0]->get_data_ptr<const uint8_t>(),
                    args[1]->get_data_ptr<const uint8_t>(),
                    out[0]->get_data_ptr<uint8_t>(),
                    node.get_input_shape(0),
                    node.get_input_shape(1),
                    node.get_output_shape(0),
                    1,
                    args[2]->get_data_ptr<const float>(),
                    args[3]->get_data_ptr<const uint8_t>(),
                    args[4]->get_data_ptr<const float>(),
                    args[5]->get_data_ptr<const uint8_t>(),
                    args[6]->get_data_ptr<const float>(),
                    args[7]->get_data_ptr<const uint8_t>());
            }
            else if (input0_element_type == element::u8 && input1_element_type == element::u8 &&
                     output_element_type == element::i32)
            {
                reference::dot<uint8_t, uint8_t, int32_t, int32_t>(
                    args[0]->get_data_ptr<const uint8_t>(),
                    args[1]->get_data_ptr<const uint8_t>(),
                    out[0]->get_data_ptr<int32_t>(),
                    node.get_input_shape(0),
                    node.get_input_shape(1),
                    node.get_output_shape(0),
                    1,
                    args[2]->get_data_ptr<const float>(),
                    args[3]->get_data_ptr<const uint8_t>(),
                    args[4]->get_data_ptr<const float>(),
                    args[5]->get_data_ptr<const uint8_t>(),
                    args[6]->get_data_ptr<const float>(),
                    args[7]->get_data_ptr<const int32_t>());
            }
            else if (input0_element_type == element::u8 && input1_element_type == element::i8 &&
                     output_element_type == element::i32)
            {
                reference::dot<uint8_t, int8_t, int32_t, int32_t>(
                    args[0]->get_data_ptr<const uint8_t>(),
                    args[1]->get_data_ptr<const int8_t>(),
                    out[0]->get_data_ptr<int32_t>(),
                    node.get_input_shape(0),
                    node.get_input_shape(1),
                    node.get_output_shape(0),
                    1,
                    args[2]->get_data_ptr<const float>(),
                    args[3]->get_data_ptr<const uint8_t>(),
                    args[4]->get_data_ptr<const float>(),
                    args[5]->get_data_ptr<const int8_t>(),
                    args[6]->get_data_ptr<const float>(),
                    args[7]->get_data_ptr<const int32_t>());
            }
            else
            {
                std::stringstream ss;
                ss << "unsupported element type";
                throw std::runtime_error(ss.str());
            }

            break;
        }
        case OP_TYPEID::Relu:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::relu<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::ReplaceSlice:
        {
            const op::ReplaceSlice* slice = static_cast<const op::ReplaceSlice*>(&node);
            reference::replace_slice<T>(args[0]->get_data_ptr<const T>(),
                                        args[1]->get_data_ptr<const T>(),
                                        out[0]->get_data_ptr<T>(),
                                        node.get_input_shape(1),
                                        slice->get_lower_bounds(),
                                        slice->get_upper_bounds(),
                                        slice->get_strides(),
                                        node.get_output_shape(0));
            break;
        }
        case OP_TYPEID::Reverse:
        {
            const op::Reverse* reverse = static_cast<const op::Reverse*>(&node);
            reference::reverse(args[0]->get_data_ptr<const char>(),
                               out[0]->get_data_ptr<char>(),
                               node.get_input_shape(0),
                               node.get_output_shape(0),
                               reverse->get_reversed_axes(),
                               args[0]->get_element_type().size());
            break;
        }
        case OP_TYPEID::ReverseSequence:
        {
            const op::ReverseSequence* reverse = static_cast<const op::ReverseSequence*>(&node);

            if (node.get_input_element_type(1) == element::i32)
            {
                reference::reverse_sequence<T, int32_t>(args[0]->get_data_ptr<const T>(),
                                                        out[0]->get_data_ptr<T>(),
                                                        node.get_input_shape(0),
                                                        reverse->get_batch_axis(),
                                                        reverse->get_sequence_axis(),
                                                        args[1]->get_data_ptr<const int32_t>());
            }
            else
            {
                throw ngraph_error("only int32 indices are supported");
            }
            break;
        }
        case OP_TYPEID::Round:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::round<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Select:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::select<T>(args[0]->get_data_ptr<const char>(),
                                 args[1]->get_data_ptr<const T>(),
                                 args[2]->get_data_ptr<const T>(),
                                 out[0]->get_data_ptr<T>(),
                                 element_count);
            break;
        }
        case OP_TYPEID::Sigmoid:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sigmoid<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Sign:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sign<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Sin:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sin<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Sinh:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sinh<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Sqrt:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::sqrt<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Tan:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::tan<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::Tanh:
        {
            size_t element_count = shape_size(node.get_output_shape(0));
            reference::tanh<T>(
                args[0]->get_data_ptr<const T>(), out[0]->get_data_ptr<T>(), element_count);
            break;
        }
        case OP_TYPEID::TopK:
        {
            const op::TopK* topk = static_cast<const op::TopK*>(&node);
            if (node.get_output_element_type(0) == element::i64)
            {
                reference::topk<T, int64_t>(args[0]->get_data_ptr<const T>(),
                                            out[0]->get_data_ptr<int64_t>(),
                                            out[1]->get_data_ptr<T>(),
                                            node.get_input_shape(0),
                                            node.get_output_shape(0),
                                            topk->get_top_k_axis(),
                                            topk->get_k(),
                                            topk->get_compute_max(),
                                            topk->get_sort());
            }
            else if (node.get_output_element_type(0) == element::i32)
            {
                reference::topk<T, int32_t>(args[0]->get_data_ptr<const T>(),
                                            out[0]->get_data_ptr<int32_t>(),
                                            out[1]->get_data_ptr<T>(),
                                            node.get_input_shape(0),
                                            node.get_output_shape(0),
                                            topk->get_top_k_axis(),
                                            topk->get_k(),
                                            topk->get_compute_max(),
                                            topk->get_sort());
            }
            else
            {
                throw ngraph_error("Unexpected type");
            }
            break;
        }
        case OP_TYPEID::DetectionOutput_v0:
        {
            const op::DetectionOutput* detOut = static_cast<const op::DetectionOutput*>(&node);
            reference::referenceDetectionOutput<T> refDetOut(
                detOut->get_attrs(), node.get_input_shape(0), node.get_input_shape(2));
            if (node.get_input_size() == 3)
            {
                refDetOut.run(args[0]->get_data_ptr<const T>(),
                              args[1]->get_data_ptr<const T>(),
                              args[2]->get_data_ptr<const T>(),
                              nullptr,
                              nullptr,
                              out[0]->get_data_ptr<T>());
            }
            else if (node.get_input_size() == 5)
            {
                refDetOut.run(args[0]->get_data_ptr<const T>(),
                              args[1]->get_data_ptr<const T>(),
                              args[2]->get_data_ptr<const T>(),
                              args[3]->get_data_ptr<const T>(),
                              args[4]->get_data_ptr<const T>(),
                              out[0]->get_data_ptr<T>());
            }
            else
            {
                throw ngraph_error("DetectionOutput layer supports only 3 or 5 inputs");
            }

            break;
        }
        case OP_TYPEID::ScatterNDUpdate_v3:
        {
            const op::ScatterNDUpdate* scatterNDUpd =
                static_cast<const op::v3::ScatterNDUpdate*>(&node);
            auto idxType = scatterNDUpd->get_input_element_type(1);
            if (idxType == element::i32)
            {
                reference::scatterNdUpdate<T, int32_t>(args[0]->get_data_ptr<const T>(),
                                                       args[1]->get_data_ptr<const int32_t>(),
                                                       args[2]->get_data_ptr<const T>(),
                                                       out[0]->get_data_ptr<T>(),
                                                       node.get_input_shape(0),
                                                       node.get_input_shape(1),
                                                       node.get_input_shape(2));
            }
            else if (idxType == element::i64)
            {
                reference::scatterNdUpdate<T, int64_t>(args[0]->get_data_ptr<const T>(),
                                                       args[1]->get_data_ptr<const int64_t>(),
                                                       args[2]->get_data_ptr<const T>(),
                                                       out[0]->get_data_ptr<T>(),
                                                       node.get_input_shape(0),
                                                       node.get_input_shape(1),
                                                       node.get_input_shape(2));
            }
            else
            {
                throw ngraph_error(
                    "ScatterNDUpdate layer support only i32 and i64 'indices' input precision!");
            }

            break;
        }
        case OP_TYPEID::ScatterUpdate_v3:
        {
            const op::v3::ScatterUpdate* scatterUpd =
                static_cast<const op::v3::ScatterUpdate*>(&node);

            if (scatterUpd->get_input_element_type(3) != element::i64)
                throw ngraph_error(
                    "ScatterNDUpdate layer support only i64 'axis' input precision!");

            auto idxType = scatterUpd->get_input_element_type(1);
            if (idxType == element::i32)
            {
                reference::scatterUpdate<T, int32_t, int64_t>(
                    args[0]->get_data_ptr<const T>(),
                    args[1]->get_data_ptr<const int32_t>(),
                    args[2]->get_data_ptr<const T>(),
                    args[3]->get_data_ptr<const int64_t>(),
                    out[0]->get_data_ptr<T>(),
                    node.get_input_shape(0),
                    node.get_input_shape(1),
                    node.get_input_shape(2));
            }
            else if (idxType == element::i64)
            {
                reference::scatterUpdate<T, int64_t, int64_t>(
                    args[0]->get_data_ptr<const T>(),
                    args[1]->get_data_ptr<const int64_t>(),
                    args[2]->get_data_ptr<const T>(),
                    args[3]->get_data_ptr<const int64_t>(),
                    out[0]->get_data_ptr<T>(),
                    node.get_input_shape(0),
                    node.get_input_shape(1),
                    node.get_input_shape(2));
            }
            else
            {
                throw ngraph_error(
                    "ScatterUpdate layer support only i32 and i64 'indices' input precision!");
            }

            break;
        }

        // Fused Ops are not supported in interpreter. They need to be decomposed before execution
        case OP_TYPEID::DepthToSpace:
        case OP_TYPEID::FakeQuantize:
        case OP_TYPEID::Gather:
        case OP_TYPEID::Gelu:
        case OP_TYPEID::GRN:
        case OP_TYPEID::GroupConvolution:
        case OP_TYPEID::GroupConvolutionBackpropData:
        case OP_TYPEID::GRUCell:
        case OP_TYPEID::HardSigmoid:
        case OP_TYPEID::Interpolate:
        case OP_TYPEID::LSTMCell:
        case OP_TYPEID::LSTMSequence:
        case OP_TYPEID::MVN:
        case OP_TYPEID::NormalizeL2:
        case OP_TYPEID::Passthrough:
        case OP_TYPEID::PRelu:
        case OP_TYPEID::RNNCell:
        case OP_TYPEID::Selu:
        case OP_TYPEID::ShuffleChannels:
        case OP_TYPEID::SpaceToDepth:
        case OP_TYPEID::Split:
        case OP_TYPEID::SquaredDifference:
        case OP_TYPEID::StopGradient:
        case OP_TYPEID::TensorIterator:
        case OP_TYPEID::Tile:
        case OP_TYPEID::UnknownOp:
            throw unsupported_op("Unsupported op '" + node.description() + "'");
        case OP_TYPEID::Add:
        case OP_TYPEID::Broadcast:
        case OP_TYPEID::Clamp:
        case OP_TYPEID::Concat:
        case OP_TYPEID::Constant:
        case OP_TYPEID::Divide:
        case OP_TYPEID::Equal:
        case OP_TYPEID::Greater:
        case OP_TYPEID::GreaterEq:
        case OP_TYPEID::Less:
        case OP_TYPEID::LessEq:
        case OP_TYPEID::LessEqual_v1:
        case OP_TYPEID::LogicalAnd_v1:
        case OP_TYPEID::LogicalOr_v1:
        case OP_TYPEID::LogicalXor_v1:
        case OP_TYPEID::MatMul:
        case OP_TYPEID::Max:
        case OP_TYPEID::Maximum:
        case OP_TYPEID::Min:
        case OP_TYPEID::Minimum:
        case OP_TYPEID::Multiply:
        case OP_TYPEID::NonZero_v3:
        case OP_TYPEID::NotEqual:
        case OP_TYPEID::Or:
        case OP_TYPEID::Power:
        case OP_TYPEID::Product:
        case OP_TYPEID::Range:
        case OP_TYPEID::Reshape:
        case OP_TYPEID::Result:
        case OP_TYPEID::ShapeOf_v3:
        case OP_TYPEID::ShapeOf:
        case OP_TYPEID::Softmax:
        case OP_TYPEID::Squeeze:
        case OP_TYPEID::Sum:
        case OP_TYPEID::Subtract:
        case OP_TYPEID::Unsqueeze:
        case OP_TYPEID::Xor:
        case OP_TYPEID::Slice:
            // These ops are handled by op evaluators so nothing to do
            break;
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
        }
    }
};
