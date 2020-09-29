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

#include "evaluates_map.hpp"
#include <interpreter/reference/mod.hpp>
#include <ngraph/runtime/reference/any.hpp>
#include <ngraph/runtime/reference/batch_norm.hpp>
#include <ngraph/runtime/reference/ceiling.hpp>
#include <ngraph/runtime/reference/convert.hpp>
#include <ngraph/runtime/reference/dequantize.hpp>
#include <ngraph/runtime/reference/dot.hpp>
#include <ngraph/runtime/reference/extract_image_patches.hpp>
#include <ngraph/runtime/reference/gather_nd.hpp>
#include <ngraph/runtime/reference/gru_cell.hpp>
#include <ngraph/runtime/reference/lstm_cell.hpp>
#include <ngraph/runtime/reference/one_hot.hpp>
#include <ngraph/runtime/reference/pad.hpp>
#include <ngraph/runtime/reference/prior_box.hpp>
#include <ngraph/runtime/reference/quantize.hpp>
#include <ngraph/runtime/reference/replace_slice.hpp>
#include <ngraph/runtime/reference/reverse_sequence.hpp>
#include <ngraph/runtime/reference/rnn_cell.hpp>
#include <ngraph/runtime/reference/select.hpp>
#include <ngraph/runtime/reference/sequences.hpp>
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/avg_pool.hpp"
#include "ngraph/runtime/reference/batch_norm.hpp"
#include "ngraph/runtime/reference/batch_norm.hpp"
#include "ngraph/runtime/reference/convolution.hpp"
#include "ngraph/runtime/reference/ctc_loss.hpp"
#include "ngraph/runtime/reference/cum_sum.hpp"
#include "ngraph/runtime/reference/detection_output.hpp"
#include "ngraph/runtime/reference/embedding_bag_offsets_sum.hpp"
#include "ngraph/runtime/reference/embedding_bag_packed_sum.hpp"
#include "ngraph/runtime/reference/embedding_segments_sum.hpp"
#include "ngraph/runtime/reference/fake_quantize.hpp"
#include "ngraph/runtime/reference/gather_tree.hpp"
#include "ngraph/runtime/reference/lrn.hpp"
#include "ngraph/runtime/reference/mvn.hpp"
#include "ngraph/runtime/reference/normalize_l2.hpp"
#include "ngraph/runtime/reference/reverse_sequence.hpp"
#include "ngraph/runtime/reference/scatter_nd_update.hpp"
#include "reference/elu.hpp"
#include "reference/gelu.hpp"
#include "reference/hard_sigmoid.hpp"
#include "reference/selu.hpp"

using namespace ngraph;
using namespace std;

namespace
{
    template <element::Type_t ET>
    bool evaluate(shared_ptr<Node> op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        return false;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::Convolution>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto filter_data = inputs[1]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto& out_shape = outputs[0]->get_shape();
        const auto& in_shape = inputs[0]->get_shape();
        const auto& filter_shape = inputs[1]->get_shape();
        Strides in_dilation(std::vector<size_t>(in_shape.size() - 2));
        std::fill(in_dilation.begin(), in_dilation.end(), 1);
        runtime::reference::convolution<typename element_type_traits<ET>::value_type>(
            in_data_ptr,
            filter_data,
            out_data_ptr,
            in_shape,
            filter_shape,
            out_shape,
            op->get_strides(),
            op->get_dilations(),
            op->get_pads_begin(),
            op->get_pads_end(),
            in_dilation);
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::ConvolutionBackpropData>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto filter_data = inputs[1]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto& out_shape = outputs[0]->get_shape();
        const auto& in_shape = inputs[0]->get_shape();
        const auto& filter_shape = inputs[1]->get_shape();
        Strides in_dilation(std::vector<size_t>(in_shape.size() - 2));
        std::fill(in_dilation.begin(), in_dilation.end(), 1);
        runtime::reference::convolution_backprop_in<typename element_type_traits<ET>::value_type>(
            in_data_ptr,
            filter_data,
            out_data_ptr,
            in_shape,
            filter_shape,
            out_shape,
            in_dilation,
            op->get_dilations(),
            op->get_pads_begin(),
            op->get_pads_end(),
            op->get_strides());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::GroupConvolution>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto filter_data = inputs[1]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto& out_shape = outputs[0]->get_shape();
        const auto& in_shape = inputs[0]->get_shape();
        const auto& filter_shape = inputs[1]->get_shape();
        Strides in_dilation(std::vector<size_t>(in_shape.size() - 2));
        std::fill(in_dilation.begin(), in_dilation.end(), 1);
        runtime::reference::convolution<typename element_type_traits<ET>::value_type>(
            in_data_ptr,
            filter_data,
            out_data_ptr,
            in_shape,
            filter_shape,
            out_shape,
            op->get_strides(),
            op->get_dilations(),
            op->get_pads_begin(),
            op->get_pads_end(),
            in_dilation,
            filter_shape.at(0));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::GroupConvolutionBackpropData>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        const auto filter_data = inputs[1]->get_data_ptr<ET>();
        auto out_data_ptr = outputs[0]->get_data_ptr<ET>();
        const auto in_data_ptr = inputs[0]->get_data_ptr<ET>();
        const auto& out_shape = outputs[0]->get_shape();
        const auto& in_shape = inputs[0]->get_shape();
        const auto& filter_shape = inputs[1]->get_shape();
        Strides in_dilation(std::vector<size_t>(in_shape.size() - 2));
        std::fill(in_dilation.begin(), in_dilation.end(), 1);
        runtime::reference::convolution_backprop_in<typename element_type_traits<ET>::value_type>(
            in_data_ptr,
            filter_data,
            out_data_ptr,
            in_shape,
            filter_shape,
            out_shape,
            in_dilation,
            op->get_dilations(),
            op->get_pads_begin(),
            op->get_pads_end(),
            op->get_strides(),
            filter_shape.at(0));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::CumSum>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;

#define REF_CALL(U)                                                                                \
    runtime::reference::cumsum<T, typename element_type_traits<U>::value_type>(                    \
        inputs[0]->get_data_ptr<ET>(),                                                             \
        inputs[1]->get_data_ptr<U>(),                                                              \
        outputs[0]->get_data_ptr<ET>(),                                                            \
        inputs[0]->get_shape(),                                                                    \
        op->is_exclusive(),                                                                        \
        op->is_reverse());                                                                         \
    break;

        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::i64: { REF_CALL(element::Type_t::i64);
        }
        default: REF_CALL(element::Type_t::i32);
        }
#undef REF_CALL
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::EmbeddingSegmentsSum>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
#define REF_CALL(elType)                                                                           \
    runtime::reference::embeddingSegmentsSum<T, typename element_type_traits<elType>::value_type>( \
        inputs[0]->get_data_ptr<ET>(),                                                             \
        inputs[1]->get_data_ptr<elType>(),                                                         \
        inputs[2]->get_data_ptr<elType>(),                                                         \
        inputs.size() > 4 ? inputs[4]->get_data_ptr<elType>() : nullptr,                           \
        inputs.size() > 5 ? inputs[5]->get_data_ptr<ET>() : nullptr,                               \
        outputs[0]->get_data_ptr<ET>(),                                                            \
        inputs[0]->get_shape(),                                                                    \
        inputs[1]->get_shape(),                                                                    \
        outputs[0]->get_shape());                                                                  \
    break;

        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::i32: REF_CALL(element::Type_t::i32);
        case element::Type_t::i64: REF_CALL(element::Type_t::i64);
        default: return false;
        }
#undef REF_CALL
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::EmbeddingBagOffsetsSum>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
#define REF_CALL(elType)                                                                           \
    runtime::reference::embeddingBagOffsetsSum<T,                                                  \
                                               typename element_type_traits<elType>::value_type>(  \
        inputs[0]->get_data_ptr<ET>(),                                                             \
        inputs[1]->get_data_ptr<elType>(),                                                         \
        inputs[2]->get_data_ptr<elType>(),                                                         \
        inputs.size() > 3 ? inputs[3]->get_data_ptr<elType>() : nullptr,                           \
        inputs.size() > 4 ? inputs[4]->get_data_ptr<ET>() : nullptr,                               \
        outputs[0]->get_data_ptr<ET>(),                                                            \
        shape_size(inputs[1]->get_shape()),                                                        \
        outputs[0]->get_shape());                                                                  \
    break;

        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::i32: REF_CALL(element::Type_t::i32);
        case element::Type_t::i64: REF_CALL(element::Type_t::i64);
        default: return false;
        }
#undef REF_CALL
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::EmbeddingBagPackedSum>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
#define REF_CALL(elType)                                                                           \
    runtime::reference::embeddingBagPackedSum<T,                                                   \
                                              typename element_type_traits<elType>::value_type>(   \
        inputs[0]->get_data_ptr<ET>(),                                                             \
        inputs[1]->get_data_ptr<elType>(),                                                         \
        inputs.size() > 2 ? inputs[2]->get_data_ptr<ET>() : nullptr,                               \
        outputs[0]->get_data_ptr<ET>(),                                                            \
        inputs[1]->get_shape(),                                                                    \
        outputs[0]->get_shape());                                                                  \
    break;

        switch (inputs[1]->get_element_type())
        {
        case element::Type_t::i32: REF_CALL(element::Type_t::i32);
        case element::Type_t::i64: REF_CALL(element::Type_t::i64);
        default: return false;
        }
#undef REF_CALL
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::MVN>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::mvn<T>(inputs[0]->get_data_ptr<ET>(),
                                   outputs[0]->get_data_ptr<ET>(),
                                   inputs[0]->get_shape(),
                                   op->get_normalize_variance(),
                                   op->get_reduction_axes(),
                                   op->get_eps());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::LRN>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::lrn<T>(inputs[0]->get_data_ptr<ET>(),
                                   op->get_reduction_axes(),
                                   outputs[0]->get_data_ptr<ET>(),
                                   inputs[0]->get_shape(),
                                   op->get_alpha(),
                                   op->get_beta(),
                                   op->get_bias(),
                                   op->get_nsize());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::DetectionOutput>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::referenceDetectionOutput<T> refDetOut(
            op->get_attrs(), op->get_input_shape(0), op->get_input_shape(2));
        if (op->get_input_size() == 3)
        {
            refDetOut.run(input[0]->get_data_ptr<const T>(),
                          input[1]->get_data_ptr<const T>(),
                          input[2]->get_data_ptr<const T>(),
                          nullptr,
                          nullptr,
                          outputs[0]->get_data_ptr<T>());
        }
        else if (op->get_input_size() == 5)
        {
            refDetOut.run(input[0]->get_data_ptr<const T>(),
                          input[1]->get_data_ptr<const T>(),
                          input[2]->get_data_ptr<const T>(),
                          input[3]->get_data_ptr<const T>(),
                          input[4]->get_data_ptr<const T>(),
                          outputs[0]->get_data_ptr<T>());
        }
        else
        {
            throw ngraph_error("DetectionOutput layer supports only 3 or 5 inputs");
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::ScatterNDUpdate>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using T = typename element_type_traits<ET>::value_type;
        auto idxType = op->get_input_element_type(1);
        if (idxType == element::i32)
        {
            runtime::reference::scatterNdUpdate<T, int32_t>(input[0]->get_data_ptr<const T>(),
                                                            input[1]->get_data_ptr<const int32_t>(),
                                                            input[2]->get_data_ptr<const T>(),
                                                            outputs[0]->get_data_ptr<T>(),
                                                            op->get_input_shape(0),
                                                            op->get_input_shape(1),
                                                            op->get_input_shape(2));
        }
        else if (idxType == element::i64)
        {
            runtime::reference::scatterNdUpdate<T, int64_t>(input[0]->get_data_ptr<const T>(),
                                                            input[1]->get_data_ptr<const int64_t>(),
                                                            input[2]->get_data_ptr<const T>(),
                                                            outputs[0]->get_data_ptr<T>(),
                                                            op->get_input_shape(0),
                                                            op->get_input_shape(1),
                                                            op->get_input_shape(2));
        }
        else
        {
            throw ngraph_error(
                "ScatterNDUpdate layer support only i32 and i64 'indices' input precision!");
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::Select>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using T = typename element_type_traits<ET>::value_type;

        runtime::reference::select<T>(input[0]->get_data_ptr<const char>(),
                                      input[1]->get_data_ptr<const T>(),
                                      input[2]->get_data_ptr<const T>(),
                                      outputs[0]->get_data_ptr<T>(),
                                      op->get_input_shape(0),
                                      op->get_input_shape(1),
                                      op->get_input_shape(2),
                                      op->get_auto_broadcast());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::AvgPool>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::avg_pool<T>(input[0]->get_data_ptr<T>(),
                                        outputs[0]->get_data_ptr<T>(),
                                        input[0]->get_shape(),
                                        op->get_output_shape(0),
                                        op->get_kernel(),
                                        op->get_strides(),
                                        op->get_pads_begin(),
                                        op->get_pads_end(),
                                        !op->get_exclude_pad());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::HardSigmoid>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::hard_sigmoid<T>(input[0]->get_data_ptr<T>(),
                                            input[1]->get_data_ptr<T>(),
                                            input[2]->get_data_ptr<T>(),
                                            outputs[0]->get_data_ptr<T>(),
                                            shape_size(input[0]->get_shape()),
                                            shape_size(input[1]->get_shape()),
                                            shape_size(input[2]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Elu>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::elu<T>(input[0]->get_data_ptr<T>(),
                                   outputs[0]->get_data_ptr<T>(),
                                   shape_size(input[0]->get_shape()),
                                   op->get_alpha());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::PriorBox>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::prior_box<T>(input[0]->get_data_ptr<T>(),
                                         input[1]->get_data_ptr<T>(),
                                         outputs[0]->get_data_ptr<float>(),
                                         outputs[0]->get_shape(),
                                         op->get_attrs());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::Mod>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::mod<T>(input[0]->get_data_ptr<T>(),
                                   input[1]->get_data_ptr<T>(),
                                   outputs[0]->get_data_ptr<T>(),
                                   input[0]->get_shape(),
                                   op->get_auto_broadcast());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Selu>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::selu<T>(input[0]->get_data_ptr<T>(),
                                    input[1]->get_data_ptr<T>(),
                                    input[2]->get_data_ptr<T>(),
                                    outputs[0]->get_data_ptr<T>(),
                                    shape_size(input[0]->get_shape()),
                                    shape_size(input[1]->get_shape()),
                                    shape_size(input[2]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Ceiling>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::ceiling<T>(input[0]->get_data_ptr<T>(),
                                       outputs[0]->get_data_ptr<T>(),
                                       shape_size(input[0]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::Gelu>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::gelu<T>(input[0]->get_data_ptr<T>(),
                                    outputs[0]->get_data_ptr<T>(),
                                    shape_size(input[0]->get_shape()));
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v4::CTCLoss>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using T = typename element_type_traits<ET>::value_type;
#define REF_CALL(elType)                                                                           \
    runtime::reference::CTCLoss<T, typename element_type_traits<elType>::value_type>(              \
        input[0]->get_data_ptr<T>(),                                                               \
        input[0]->get_shape(),                                                                     \
        input[1]->get_data_ptr<elType>(),                                                          \
        input[2]->get_data_ptr<elType>(),                                                          \
        input[3]->get_data_ptr<elType>(),                                                          \
        input[4]->get_data_ptr<elType>(),                                                          \
        op->get_preprocess_collapse_repeated(),                                                    \
        op->get_ctc_merge_repeated(),                                                              \
        op->get_unique(),                                                                          \
        outputs[0]->get_data_ptr<T>());                                                            \
    break;

        switch (input[1]->get_element_type())
        {
        case element::Type_t::i32: REF_CALL(element::Type_t::i32);
        case element::Type_t::i64: REF_CALL(element::Type_t::i64);
        default: return false;
        }
#undef REF_CALL
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::BatchNormInference>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::batch_norm_inference<T>(op->get_eps_value(),
                                                    input[0]->get_data_ptr<T>(),
                                                    input[1]->get_data_ptr<T>(),
                                                    input[2]->get_data_ptr<T>(),
                                                    input[3]->get_data_ptr<T>(),
                                                    input[4]->get_data_ptr<T>(),
                                                    outputs[0]->get_data_ptr<T>(),
                                                    input[2]->get_shape());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::ReverseSequence>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using T = typename element_type_traits<ET>::value_type;

#define REF_CALL(U)                                                                                \
    runtime::reference::reverse_sequence<T, typename element_type_traits<U>::value_type>(          \
        input[0]->get_data_ptr<T>(),                                                               \
        outputs[0]->get_data_ptr<T>(),                                                             \
        input[0]->get_shape(),                                                                     \
        op->get_batch_axis(),                                                                      \
        op->get_sequence_axis(),                                                                   \
        input[1]->get_data_ptr<U>());                                                              \
    break;

        switch (input[1]->get_element_type())
        {
        case element::Type_t::boolean: REF_CALL(element::Type_t::boolean)
        case element::Type_t::i8: REF_CALL(element::Type_t::i8);
        case element::Type_t::i16: REF_CALL(element::Type_t::i16);
        case element::Type_t::i32: REF_CALL(element::Type_t::i32);
        case element::Type_t::i64: REF_CALL(element::Type_t::i64);
        case element::Type_t::u8: REF_CALL(element::Type_t::u8);
        case element::Type_t::u16: REF_CALL(element::Type_t::u16);
        case element::Type_t::u32: REF_CALL(element::Type_t::u32);
        case element::Type_t::u64: REF_CALL(element::Type_t::u64);
        case element::Type_t::f16: REF_CALL(element::Type_t::f16);
        case element::Type_t::f32: REF_CALL(element::Type_t::f32);
        case element::Type_t::f64: REF_CALL(element::Type_t::f64);
        default: return false;
        }
#undef REF_CALL
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::ExtractImagePatches>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::extract_image_patches<T>(op,
                                                     input[0]->get_data_ptr<T>(),
                                                     outputs[0]->get_data_ptr<T>(),
                                                     input[0]->get_shape(),
                                                     outputs[0]->get_shape());
        return true;
    }

    template <element::Type_t OUT_ET>
    bool evaluate(const shared_ptr<op::v0::Convert>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& input)
    {
        using TO = typename element_type_traits<OUT_ET>::value_type;
        if (OUT_ET == element::Type_t::boolean)
        {
#define REF_CALL_BOOL(TI)                                                                          \
    runtime::reference::convert_to_bool<typename element_type_traits<TI>::value_type>(             \
        input[0]->get_data_ptr<TI>(),                                                              \
        outputs[0]->get_data_ptr<char>(),                                                          \
        shape_size(input[0]->get_shape()));                                                        \
    break;
            switch (input[0]->get_element_type())
            {
            case element::Type_t::boolean: REF_CALL_BOOL(element::Type_t::boolean);
            case element::Type_t::i8: REF_CALL_BOOL(element::Type_t::i8);
            case element::Type_t::i16: REF_CALL_BOOL(element::Type_t::i16);
            case element::Type_t::i32: REF_CALL_BOOL(element::Type_t::i32);
            case element::Type_t::i64: REF_CALL_BOOL(element::Type_t::i64);
            case element::Type_t::u8: REF_CALL_BOOL(element::Type_t::u8);
            case element::Type_t::u16: REF_CALL_BOOL(element::Type_t::u16);
            case element::Type_t::u32: REF_CALL_BOOL(element::Type_t::u32);
            case element::Type_t::u64: REF_CALL_BOOL(element::Type_t::u64);
            case element::Type_t::f16: REF_CALL_BOOL(element::Type_t::f16);
            case element::Type_t::f32: REF_CALL_BOOL(element::Type_t::f32);
            case element::Type_t::f64: REF_CALL_BOOL(element::Type_t::f64);
            default: return false;
            }
#undef REF_CALL_BOOL
        }
        else
        {
#define REF_CALL(TI)                                                                               \
    runtime::reference::convert<typename element_type_traits<TI>::value_type, TO>(                 \
        input[0]->get_data_ptr<TI>(),                                                              \
        outputs[0]->get_data_ptr<TO>(),                                                            \
        shape_size(input[0]->get_shape()));                                                        \
    break;

            switch (input[0]->get_element_type())
            {
            case element::Type_t::boolean: REF_CALL(element::Type_t::boolean);
            case element::Type_t::i8: REF_CALL(element::Type_t::i8);
            case element::Type_t::i16: REF_CALL(element::Type_t::i16);
            case element::Type_t::i32: REF_CALL(element::Type_t::i32);
            case element::Type_t::i64: REF_CALL(element::Type_t::i64);
            case element::Type_t::u8: REF_CALL(element::Type_t::u8);
            case element::Type_t::u16: REF_CALL(element::Type_t::u16);
            case element::Type_t::u32: REF_CALL(element::Type_t::u32);
            case element::Type_t::u64: REF_CALL(element::Type_t::u64);
            case element::Type_t::f16: REF_CALL(element::Type_t::f16);
            case element::Type_t::f32: REF_CALL(element::Type_t::f32);
            case element::Type_t::f64: REF_CALL(element::Type_t::f64);
            default: return false;
            }
#undef REF_CALL
        }
        return true;
    }

    //    TODO: Rewrite to v1
    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::OneHot>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        switch (inputs[0]->get_element_type())
        {
        case element::Type_t::i32:
            runtime::reference::one_hot<element_type_traits<element::Type_t::i32>::value_type, T>(
                inputs[0]->get_data_ptr<element::Type_t::i32>(),
                outputs[0]->get_data_ptr<T>(),
                inputs[0]->get_shape(),
                outputs[0]->get_shape(),
                op->get_axis(),
                inputs[2]->get_data_ptr<T>()[0],
                inputs[3]->get_data_ptr<T>()[0]);
            break;
        case element::Type_t::i64:
            runtime::reference::one_hot<element_type_traits<element::Type_t::i64>::value_type, T>(
                inputs[0]->get_data_ptr<element::Type_t::i64>(),
                outputs[0]->get_data_ptr<T>(),
                inputs[0]->get_shape(),
                outputs[0]->get_shape(),
                op->get_axis(),
                inputs[2]->get_data_ptr<T>()[0],
                inputs[3]->get_data_ptr<T>()[0]);
            break;
        default:
            std::stringstream ss;
            ss << "Unhandled input precision " << inputs[0]->get_element_type().get_type_name()
               << " in v1::OneHot evaluate call";
            throw ngraph_error(ss.str());
        }
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::RNNCell>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::rnn_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                        inputs[0]->get_shape(),
                                        inputs[1]->get_data_ptr<ET>(),
                                        inputs[1]->get_shape(),
                                        inputs[2]->get_data_ptr<ET>(),
                                        inputs[2]->get_shape(),
                                        inputs[3]->get_data_ptr<ET>(),
                                        inputs[3]->get_shape(),
                                        inputs[4]->get_data_ptr<ET>(),
                                        inputs[4]->get_shape(),
                                        outputs[0]->get_data_ptr<ET>(),
                                        op->get_activations().front(),
                                        op->get_clip());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v4::LSTMCell>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::lstm_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                         inputs[0]->get_shape(),
                                         inputs[1]->get_data_ptr<ET>(),
                                         inputs[1]->get_shape(),
                                         inputs[2]->get_data_ptr<ET>(),
                                         inputs[2]->get_shape(),
                                         inputs[3]->get_data_ptr<ET>(),
                                         inputs[3]->get_shape(),
                                         inputs[4]->get_data_ptr<ET>(),
                                         inputs[4]->get_shape(),
                                         inputs[5]->get_data_ptr<ET>(),
                                         inputs[5]->get_shape(),
                                         outputs[0]->get_data_ptr<ET>(),
                                         outputs[1]->get_data_ptr<ET>(),
                                         op->get_activations()[0],
                                         op->get_activations()[1],
                                         op->get_activations()[2],
                                         op->get_clip());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v3::GRUCell>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::gru_cell<T>(inputs[0]->get_data_ptr<ET>(),
                                        inputs[0]->get_shape(),
                                        inputs[1]->get_data_ptr<ET>(),
                                        inputs[1]->get_shape(),
                                        inputs[2]->get_data_ptr<ET>(),
                                        inputs[2]->get_shape(),
                                        inputs[3]->get_data_ptr<ET>(),
                                        inputs[3]->get_shape(),
                                        inputs[4]->get_data_ptr<ET>(),
                                        inputs[4]->get_shape(),
                                        outputs[0]->get_data_ptr<ET>(),
                                        op->get_activations()[0],
                                        op->get_activations()[1],
                                        op->get_clip(),
                                        op->get_linear_before_reset());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::RNNSequence>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::rnn_sequence<T>(inputs[0]->get_data_ptr<char>(),
                                            inputs[0]->get_shape(),
                                            inputs[1]->get_data_ptr<char>(),
                                            inputs[1]->get_shape(),
                                            inputs[2]->get_data_ptr<char>(),
                                            inputs[2]->get_shape(),
                                            inputs[3]->get_data_ptr<char>(),
                                            inputs[3]->get_shape(),
                                            inputs[4]->get_data_ptr<char>(),
                                            inputs[4]->get_shape(),
                                            inputs[5]->get_data_ptr<char>(),
                                            inputs[5]->get_shape(),
                                            outputs[0]->get_data_ptr<char>(),
                                            outputs[1]->get_data_ptr<char>(),
                                            op->get_activations()[0],
                                            op->get_clip(),
                                            op->get_direction());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::LSTMSequence>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::lstm_sequence<T>(inputs[0]->get_data_ptr<char>(),
                                             inputs[0]->get_shape(),
                                             inputs[1]->get_data_ptr<char>(),
                                             inputs[1]->get_shape(),
                                             inputs[2]->get_data_ptr<char>(),
                                             inputs[2]->get_shape(),
                                             inputs[3]->get_data_ptr<char>(),
                                             inputs[3]->get_shape(),
                                             inputs[4]->get_data_ptr<char>(),
                                             inputs[4]->get_shape(),
                                             inputs[5]->get_data_ptr<char>(),
                                             inputs[5]->get_shape(),
                                             inputs[6]->get_data_ptr<char>(),
                                             inputs[6]->get_shape(),
                                             outputs[0]->get_data_ptr<char>(),
                                             outputs[1]->get_data_ptr<char>(),
                                             outputs[2]->get_data_ptr<char>(),
                                             op->get_activations()[0],
                                             op->get_activations()[1],
                                             op->get_activations()[2],
                                             op->get_clip(),
                                             op->get_direction());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v5::GRUSequence>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::gru_sequence<T>(inputs[0]->get_data_ptr<char>(),
                                            inputs[0]->get_shape(),
                                            inputs[1]->get_data_ptr<char>(),
                                            inputs[1]->get_shape(),
                                            inputs[2]->get_data_ptr<char>(),
                                            inputs[2]->get_shape(),
                                            inputs[3]->get_data_ptr<char>(),
                                            inputs[3]->get_shape(),
                                            inputs[4]->get_data_ptr<char>(),
                                            inputs[4]->get_shape(),
                                            inputs[5]->get_data_ptr<char>(),
                                            inputs[5]->get_shape(),
                                            outputs[0]->get_data_ptr<char>(),
                                            outputs[1]->get_data_ptr<char>(),
                                            op->get_activations()[0],
                                            op->get_activations()[1],
                                            op->get_clip(),
                                            op->get_direction(),
                                            op->get_linear_before_reset());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::Pad>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::pad(inputs[0]->get_data_ptr<char>(),
                                inputs[1]->get_data_ptr<char>(),
                                outputs[0]->get_data_ptr<char>(),
                                shape_size(inputs[0]->get_shape()),
                                inputs[1]->get_shape(),
                                outputs[0]->get_shape(),
                                op->get_pads_end(),
                                op->get_pads_begin(),
                                op->get_pad_mode());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v1::GatherTree>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::gather_tree(inputs[0]->get_data_ptr<const char>(),
                                        inputs[1]->get_data_ptr<const char>(),
                                        inputs[2]->get_data_ptr<const char>(),
                                        inputs[3]->get_data_ptr<const char>(),
                                        outputs[0]->get_data_ptr<char>(),
                                        op->get_input_shape(0),
                                        op->get_input_shape(1),
                                        op->get_input_shape(2),
                                        op->get_input_shape(3),
                                        inputs[1]->get_element_type());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::FakeQuantize>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::fake_quantize<T>(inputs[0]->get_data_ptr<const T>(),
                                             inputs[1]->get_data_ptr<const T>(),
                                             inputs[2]->get_data_ptr<const T>(),
                                             inputs[3]->get_data_ptr<const T>(),
                                             inputs[4]->get_data_ptr<const T>(),
                                             outputs[0]->get_data_ptr<T>(),
                                             op->get_input_shape(0),
                                             op->get_input_shape(1),
                                             op->get_input_shape(2),
                                             op->get_input_shape(3),
                                             op->get_input_shape(4),
                                             op->get_levels());
        return true;
    }

    template <element::Type_t ET>
    bool evaluate(const shared_ptr<op::v0::NormalizeL2>& op,
                  const HostTensorVector& outputs,
                  const HostTensorVector& inputs)
    {
        using T = typename element_type_traits<ET>::value_type;
        runtime::reference::normalize_l2<T>(inputs[0]->get_data_ptr<const T>(),
                                            outputs[0]->get_data_ptr<T>(),
                                            op->get_input_shape(0),
                                            op->get_reduction_axes(),
                                            op->get_eps(),
                                            op->get_eps_mode());
        return true;
    }

    template <typename T>
    bool evaluate_node(std::shared_ptr<Node> node,
                       const HostTensorVector& outputs,
                       const HostTensorVector& inputs)
    {
        auto element_type = node->get_output_element_type(0);
        if (is_type<op::v1::Select>(node))
        {
            element_type = node->get_input_element_type(1);
        }
        else if (is_type<op::v0::PriorBox>(node))
        {
            element_type = node->get_input_element_type(0);
        }
        for (size_t i = 1; i < node->outputs().size(); i++)
        {
            if (element_type != node->get_output_element_type(i))
            {
                throw std::logic_error("Output node element types is not equal");
            }
        }
        switch (element_type)
        {
        case element::Type_t::boolean:
            return evaluate<element::Type_t::boolean>(as_type_ptr<T>(node), outputs, inputs);
            ;
        //            case element::Type_t::bf16:
        //                break;
        case element::Type_t::f16:
            return evaluate<element::Type_t::f16>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::f64:
            return evaluate<element::Type_t::f64>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::f32:
            return evaluate<element::Type_t::f32>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::i8:
            return evaluate<element::Type_t::i8>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::i16:
            return evaluate<element::Type_t::i16>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::i32:
            return evaluate<element::Type_t::i32>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::i64:
            return evaluate<element::Type_t::i64>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::u8:
            return evaluate<element::Type_t::u8>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::u16:
            return evaluate<element::Type_t::u16>(as_type_ptr<T>(node), outputs, inputs);
        case element::Type_t::u32:
            return evaluate<element::Type_t::u32>(as_type_ptr<T>(node), outputs, inputs);
        default:
            throw ngraph_error(std::string("Unhandled data type ") +
                               node->get_element_type().get_type_name() +
                               std::string("in evaluate_node()"));
        }
    }
} // namespace

runtime::interpreter::EvaluatorsMap& runtime::interpreter::get_evaluators_map()
{
    static runtime::interpreter::EvaluatorsMap evaluatorsMap{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, evaluate_node<NAMESPACE::NAME>},

#include "opset_int_tbl.hpp"

#undef NGRAPH_OP
    };
    return evaluatorsMap;
}