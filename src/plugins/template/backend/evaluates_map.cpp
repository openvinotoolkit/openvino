// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

// #include <ngraph/runtime/reference/abs.hpp>
// #include <ngraph/runtime/reference/adaptive_avg_pool.hpp>
// #include <ngraph/runtime/reference/adaptive_max_pool.hpp>
// #include <ngraph/runtime/reference/avg_pool.hpp>
// #include <ngraph/runtime/reference/batch_norm.hpp>
// #include <ngraph/runtime/reference/binary_convolution.hpp>
// #include <ngraph/runtime/reference/bucketize.hpp>
// #include <ngraph/runtime/reference/ceiling.hpp>
// #include <ngraph/runtime/reference/convert.hpp>
// #include <ngraph/runtime/reference/convolution.hpp>
// #include <ngraph/runtime/reference/convolution_backprop_data.hpp>
// #include <ngraph/runtime/reference/ctc_greedy_decoder.hpp>
// #include <ngraph/runtime/reference/ctc_greedy_decoder_seq_len.hpp>
// #include <ngraph/runtime/reference/ctc_loss.hpp>
// #include <ngraph/runtime/reference/cum_sum.hpp>
// #include <ngraph/runtime/reference/deformable_convolution.hpp>
// #include <ngraph/runtime/reference/deformable_psroi_pooling.hpp>
// #include <ngraph/runtime/reference/detection_output.hpp>
// #include <ngraph/runtime/reference/einsum.hpp>
// #include <ngraph/runtime/reference/elu.hpp>
// #include <ngraph/runtime/reference/embedding_bag_offsets_sum.hpp>
// #include <ngraph/runtime/reference/embedding_bag_packed_sum.hpp>
// #include <ngraph/runtime/reference/embedding_segments_sum.hpp>
// #include <ngraph/runtime/reference/equal.hpp>
// #include <ngraph/runtime/reference/exp.hpp>
// #include <ngraph/runtime/reference/experimental_detectron_detection_output.hpp>
// #include <ngraph/runtime/reference/experimental_detectron_prior_grid_generator.hpp>
// #include <ngraph/runtime/reference/experimental_detectron_proposal_single_image.hpp>
// #include <ngraph/runtime/reference/experimental_detectron_roi_feature_extractor.hpp>
// #include <ngraph/runtime/reference/experimental_detectron_topk_rois.hpp>
// #include <ngraph/runtime/reference/extract_image_patches.hpp>
// #include <ngraph/runtime/reference/fft.hpp>
// #include <ngraph/runtime/reference/gather.hpp>
// #include <ngraph/runtime/reference/gather_elements.hpp>
// #include <ngraph/runtime/reference/gather_nd.hpp>
// #include <ngraph/runtime/reference/gather_tree.hpp>
// #include <ngraph/runtime/reference/gelu.hpp>
#include <ngraph/runtime/reference/generate_proposal.hpp>
// #include <ngraph/runtime/reference/greater.hpp>
// #include <ngraph/runtime/reference/grid_sample.hpp>
// #include <ngraph/runtime/reference/grn.hpp>
// #include <ngraph/runtime/reference/group_convolution.hpp>
// #include <ngraph/runtime/reference/group_convolution_backprop_data.hpp>
// #include <ngraph/runtime/reference/gru_cell.hpp>
// #include <ngraph/runtime/reference/hard_sigmoid.hpp>
// #include <ngraph/runtime/reference/if.hpp>
// #include <ngraph/runtime/reference/interpolate.hpp>
// #include <ngraph/runtime/reference/irdft.hpp>
// #include <ngraph/runtime/reference/is_finite.hpp>
// #include <ngraph/runtime/reference/is_inf.hpp>
// #include <ngraph/runtime/reference/is_nan.hpp>
// #include <ngraph/runtime/reference/log.hpp>
// #include <ngraph/runtime/reference/log_softmax.hpp>
// #include <ngraph/runtime/reference/lrn.hpp>
// #include <ngraph/runtime/reference/lstm_cell.hpp>
// #include <ngraph/runtime/reference/matrix_nms.hpp>
// #include <ngraph/runtime/reference/mod.hpp>
// #include <ngraph/runtime/reference/multiclass_nms.hpp>
// #include <ngraph/runtime/reference/mvn.hpp>
// #include <ngraph/runtime/reference/non_max_suppression.hpp>
// #include <ngraph/runtime/reference/normalize_l2.hpp>
// #include <ngraph/runtime/reference/pad.hpp>
#include <ngraph/runtime/reference/prelu.hpp>
#include <ngraph/runtime/reference/prior_box.hpp>
#include <ngraph/runtime/reference/proposal.hpp>
#include <ngraph/runtime/reference/psroi_pooling.hpp>
#include <ngraph/runtime/reference/rdft.hpp>
#include <ngraph/runtime/reference/region_yolo.hpp>
#include <ngraph/runtime/reference/reorg_yolo.hpp>
#include <ngraph/runtime/reference/reverse_sequence.hpp>
#include <ngraph/runtime/reference/rnn_cell.hpp>
#include <ngraph/runtime/reference/roi_align.hpp>
#include <ngraph/runtime/reference/roi_pooling.hpp>
#include <ngraph/runtime/reference/roll.hpp>
#include <ngraph/runtime/reference/scatter_nd_update.hpp>
#include <ngraph/runtime/reference/selu.hpp>
#include <ngraph/runtime/reference/sequences.hpp>
#include <ngraph/runtime/reference/sigmoid.hpp>
#include <ngraph/runtime/reference/sign.hpp>
#include <ngraph/runtime/reference/softsign.hpp>
#include <ngraph/runtime/reference/squared_difference.hpp>
#include <ngraph/runtime/reference/tanh.hpp>
#include <ngraph/runtime/reference/tensor_iterator.hpp>
#include <ngraph/runtime/reference/unique.hpp>
#include <ngraph/runtime/reference/utils/nms_common.hpp>

#include "backend.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/convert_color_nv12.hpp"
#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"

using namespace ngraph;
using namespace std;
namespace {
template <element::Type_t ET>
bool evaluate(shared_ptr<Node> op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    return false;
}

namespace {
std::vector<float> get_floats(const std::shared_ptr<HostTensor>& input, const Shape& shape) {
    size_t input_size = shape_size(shape);
    std::vector<float> result(input_size);

    switch (input->get_element_type()) {
    case element::Type_t::bf16: {
        bfloat16* p = input->get_data_ptr<bfloat16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case element::Type_t::f16: {
        float16* p = input->get_data_ptr<float16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case element::Type_t::f32: {
        float* p = input->get_data_ptr<float>();
        memcpy(result.data(), p, input_size * sizeof(float));
    } break;
    default:
        throw std::runtime_error("Unsupported data type.");
        break;
    }

    return result;
}
}  // namespace

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v3::ScatterNDUpdate>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    auto idxType = op->get_input_element_type(1);
    if (idxType == element::i32) {
        runtime::reference::scatterNdUpdate<T, int32_t>(inputs[0]->get_data_ptr<const T>(),
                                                        inputs[1]->get_data_ptr<const int32_t>(),
                                                        inputs[2]->get_data_ptr<const T>(),
                                                        outputs[0]->get_data_ptr<T>(),
                                                        op->get_input_shape(0),
                                                        op->get_input_shape(1),
                                                        op->get_input_shape(2));
    } else if (idxType == element::i64) {
        runtime::reference::scatterNdUpdate<T, int64_t>(inputs[0]->get_data_ptr<const T>(),
                                                        inputs[1]->get_data_ptr<const int64_t>(),
                                                        inputs[2]->get_data_ptr<const T>(),
                                                        outputs[0]->get_data_ptr<T>(),
                                                        op->get_input_shape(0),
                                                        op->get_input_shape(1),
                                                        op->get_input_shape(2));
    } else {
        throw ngraph_error("ScatterNDUpdate layer support only i32 and i64 'indices' input precision!");
    }
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::Proposal>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::proposal_v0<T>(inputs[0]->get_data_ptr<T>(),
                                       inputs[1]->get_data_ptr<T>(),
                                       inputs[2]->get_data_ptr<T>(),
                                       outputs[0]->get_data_ptr<T>(),
                                       inputs[0]->get_shape(),
                                       inputs[1]->get_shape(),
                                       inputs[2]->get_shape(),
                                       outputs[0]->get_shape(),
                                       op.get()->get_attrs());
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v4::Proposal>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::proposal_v4<T>(inputs[0]->get_data_ptr<T>(),
                                       inputs[1]->get_data_ptr<T>(),
                                       inputs[2]->get_data_ptr<T>(),
                                       outputs[0]->get_data_ptr<T>(),
                                       outputs[1]->get_data_ptr<T>(),
                                       inputs[0]->get_shape(),
                                       inputs[1]->get_shape(),
                                       inputs[2]->get_shape(),
                                       outputs[0]->get_shape(),
                                       outputs[1]->get_shape(),
                                       op.get()->get_attrs());
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::Selu>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::selu<T>(inputs[0]->get_data_ptr<T>(),
                                inputs[1]->get_data_ptr<T>(),
                                inputs[2]->get_data_ptr<T>(),
                                outputs[0]->get_data_ptr<T>(),
                                shape_size(inputs[0]->get_shape()),
                                shape_size(inputs[1]->get_shape()),
                                shape_size(inputs[2]->get_shape()));
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::Relu>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::relu<T>(inputs[0]->get_data_ptr<T>(),
                                outputs[0]->get_data_ptr<T>(),
                                shape_size(inputs[0]->get_shape()));
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::PRelu>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::prelu<T>(inputs[0]->get_data_ptr<T>(),
                                 inputs[1]->get_data_ptr<T>(),
                                 outputs[0]->get_data_ptr<T>(),
                                 inputs[0]->get_shape(),
                                 inputs[1]->get_shape());
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::Sign>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::sign<T>(inputs[0]->get_data_ptr<T>(),
                                outputs[0]->get_data_ptr<T>(),
                                shape_size(inputs[0]->get_shape()));
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::Sigmoid>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::sigmoid<T>(inputs[0]->get_data_ptr<T>(),
                                   outputs[0]->get_data_ptr<T>(),
                                   shape_size(inputs[0]->get_shape()));
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::Tanh>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::tanh<T>(inputs[0]->get_data_ptr<T>(),
                                outputs[0]->get_data_ptr<T>(),
                                shape_size(inputs[0]->get_shape()));
    return true;
}

namespace reverse_sequence_v0 {
template <element::Type_t t1, element::Type_t t2>
inline void evaluate(const shared_ptr<op::v0::ReverseSequence>& op,
                     const HostTensorVector& outputs,
                     const HostTensorVector& inputs) {
    using T1 = typename element_type_traits<t1>::value_type;
    using T2 = typename element_type_traits<t2>::value_type;
    runtime::reference::reverse_sequence<T1, T2>(inputs[0]->get_data_ptr<T1>(),
                                                 outputs[0]->get_data_ptr<T1>(),
                                                 inputs[0]->get_shape(),
                                                 op->get_batch_axis(),
                                                 op->get_sequence_axis(),
                                                 inputs[1]->get_data_ptr<T2>());
}
}  // namespace reverse_sequence_v0

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::ReverseSequence>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    switch (inputs[1]->get_element_type()) {
    case element::Type_t::boolean:
        reverse_sequence_v0::evaluate<ET, element::Type_t::boolean>(op, outputs, inputs);
        break;
    case element::Type_t::i8:
        reverse_sequence_v0::evaluate<ET, element::Type_t::i8>(op, outputs, inputs);
        break;
    case element::Type_t::i16:
        reverse_sequence_v0::evaluate<ET, element::Type_t::i16>(op, outputs, inputs);
        break;
    case element::Type_t::i32:
        reverse_sequence_v0::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
        break;
    case element::Type_t::i64:
        reverse_sequence_v0::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
        break;
    case element::Type_t::u8:
        reverse_sequence_v0::evaluate<ET, element::Type_t::u8>(op, outputs, inputs);
        break;
    case element::Type_t::u16:
        reverse_sequence_v0::evaluate<ET, element::Type_t::u16>(op, outputs, inputs);
        break;
    case element::Type_t::u32:
        reverse_sequence_v0::evaluate<ET, element::Type_t::u32>(op, outputs, inputs);
        break;
    case element::Type_t::u64:
        reverse_sequence_v0::evaluate<ET, element::Type_t::u64>(op, outputs, inputs);
        break;
    case element::Type_t::f16:
        reverse_sequence_v0::evaluate<ET, element::Type_t::f16>(op, outputs, inputs);
        break;
    case element::Type_t::f32:
        reverse_sequence_v0::evaluate<ET, element::Type_t::f32>(op, outputs, inputs);
        break;
    case element::Type_t::f64:
        reverse_sequence_v0::evaluate<ET, element::Type_t::f64>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::RNNCell>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
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
bool evaluate(const shared_ptr<op::v0::LSTMCell>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::lstm_cell_v1<T>(inputs[0]->get_data_ptr<ET>(),
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
                                        inputs[6]->get_data_ptr<ET>(),
                                        inputs[6]->get_shape(),
                                        outputs[0]->get_data_ptr<ET>(),
                                        outputs[1]->get_data_ptr<ET>(),
                                        op->get_activations()[0],
                                        op->get_activations()[1],
                                        op->get_activations()[2],
                                        op->get_clip(),
                                        op->get_weights_format(),
                                        op->get_input_forget());
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v4::LSTMCell>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
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
bool evaluate(const shared_ptr<op::v3::GRUCell>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
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
bool evaluate(const shared_ptr<ov::op::internal::AUGRUCell>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
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
                                    op->get_linear_before_reset(),
                                    inputs[5]->get_data_ptr<ET>());
    return true;
}

namespace rnn_seq_v5 {
template <element::Type_t t1, element::Type_t t2>
inline void evaluate(const shared_ptr<op::v5::RNNSequence>& op,
                     const HostTensorVector& outputs,
                     const HostTensorVector& inputs) {
    using T1 = typename element_type_traits<t1>::value_type;
    using T2 = typename element_type_traits<t2>::value_type;
    runtime::reference::rnn_sequence<T1, T2>(inputs[0]->get_data_ptr<char>(),
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
}
}  // namespace rnn_seq_v5

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v5::RNNSequence>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    switch (inputs[2]->get_element_type()) {
    case element::Type_t::i64:
    case element::Type_t::u64:
        rnn_seq_v5::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
        break;
    case element::Type_t::i32:
    case element::Type_t::u32:
        rnn_seq_v5::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

namespace lstm_seq_v1 {
template <element::Type_t t1, element::Type_t t2>
inline void evaluate(const shared_ptr<op::v0::LSTMSequence>& op,
                     const HostTensorVector& outputs,
                     const HostTensorVector& inputs) {
    using T1 = typename element_type_traits<t1>::value_type;
    using T2 = typename element_type_traits<t2>::value_type;
    runtime::reference::lstm_sequence_v1<T1, T2>(inputs[0]->get_data_ptr<char>(),
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
                                                 inputs[7]->get_data_ptr<char>(),
                                                 inputs[7]->get_shape(),
                                                 outputs[0]->get_data_ptr<char>(),
                                                 outputs[1]->get_data_ptr<char>(),
                                                 outputs[2]->get_data_ptr<char>(),
                                                 op->get_activations()[0],
                                                 op->get_activations()[1],
                                                 op->get_activations()[2],
                                                 op->get_clip_threshold(),
                                                 op->get_weights_format(),
                                                 op->get_input_forget(),
                                                 op->get_direction());
}
}  // namespace lstm_seq_v1

namespace lstm_seq_v5 {
template <element::Type_t t1, element::Type_t t2>
inline void evaluate(const shared_ptr<op::v5::LSTMSequence>& op,
                     const HostTensorVector& outputs,
                     const HostTensorVector& inputs) {
    using T1 = typename element_type_traits<t1>::value_type;
    using T2 = typename element_type_traits<t2>::value_type;
    runtime::reference::lstm_sequence<T1, T2>(inputs[0]->get_data_ptr<char>(),
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
}
}  // namespace lstm_seq_v5

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::LSTMSequence>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    switch (inputs[3]->get_element_type()) {
    case element::Type_t::i64:
    case element::Type_t::u64:
        lstm_seq_v1::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
        break;
    case element::Type_t::i32:
    case element::Type_t::u32:
        lstm_seq_v1::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v5::LSTMSequence>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    switch (inputs[3]->get_element_type()) {
    case element::Type_t::i64:
    case element::Type_t::u64:
        lstm_seq_v5::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
        break;
    case element::Type_t::i32:
    case element::Type_t::u32:
        lstm_seq_v5::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

namespace ti_v0 {
runtime::reference::custom_evaluate_function evaluate = [](const std::shared_ptr<ngraph::Function>& function,
                                                           const HostTensorVector& inputs,
                                                           HostTensorVector& outputs) -> void {
    const auto& parameters = function->get_parameters();
    const auto& parametersNumber = parameters.size();
    const auto& inputsNumber = inputs.size();
    NGRAPH_CHECK(parametersNumber == inputsNumber,
                 "Got function (",
                 function->get_friendly_name(),
                 ") with ",
                 parametersNumber,
                 " parameters, but ",
                 inputsNumber,
                 " input blobs");

    auto inputTensors = std::vector<std::shared_ptr<runtime::Tensor>>{};
    for (const auto& parameter : parameters) {
        const auto& parameterIndex = function->get_parameter_index(parameter);
        const auto& parameterShape = parameter->get_shape();
        const auto& parameterType = parameter->get_element_type();
        const auto& parameterSize = shape_size(parameterShape) * parameterType.size();

        const auto& input = inputs[parameterIndex];
        const auto& inputSize = input->get_size_in_bytes();
        NGRAPH_CHECK(parameterSize == inputSize,
                     "Got parameter (",
                     parameter->get_friendly_name(),
                     ") of size ",
                     parameterSize,
                     " bytes, but corresponding input with index ",
                     parameterIndex,
                     " has ",
                     inputSize,
                     " bytes");

        auto tensor = std::make_shared<runtime::HostTensor>(parameterType, parameterShape);
        tensor->write(input->get_data_ptr(), parameterSize);
        inputTensors.push_back(tensor);
    }

    const auto& results = function->get_results();
    std::vector<std::shared_ptr<ngraph::runtime::Tensor>> outputTensors;
    outputTensors.reserve(results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        outputTensors.push_back(std::make_shared<HostTensor>());
    }
    auto backend = runtime::Backend::create();
    auto handle = backend->compile(function);
    handle->call_with_validate(outputTensors, inputTensors);

    outputs.reserve(outputTensors.size());
    for (const auto& tensor : outputTensors) {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        outputs.push_back(host_tensor);
    }
};
}  // namespace ti_v0

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::TensorIterator>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    runtime::reference::tensor_iterator(op->get_num_iterations(),
                                        op->get_function(),
                                        op->get_output_descriptions(),
                                        op->get_input_descriptions(),
                                        outputs,
                                        inputs,
                                        ti_v0::evaluate);
    return true;
}

namespace gru_seq_v5 {
template <element::Type_t t1, element::Type_t t2>
inline void evaluate(const shared_ptr<op::v5::GRUSequence>& op,
                     const HostTensorVector& outputs,
                     const HostTensorVector& inputs) {
    using T1 = typename element_type_traits<t1>::value_type;
    using T2 = typename element_type_traits<t2>::value_type;
    runtime::reference::gru_sequence<T1, T2>(inputs[0]->get_data_ptr<char>(),
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
}
}  // namespace gru_seq_v5

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v5::GRUSequence>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    switch (inputs[2]->get_element_type()) {
    case element::Type_t::i64:
    case element::Type_t::u64:
        gru_seq_v5::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
        break;
    case element::Type_t::i32:
    case element::Type_t::u32:
        gru_seq_v5::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

namespace augru_seq {
template <element::Type_t t1, element::Type_t t2>
inline void evaluate(const shared_ptr<ov::op::internal::AUGRUSequence>& op,
                     const HostTensorVector& outputs,
                     const HostTensorVector& inputs) {
    using T1 = typename element_type_traits<t1>::value_type;
    using T2 = typename element_type_traits<t2>::value_type;
    runtime::reference::gru_sequence<T1, T2>(inputs[0]->get_data_ptr<char>(),
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
                                             op->get_linear_before_reset(),
                                             inputs[6]->get_data_ptr<char>());
}
}  // namespace augru_seq

template <element::Type_t ET>
bool evaluate(const shared_ptr<ov::op::internal::AUGRUSequence>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    switch (inputs[2]->get_element_type()) {
    case element::Type_t::i64:
    case element::Type_t::u64:
        augru_seq::evaluate<ET, element::Type_t::i64>(op, outputs, inputs);
        break;
    case element::Type_t::i32:
    case element::Type_t::u32:
        augru_seq::evaluate<ET, element::Type_t::i32>(op, outputs, inputs);
        break;
    default:
        return false;
    }
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v9::ROIAlign>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<int64_t> batch_indices_vec_scaled_up = host_tensor_2_vector<int64_t>(inputs[2]);
    op::v3::ROIAlign::PoolingMode m_mode_v3;
    switch (op->get_mode()) {
    case op::v9::ROIAlign::PoolingMode::AVG: {
        m_mode_v3 = op::v3::ROIAlign::PoolingMode::AVG;
        break;
    }
    case op::v9::ROIAlign::PoolingMode::MAX: {
        m_mode_v3 = op::v3::ROIAlign::PoolingMode::MAX;
        break;
    }
    default: {
        NGRAPH_CHECK(false, "unsupported PoolingMode ");
    }
    }
    runtime::reference::roi_align<T>(inputs[0]->get_data_ptr<const T>(),
                                     inputs[1]->get_data_ptr<const T>(),
                                     batch_indices_vec_scaled_up.data(),
                                     outputs[0]->get_data_ptr<T>(),
                                     op->get_input_shape(0),
                                     op->get_input_shape(1),
                                     op->get_input_shape(2),
                                     op->get_output_shape(0),
                                     op->get_pooled_h(),
                                     op->get_pooled_w(),
                                     op->get_sampling_ratio(),
                                     op->get_spatial_scale(),
                                     m_mode_v3,
                                     op->get_aligned_mode());
    return true;
}
template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::ROIPooling>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::roi_pooling<T>(inputs[0]->get_data_ptr<const T>(),
                                       inputs[1]->get_data_ptr<const T>(),
                                       outputs[0]->get_data_ptr<T>(),
                                       op->get_input_shape(0),
                                       op->get_input_shape(1),
                                       op->get_output_shape(0),
                                       op->get_spatial_scale(),
                                       op->get_method());
    return true;
}
template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::ReorgYolo>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    runtime::reference::reorg_yolo(inputs[0]->get_data_ptr<char>(),
                                   outputs[0]->get_data_ptr<char>(),
                                   inputs[0]->get_shape(),
                                   op->get_strides().at(0),
                                   inputs[0]->get_element_type().size());
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::RegionYolo>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::region_yolo<T>(inputs[0]->get_data_ptr<const T>(),
                                       outputs[0]->get_data_ptr<T>(),
                                       inputs[0]->get_shape(),
                                       static_cast<int>(op->get_num_coords()),
                                       static_cast<int>(op->get_num_classes()),
                                       static_cast<int>(op->get_num_regions()),
                                       op->get_do_softmax(),
                                       op->get_mask());
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::SquaredDifference>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::squared_difference<T>(inputs[0]->get_data_ptr<const T>(),
                                              inputs[1]->get_data_ptr<const T>(),
                                              outputs[0]->get_data_ptr<T>(),
                                              inputs[0]->get_shape(),
                                              inputs[1]->get_shape(),
                                              op->get_autob());
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::PSROIPooling>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::psroi_pooling<T>(inputs[0]->get_data_ptr<T>(),
                                         inputs[0]->get_shape(),
                                         inputs[1]->get_data_ptr<T>(),
                                         inputs[1]->get_shape(),
                                         outputs[0]->get_data_ptr<T>(),
                                         outputs[0]->get_shape(),
                                         op->get_mode(),
                                         op->get_spatial_scale(),
                                         op->get_spatial_bins_x(),
                                         op->get_spatial_bins_y());

    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v7::Roll>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    const auto& shiftType = inputs[1]->get_element_type();
    std::vector<int64_t> shift_int64;
    if (shiftType == element::Type_t::i32) {
        auto shift = inputs[1]->get_data_ptr<const int32_t>();
        shift_int64.resize(shape_size(inputs[1]->get_shape()));
        std::transform(shift, shift + shape_size(inputs[1]->get_shape()), shift_int64.begin(), [](const int32_t& elem) {
            return static_cast<int64_t>(elem);
        });
    }
    const auto& axesType = inputs[2]->get_element_type();
    std::vector<int64_t> axes_int64;
    if (axesType == element::Type_t::i32) {
        auto axes = inputs[2]->get_data_ptr<const int32_t>();
        axes_int64.resize(shape_size(inputs[2]->get_shape()));
        std::transform(axes, axes + shape_size(inputs[2]->get_shape()), axes_int64.begin(), [](const int32_t& elem) {
            return static_cast<int64_t>(elem);
        });
    }
    runtime::reference::roll(
        inputs[0]->get_data_ptr<const char>(),
        inputs[1]->get_element_type() != element::Type_t::i64 ? shift_int64.data()
                                                              : inputs[1]->get_data_ptr<const int64_t>(),
        inputs[2]->get_element_type() != element::Type_t::i64 ? axes_int64.data()
                                                              : inputs[2]->get_data_ptr<const int64_t>(),
        outputs[0]->get_data_ptr<char>(),
        inputs[0]->get_shape(),
        inputs[1]->get_shape(),
        inputs[2]->get_shape(),
        inputs[0]->get_element_type().size());
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v8::Gather>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    if (op->get_input_element_type(1) == element::i64) {
        runtime::reference::gather<T, int64_t>(inputs[0]->get_data_ptr<T>(),
                                               inputs[1]->get_data_ptr<int64_t>(),
                                               outputs[0]->get_data_ptr<T>(),
                                               op->get_input_shape(0),
                                               op->get_input_shape(1),
                                               op->get_output_shape(0),
                                               op->get_axis(),
                                               op->get_batch_dims());
    } else if (op->get_input_element_type(1) == element::i32) {
        runtime::reference::gather<T, int32_t>(inputs[0]->get_data_ptr<T>(),
                                               inputs[1]->get_data_ptr<int32_t>(),
                                               outputs[0]->get_data_ptr<T>(),
                                               op->get_input_shape(0),
                                               op->get_input_shape(1),
                                               op->get_output_shape(0),
                                               op->get_axis(),
                                               op->get_batch_dims());
    } else {
        throw ngraph_error("Unexpected indices type for Gather operation");
    }
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v3::Assign>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    outputs[0]->set_unary(inputs[0]);
    void* input = inputs[0]->get_data_ptr();
    outputs[0]->write(input, outputs[0]->get_size_in_bytes());
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v3::ReadValue>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    outputs[0]->set_unary(inputs[0]);
    void* input = inputs[0]->get_data_ptr();
    outputs[0]->write(input, outputs[0]->get_size_in_bytes());
    return true;
}

template <ov::element::Type_t ET>
inline bool evaluate(const shared_ptr<op::v8::NV12toRGB>& op,
                     const HostTensorVector& outputs,
                     const HostTensorVector& inputs) {
    return runtime::reference::color_convert_nv12<ET>(op,
                                                      outputs,
                                                      inputs,
                                                      ov::op::util::ConvertColorNV12Base::ColorConversion::NV12_TO_RGB);
}

template <ov::element::Type_t ET>
inline bool evaluate(const shared_ptr<op::v8::NV12toBGR>& op,
                     const HostTensorVector& outputs,
                     const HostTensorVector& inputs) {
    return runtime::reference::color_convert_nv12<ET>(op,
                                                      outputs,
                                                      inputs,
                                                      ov::op::util::ConvertColorNV12Base::ColorConversion::NV12_TO_BGR);
}

template <ov::element::Type_t ET>
inline bool evaluate(const shared_ptr<op::v8::I420toRGB>& op,
                     const HostTensorVector& outputs,
                     const HostTensorVector& inputs) {
    return runtime::reference::color_convert_i420<ET>(op,
                                                      outputs,
                                                      inputs,
                                                      ov::op::util::ConvertColorI420Base::ColorConversion::I420_TO_RGB);
}

template <ov::element::Type_t ET>
inline bool evaluate(const shared_ptr<op::v8::I420toBGR>& op,
                     const HostTensorVector& outputs,
                     const HostTensorVector& inputs) {
    return runtime::reference::color_convert_i420<ET>(op,
                                                      outputs,
                                                      inputs,
                                                      ov::op::util::ConvertColorI420Base::ColorConversion::I420_TO_BGR);
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v9::SoftSign>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    element::Type input_et = op->get_input_element_type(0);
    switch (input_et) {
    case element::Type_t::f64:
        runtime::reference::softsign<double>(inputs[0]->get_data_ptr<double>(),
                                             outputs[0]->get_data_ptr<double>(),
                                             shape_size(inputs[0]->get_shape()));
        break;
    case element::Type_t::f32:
        runtime::reference::softsign<float>(inputs[0]->get_data_ptr<float>(),
                                            outputs[0]->get_data_ptr<float>(),
                                            shape_size(inputs[0]->get_shape()));
        break;
    case element::Type_t::f16:
        runtime::reference::softsign<float16>(inputs[0]->get_data_ptr<float16>(),
                                              outputs[0]->get_data_ptr<float16>(),
                                              shape_size(inputs[0]->get_shape()));
        break;
    case element::Type_t::bf16:
        runtime::reference::softsign<bfloat16>(inputs[0]->get_data_ptr<bfloat16>(),
                                               outputs[0]->get_data_ptr<bfloat16>(),
                                               shape_size(inputs[0]->get_shape()));
        break;
    default:
        return false;
    }
    return true;
}

template <typename Data_t, typename Index_t, typename Count_t>
void execute_unique(const HostTensorVector& outputs,
                    const HostTensorVector& inputs,
                    const shared_ptr<op::v10::Unique>& op) {
    const auto maybe_extract_axis = [&op]() {
        std::unique_ptr<int64_t> axis;
        if (op->get_input_size() == 2 && ov::op::util::is_constant(op->input_value(1).get_node())) {
            const auto axis_constant =
                std::dynamic_pointer_cast<op::v0::Constant>(op->input_value(1).get_node_shared_ptr());
            const auto axis_vec = axis_constant->cast_vector<int64_t>();
            axis = std::unique_ptr<int64_t>(new int64_t{axis_vec.at(0)});
        }
        return axis;
    };

    const auto unique_elements =
        runtime::reference::find_unique_elements<Data_t, Index_t, Count_t>(inputs[0]->get_data_ptr<Data_t>(),
                                                                           inputs[0]->get_shape(),
                                                                           maybe_extract_axis(),
                                                                           op->get_sorted());
    const auto tensor_shapes =
        runtime::reference::make_tensor_shapes(unique_elements, inputs[0]->get_shape(), maybe_extract_axis());

    auto& out_unique_elements = outputs[0];
    auto& out_indices = outputs[1];
    auto& out_rev_indices = outputs[2];
    auto& out_counts = outputs[3];

    out_unique_elements->set_shape(std::get<0>(tensor_shapes));
    out_indices->set_shape(std::get<1>(tensor_shapes));
    out_rev_indices->set_shape(std::get<2>(tensor_shapes));
    out_counts->set_shape(std::get<1>(tensor_shapes));

    runtime::reference::unique(out_unique_elements->get_data_ptr<Data_t>(),
                               out_indices->get_data_ptr<Index_t>(),
                               out_rev_indices->get_data_ptr<Index_t>(),
                               out_counts->get_data_ptr<Count_t>(),
                               inputs[0]->get_data_ptr<Data_t>(),
                               inputs[0]->get_shape(),
                               std::get<0>(tensor_shapes),
                               unique_elements);
}

template <element::Type_t Data_ET>
bool evaluate(const shared_ptr<op::v10::Unique>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using Data_t = typename element_type_traits<Data_ET>::value_type;
    if (op->get_index_element_type() == element::i32 && op->get_count_element_type() == element::i32) {
        execute_unique<Data_t, int32_t, int32_t>(outputs, inputs, op);
    } else if (op->get_index_element_type() == element::i64 && op->get_count_element_type() == element::i64) {
        execute_unique<Data_t, int64_t, int64_t>(outputs, inputs, op);
    } else if (op->get_index_element_type() == element::i32 && op->get_count_element_type() == element::i64) {
        execute_unique<Data_t, int32_t, int64_t>(outputs, inputs, op);
    } else if (op->get_index_element_type() == element::i64 && op->get_count_element_type() == element::i32) {
        execute_unique<Data_t, int64_t, int32_t>(outputs, inputs, op);
    } else {
        return false;
    }

    return true;
}

template <typename T>
bool evaluate_node(std::shared_ptr<Node> node, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<op::v1::Select>(node) || ov::is_type<op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case element::Type_t::boolean:
        return evaluate<element::Type_t::boolean>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::bf16:
        return evaluate<element::Type_t::bf16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::f16:
        return evaluate<element::Type_t::f16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::f64:
        return evaluate<element::Type_t::f64>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::f32:
        return evaluate<element::Type_t::f32>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::i4:
        return evaluate<element::Type_t::i4>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::i8:
        return evaluate<element::Type_t::i8>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::i16:
        return evaluate<element::Type_t::i16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::i32:
        return evaluate<element::Type_t::i32>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::i64:
        return evaluate<element::Type_t::i64>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::u1:
        return evaluate<element::Type_t::u1>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::u4:
        return evaluate<element::Type_t::u4>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::u8:
        return evaluate<element::Type_t::u8>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::u16:
        return evaluate<element::Type_t::u16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::u32:
        return evaluate<element::Type_t::u32>(ov::as_type_ptr<T>(node), outputs, inputs);
    case element::Type_t::u64:
        return evaluate<element::Type_t::u64>(ov::as_type_ptr<T>(node), outputs, inputs);
    default:
        throw ngraph_error(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                           std::string("in evaluate_node()"));
    }
}
}  // namespace

runtime::interpreter::EvaluatorsMap& runtime::interpreter::get_evaluators_map() {
    static runtime::interpreter::EvaluatorsMap evaluatorsMap{
#define _OPENVINO_OP_REG(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), evaluate_node<NAMESPACE::NAME>},

#include "opset_int_tbl.hpp"

#undef _OPENVINO_OP_REG
    };
    return evaluatorsMap;
}
