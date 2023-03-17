// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

#include "ops/abs.hpp"
#include "ops/adaptive_avg_pool.hpp"
#include "ops/adaptive_max_pool.hpp"
#include "ops/avg_pool.hpp"
#include "ops/batch_norm.hpp"
#include "ops/binary_convolution.hpp"
#include "ops/bucketize.hpp"
#include "ops/ceiling.hpp"
#include "ops/convert.hpp"
#include "ops/convert_color_nv12.hpp"
#include "ops/convolution.hpp"
#include "ops/ctc_greedy_decoder.hpp"
#include "ops/ctc_greedy_decoder_seq_len.hpp"
#include "ops/ctc_loss.hpp"
#include "ops/cum_sum.hpp"
#include "ops/deformable_convolution.hpp"
#include "ops/deformable_psroi_pooling.hpp"
#include "ops/detection_output.hpp"
#include "ops/einsum.hpp"
#include "ops/elu.hpp"
#include "ops/embedding_bag_offsets_sum.hpp"
#include "ops/embedding_bag_packed_sum.hpp"
#include "ops/embedding_segments_sum.hpp"
#include "ops/equal.hpp"
#include "ops/exp.hpp"
#include "ops/experimental_detectron_detection_output.hpp"
#include "ops/experimental_detectron_prior_grid_generator.hpp"
#include "ops/experimental_detectron_proposal_single_image.hpp"
#include "ops/experimental_detectron_roi_feature_extractor.hpp"
#include "ops/experimental_detectron_topk_rois.hpp"
#include "ops/extract_image_patches.hpp"
#include "ops/fft.hpp"
#include "ops/gather_elements.hpp"
#include "ops/gather_nd.hpp"
#include "ops/gather_tree.hpp"
#include "ops/gather.hpp"
#include "ops/gelu.hpp"
#include "ops/generate_proposal.hpp"
#include "ops/greater.hpp"
#include "ops/grid_sample.hpp"
#include "ops/grn.hpp"
#include "ops/group_convolution_backprop_data.hpp"
#include "ops/group_convolution.hpp"
#include "ops/gru_cell.hpp"
#include "ops/hard_sigmoid.hpp"
#include "ops/if.hpp"
#include "ops/interpolate.hpp"
#include "ops/irdft.hpp"
#include "ops/is_finite.hpp"
#include "ops/is_inf.hpp"
#include "ops/is_nan.hpp"
#include "ops/log_softmax.hpp"
#include "ops/log.hpp"
#include "ops/lrn.hpp"
#include "ops/lstm_cell.hpp"
#include "ops/matrix_nms.hpp"
#include "ops/mod.hpp"
#include "ops/multiclass_nms.hpp"
#include "ops/mvn.hpp"
#include "ops/non_max_suppression.hpp"
#include "ops/normalize_l2.hpp"
#include "ops/pad.hpp"
#include "ops/prelu.hpp"
#include "ops/proposal.hpp"
#include "ops/psroi_pooling.hpp"
#include "ops/rdft.hpp"
#include "ops/region_yolo.hpp"
#include "ops/relu.hpp"
#include "ops/reorg_yolo.hpp"
#include "ops/reverse_sequence.hpp"
#include "ops/rnn_cell.hpp"
#include "ops/roi_align.hpp"
#include "ops/roi_pooling.hpp"
#include "ops/roll.hpp"
#include "ops/scatter_nd_update.hpp"
#include "ops/selu.hpp"
#include "ops/sequences.hpp"
#include "ops/sigmoid.hpp"
#include "ops/sign.hpp"
#include "ops/softsign.hpp"
#include "ops/squared_difference.hpp"
#include "ops/tanh.hpp"
#include "ops/tensor_iterator.hpp"
#include "ops/unique.hpp"

std::vector<float> get_floats(const std::shared_ptr<ngraph::HostTensor>& input, const ngraph::Shape& shape) {
    size_t input_size = ngraph::shape_size(shape);
    std::vector<float> result(input_size);

    switch (input->get_element_type()) {
    case ngraph::element::Type_t::bf16: {
        ngraph::bfloat16* p = input->get_data_ptr<ngraph::bfloat16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case ngraph::element::Type_t::f16: {
        ngraph::float16* p = input->get_data_ptr<ngraph::float16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case ngraph::element::Type_t::f32: {
        float* p = input->get_data_ptr<float>();
        memcpy(result.data(), p, input_size * sizeof(float));
    } break;
    default:
        throw std::runtime_error("Unsupported data type.");
        break;
    }

    return result;
}

std::vector<int64_t> get_integers(const std::shared_ptr<ngraph::HostTensor>& input, const ngraph::Shape& shape) {
    size_t input_size = ngraph::shape_size(shape);
    std::vector<int64_t> result(input_size);

    switch (input->get_element_type()) {
    case ngraph::element::Type_t::i8: {
        auto p = input->get_data_ptr<int8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ngraph::element::Type_t::i16: {
        auto p = input->get_data_ptr<int16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ngraph::element::Type_t::i32: {
        auto p = input->get_data_ptr<int32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ngraph::element::Type_t::i64: {
        auto p = input->get_data_ptr<int64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ngraph::element::Type_t::u8: {
        auto p = input->get_data_ptr<uint8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ngraph::element::Type_t::u16: {
        auto p = input->get_data_ptr<uint16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ngraph::element::Type_t::u32: {
        auto p = input->get_data_ptr<uint32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case ngraph::element::Type_t::u64: {
        auto p = input->get_data_ptr<uint64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    default:
        throw std::runtime_error("Unsupported data type in op NonMaxSuppression-5");
        break;
    }

    return result;
}

namespace {
template <ngraph::element::Type_t ET>
bool evaluate(std::shared_ptr<ngraph::Node> op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    return false;
}

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v3::Assign>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    outputs[0]->set_unary(inputs[0]);
    void* input = inputs[0]->get_data_ptr();
    outputs[0]->write(input, outputs[0]->get_size_in_bytes());
    return true;
}

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v3::ReadValue>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    outputs[0]->set_unary(inputs[0]);
    void* input = inputs[0]->get_data_ptr();
    outputs[0]->write(input, outputs[0]->get_size_in_bytes());
    return true;
}

template <typename T>
bool evaluate_node(std::shared_ptr<ngraph::Node> node,
                   const ngraph::HostTensorVector& outputs,
                   const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<T>(node), outputs, inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<T>(node), outputs, inputs);
    default:
        throw ngraph::ngraph_error(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                                   std::string("in evaluate_node()"));
    }
}
}  // namespace

ngraph::runtime::interpreter::EvaluatorsMap& ngraph::runtime::interpreter::get_evaluators_map() {
    static runtime::interpreter::EvaluatorsMap evaluatorsMap{
#define _OPENVINO_OP_REG(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), evaluate_node<NAMESPACE::NAME>},

#include "opset_int_tbl.hpp"

#undef _OPENVINO_OP_REG
    };
    return evaluatorsMap;
}
