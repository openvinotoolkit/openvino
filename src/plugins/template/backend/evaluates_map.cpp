// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "evaluates_map.hpp"

#include "backend.hpp"
// #include "ngraph/runtime/reference/abs.hpp"
// #include "ngraph/runtime/reference/adaptive_avg_pool.hpp"
// #include "ngraph/runtime/reference/adaptive_max_pool.hpp"
// #include "ngraph/runtime/reference/avg_pool.hpp"
// #include "ngraph/runtime/reference/batch_norm.hpp"
// #include "ngraph/runtime/reference/binary_convolution.hpp"
// #include "ngraph/runtime/reference/bucketize.hpp"
// #include "ngraph/runtime/reference/ceiling.hpp"
// #include "ngraph/runtime/reference/convert.hpp"
// #include "ngraph/runtime/reference/convert_color_nv12.hpp"
// #include "ngraph/runtime/reference/convolution.hpp"
// #include "ngraph/runtime/reference/convolution_backprop_data.hpp"
// #include "ngraph/runtime/reference/ctc_greedy_decoder.hpp"
// #include "ngraph/runtime/reference/ctc_greedy_decoder_seq_len.hpp"
// #include "ngraph/runtime/reference/ctc_loss.hpp"
// #include "ngraph/runtime/reference/cum_sum.hpp"
// #include "ngraph/runtime/reference/deformable_convolution.hpp"
// #include "ngraph/runtime/reference/deformable_psroi_pooling.hpp"
// #include "ngraph/runtime/reference/detection_output.hpp"
// #include "ngraph/runtime/reference/einsum.hpp"
// #include "ngraph/runtime/reference/elu.hpp"
// #include "ngraph/runtime/reference/embedding_bag_offsets_sum.hpp"
// #include "ngraph/runtime/reference/embedding_bag_packed_sum.hpp"
// #include "ngraph/runtime/reference/embedding_segments_sum.hpp"
// #include "ngraph/runtime/reference/equal.hpp"
// #include "ngraph/runtime/reference/exp.hpp"
// #include "ngraph/runtime/reference/experimental_detectron_detection_output.hpp"
// #include "ngraph/runtime/reference/experimental_detectron_prior_grid_generator.hpp"
// #include "ngraph/runtime/reference/experimental_detectron_proposal_single_image.hpp"
// #include "ngraph/runtime/reference/experimental_detectron_roi_feature_extractor.hpp"
// #include "ngraph/runtime/reference/experimental_detectron_topk_rois.hpp"
// #include "ngraph/runtime/reference/extract_image_patches.hpp"
// #include "ngraph/runtime/reference/fft.hpp"
// #include "ngraph/runtime/reference/gather.hpp"
// #include "ngraph/runtime/reference/gather_elements.hpp"
// #include "ngraph/runtime/reference/gather_nd.hpp"
// #include "ngraph/runtime/reference/gather_tree.hpp"
// #include "ngraph/runtime/reference/gelu.hpp"
// #include "ngraph/runtime/reference/generate_proposal.hpp"
// #include "ngraph/runtime/reference/greater.hpp"
// #include "ngraph/runtime/reference/grid_sample.hpp"
// #include "ngraph/runtime/reference/grn.hpp"
// #include "ngraph/runtime/reference/group_convolution.hpp"
// #include "ngraph/runtime/reference/group_convolution_backprop_data.hpp"
// #include "ngraph/runtime/reference/gru_cell.hpp"
// #include "ngraph/runtime/reference/hard_sigmoid.hpp"
// #include "ngraph/runtime/reference/if.hpp"
// #include "ngraph/runtime/reference/interpolate.hpp"
// #include "ngraph/runtime/reference/irdft.hpp"
// #include "ngraph/runtime/reference/is_finite.hpp"
// #include "ngraph/runtime/reference/is_inf.hpp"
// #include "ngraph/runtime/reference/is_nan.hpp"
// #include "ngraph/runtime/reference/log.hpp"
// #include "ngraph/runtime/reference/log_softmax.hpp"
// #include "ngraph/runtime/reference/lrn.hpp"
// #include "ngraph/runtime/reference/lstm_cell.hpp"
// #include "ngraph/runtime/reference/matrix_nms.hpp"
// #include "ngraph/runtime/reference/mod.hpp"
// #include "ngraph/runtime/reference/multiclass_nms.hpp"
// #include "ngraph/runtime/reference/mvn.hpp"
// #include "ngraph/runtime/reference/non_max_suppression.hpp"
// #include "ngraph/runtime/reference/normalize_l2.hpp"
// #include "ngraph/runtime/reference/pad.hpp"
// #include "ngraph/runtime/reference/prelu.hpp"
// #include "ngraph/runtime/reference/prior_box.hpp"
// #include "ngraph/runtime/reference/proposal.hpp"
// #include "ngraph/runtime/reference/psroi_pooling.hpp"
// #include "ngraph/runtime/reference/rdft.hpp"
// #include "ngraph/runtime/reference/region_yolo.hpp"
// #include "ngraph/runtime/reference/reorg_yolo.hpp"
// #include "ngraph/runtime/reference/reverse_sequence.hpp"
// #include "ngraph/runtime/reference/rnn_cell.hpp"
// #include "ngraph/runtime/reference/roi_align.hpp"
// #include "ngraph/runtime/reference/roi_pooling.hpp"
// #include "ngraph/runtime/reference/roll.hpp"
// #include "ngraph/runtime/reference/scatter_nd_update.hpp"
// #include "ngraph/runtime/reference/selu.hpp"
// #include "ngraph/runtime/reference/sequences.hpp"
// #include "ngraph/runtime/reference/sigmoid.hpp"
// #include "ngraph/runtime/reference/sign.hpp"
// #include "ngraph/runtime/reference/softsign.hpp"
// #include "ngraph/runtime/reference/squared_difference.hpp"
// #include "ngraph/runtime/reference/tanh.hpp"
// #include "ngraph/runtime/reference/tensor_iterator.hpp"
// #include "ngraph/runtime/reference/unique.hpp"
// #include "ngraph/runtime/reference/utils/nms_common.hpp"
// #include "ov_ops/augru_cell.hpp"
// #include "ov_ops/augru_sequence.hpp"
// #include "tensor_conversion_util.hpp"

namespace {
template <ngraph::element::Type_t ET>
bool evaluate(std::shared_ptr<ngraph::Node> op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    return false;
}

namespace {
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
}  // namespace

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v0::Relu>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    using T = typename ngraph::element_type_traits<ET>::value_type;
    ngraph::runtime::reference::relu<T>(inputs[0]->get_data_ptr<T>(),
                                        outputs[0]->get_data_ptr<T>(),
                                        ngraph::shape_size(inputs[0]->get_shape()));
    return true;
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
