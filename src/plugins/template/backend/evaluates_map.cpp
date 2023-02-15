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
#include <ngraph/runtime/reference/interpolate.hpp>
#include <ngraph/runtime/reference/irdft.hpp>
#include <ngraph/runtime/reference/is_finite.hpp>
#include <ngraph/runtime/reference/is_inf.hpp>
#include <ngraph/runtime/reference/is_nan.hpp>
#include <ngraph/runtime/reference/log.hpp>
#include <ngraph/runtime/reference/log_softmax.hpp>
#include <ngraph/runtime/reference/lrn.hpp>
#include <ngraph/runtime/reference/lstm_cell.hpp>
#include <ngraph/runtime/reference/matrix_nms.hpp>
#include <ngraph/runtime/reference/mod.hpp>
#include <ngraph/runtime/reference/multiclass_nms.hpp>
#include <ngraph/runtime/reference/mvn.hpp>
#include <ngraph/runtime/reference/non_max_suppression.hpp>
#include <ngraph/runtime/reference/normalize_l2.hpp>
#include <ngraph/runtime/reference/pad.hpp>
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

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::MVN>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::mvn<T>(inputs[0]->get_data_ptr<ET>(),
                               outputs[0]->get_data_ptr<ET>(),
                               inputs[0]->get_shape(),
                               op->get_normalize_variance(),
                               op->get_reduction_axes(),
                               op->get_eps());
    return true;
}

namespace mvn_6_axes {
template <typename T>
AxisSet mvn_6_reduction_axes(const HostTensorPtr& axes_input, size_t rank) {
    T* a = axes_input->get_data_ptr<T>();
    auto v = std::vector<T>(a, a + axes_input->get_shape()[0]);
    std::vector<size_t> axes(v.size(), 0);
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i] < 0) {
            if (rank + v[i] < 0) {
                throw ngraph_error("Unexpected axis");
            }
            axes[i] = (size_t)(rank + v[i]);
        } else {
            axes[i] = (size_t)(v[i]);
        }
    }
    return AxisSet(axes);
}
}  // namespace mvn_6_axes

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v6::MVN>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    AxisSet reduction_axes;
    auto rank = inputs[0]->get_shape().size();
    if (inputs[1]->get_element_type() == element::i64) {
        reduction_axes = mvn_6_axes::mvn_6_reduction_axes<int64_t>(inputs[1], rank);
    } else if (inputs[1]->get_element_type() == element::i32) {
        reduction_axes = mvn_6_axes::mvn_6_reduction_axes<int32_t>(inputs[1], rank);
    } else {
        throw ngraph_error("Unexpected indices type");
    }
    runtime::reference::mvn_6<T>(inputs[0]->get_data_ptr<ET>(),
                                 outputs[0]->get_data_ptr<ET>(),
                                 inputs[0]->get_shape(),
                                 reduction_axes,
                                 op->get_normalize_variance(),
                                 op->get_eps(),
                                 op->get_eps_mode());
    return true;
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

std::vector<int64_t> get_integers(const std::shared_ptr<HostTensor>& input, const Shape& shape) {
    size_t input_size = shape_size(shape);
    std::vector<int64_t> result(input_size);

    switch (input->get_element_type()) {
    case element::Type_t::i8: {
        auto p = input->get_data_ptr<int8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::i16: {
        auto p = input->get_data_ptr<int16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::i32: {
        auto p = input->get_data_ptr<int32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::i64: {
        auto p = input->get_data_ptr<int64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u8: {
        auto p = input->get_data_ptr<uint8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u16: {
        auto p = input->get_data_ptr<uint16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u32: {
        auto p = input->get_data_ptr<uint32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u64: {
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

namespace nms_v5 {
using V5BoxEncoding = op::v5::NonMaxSuppression::BoxEncodingType;

struct InfoForNMS5 {
    int64_t max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
    float soft_nms_sigma;
    Shape out_shape;
    Shape boxes_shape;
    Shape scores_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    size_t out_shape_size;
    bool sort_result_descending;
    ngraph::element::Type output_type;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;

PartialShape infer_selected_indices_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                          int64_t max_output_boxes_per_class) {
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape result = {Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();

            result[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes * scores_ps[0].get_length();
        }
    }
    return result;
}

void normalize_corner(float* boxes, const Shape& boxes_shape) {
    size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float y1 = current_box[0];
        float x1 = current_box[1];
        float y2 = current_box[2];
        float x2 = current_box[3];

        float ymin = std::min(y1, y2);
        float ymax = std::max(y1, y2);
        float xmin = std::min(x1, x2);
        float xmax = std::max(x1, x2);

        current_box[0] = ymin;
        current_box[1] = xmin;
        current_box[2] = ymax;
        current_box[3] = xmax;
    }
}

void normalize_center(float* boxes, const Shape& boxes_shape) {
    size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float x_center = current_box[0];
        float y_center = current_box[1];
        float width = current_box[2];
        float height = current_box[3];

        float y1 = y_center - height / 2.0f;
        float x1 = x_center - width / 2.0f;
        float y2 = y_center + height / 2.0f;
        float x2 = x_center + width / 2.0f;

        current_box[0] = y1;
        current_box[1] = x1;
        current_box[2] = y2;
        current_box[3] = x2;
    }
}

void normalize_box_encoding(float* boxes, const Shape& boxes_shape, const V5BoxEncoding box_encoding) {
    if (box_encoding == V5BoxEncoding::CORNER) {
        normalize_corner(boxes, boxes_shape);
    } else {
        normalize_center(boxes, boxes_shape);
    }
}

std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes,
                                      const Shape& boxes_shape,
                                      const V5BoxEncoding box_encoding) {
    auto result = get_floats(boxes, boxes_shape);
    normalize_box_encoding(result.data(), boxes_shape, box_encoding);
    return result;
}

std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores, const Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

InfoForNMS5 get_info_for_nms5_eval(const std::shared_ptr<op::v5::NonMaxSuppression>& nms5,
                                   const std::vector<std::shared_ptr<HostTensor>>& inputs) {
    InfoForNMS5 result;

    result.max_output_boxes_per_class = inputs.size() > 2 ? get_integers(inputs[2], Shape({}))[0] : 0;
    result.iou_threshold = inputs.size() > 3 ? get_floats(inputs[3], Shape({}))[0] : 0.0f;
    result.score_threshold = inputs.size() > 4 ? get_floats(inputs[4], Shape({}))[0] : 0.0f;
    result.soft_nms_sigma = inputs.size() > 5 ? get_floats(inputs[5], Shape({}))[0] : 0.0f;

    auto selected_indices_shape = infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
    result.out_shape = selected_indices_shape.to_shape();

    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape, nms5->get_box_encoding());
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    result.out_shape_size = shape_size(result.out_shape);

    result.sort_result_descending = nms5->get_sort_result_descending();

    result.output_type = nms5->get_output_type();

    return result;
}
}  // namespace nms_v5

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v5::NonMaxSuppression>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    auto info = nms_v5::get_info_for_nms5_eval(op, inputs);

    std::vector<int64_t> selected_indices(info.out_shape_size);
    std::vector<float> selected_scores(info.out_shape_size);
    int64_t valid_outputs = 0;

    runtime::reference::non_max_suppression5(info.boxes_data.data(),
                                             info.boxes_shape,
                                             info.scores_data.data(),
                                             info.scores_shape,
                                             info.max_output_boxes_per_class,
                                             info.iou_threshold,
                                             info.score_threshold,
                                             info.soft_nms_sigma,
                                             selected_indices.data(),
                                             info.out_shape,
                                             selected_scores.data(),
                                             info.out_shape,
                                             &valid_outputs,
                                             info.sort_result_descending);

    auto selected_scores_type = (outputs.size() < 3) ? element::f32 : outputs[1]->get_element_type();

    runtime::reference::nms_postprocessing(outputs,
                                           info.output_type,
                                           selected_indices,
                                           selected_scores,
                                           valid_outputs,
                                           selected_scores_type);
    return true;
}

namespace nms_v9 {
using V9BoxEncoding = op::v9::NonMaxSuppression::BoxEncodingType;

struct InfoForNMS9 {
    int64_t max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
    float soft_nms_sigma;
    Shape out_shape;
    Shape boxes_shape;
    Shape scores_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    size_t out_shape_size;
    bool sort_result_descending;
    ngraph::element::Type output_type;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;

PartialShape infer_selected_indices_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                          int64_t max_output_boxes_per_class) {
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape result = {Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();

            result[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes * scores_ps[0].get_length();
        }
    }
    return result;
}

void normalize_corner(float* boxes, const Shape& boxes_shape) {
    size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float y1 = current_box[0];
        float x1 = current_box[1];
        float y2 = current_box[2];
        float x2 = current_box[3];

        float ymin = std::min(y1, y2);
        float ymax = std::max(y1, y2);
        float xmin = std::min(x1, x2);
        float xmax = std::max(x1, x2);

        current_box[0] = ymin;
        current_box[1] = xmin;
        current_box[2] = ymax;
        current_box[3] = xmax;
    }
}

void normalize_center(float* boxes, const Shape& boxes_shape) {
    size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float x_center = current_box[0];
        float y_center = current_box[1];
        float width = current_box[2];
        float height = current_box[3];

        float y1 = y_center - height / 2.0;
        float x1 = x_center - width / 2.0;
        float y2 = y_center + height / 2.0;
        float x2 = x_center + width / 2.0;

        current_box[0] = y1;
        current_box[1] = x1;
        current_box[2] = y2;
        current_box[3] = x2;
    }
}

void normalize_box_encoding(float* boxes, const Shape& boxes_shape, const V9BoxEncoding box_encoding) {
    if (box_encoding == V9BoxEncoding::CORNER) {
        normalize_corner(boxes, boxes_shape);
    } else {
        normalize_center(boxes, boxes_shape);
    }
}

std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes,
                                      const Shape& boxes_shape,
                                      const V9BoxEncoding box_encoding) {
    auto result = get_floats(boxes, boxes_shape);
    normalize_box_encoding(result.data(), boxes_shape, box_encoding);
    return result;
}

std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores, const Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

InfoForNMS9 get_info_for_nms9_eval(const std::shared_ptr<op::v9::NonMaxSuppression>& nms9,
                                   const std::vector<std::shared_ptr<HostTensor>>& inputs) {
    InfoForNMS9 result;

    result.max_output_boxes_per_class = inputs.size() > 2 ? get_integers(inputs[2], Shape({}))[0] : 0;
    result.iou_threshold = inputs.size() > 3 ? get_floats(inputs[3], Shape({}))[0] : 0.0f;
    result.score_threshold = inputs.size() > 4 ? get_floats(inputs[4], Shape({}))[0] : 0.0f;
    result.soft_nms_sigma = inputs.size() > 5 ? get_floats(inputs[5], Shape({}))[0] : 0.0f;

    auto selected_indices_shape = infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
    result.out_shape = selected_indices_shape.to_shape();

    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape, nms9->get_box_encoding());
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    result.out_shape_size = shape_size(result.out_shape);

    result.sort_result_descending = nms9->get_sort_result_descending();

    result.output_type = nms9->get_output_type();

    return result;
}
}  // namespace nms_v9

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v9::NonMaxSuppression>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    auto info = nms_v9::get_info_for_nms9_eval(op, inputs);

    std::vector<int64_t> selected_indices(info.out_shape_size);
    std::vector<float> selected_scores(info.out_shape_size);
    int64_t valid_outputs = 0;

    runtime::reference::non_max_suppression(info.boxes_data.data(),
                                            info.boxes_shape,
                                            info.scores_data.data(),
                                            info.scores_shape,
                                            info.max_output_boxes_per_class,
                                            info.iou_threshold,
                                            info.score_threshold,
                                            info.soft_nms_sigma,
                                            selected_indices.data(),
                                            info.out_shape,
                                            selected_scores.data(),
                                            info.out_shape,
                                            &valid_outputs,
                                            info.sort_result_descending);

    auto selected_scores_type = (outputs.size() < 3) ? element::f32 : outputs[1]->get_element_type();

    runtime::reference::nms_postprocessing(outputs,
                                           info.output_type,
                                           selected_indices,
                                           selected_scores,
                                           valid_outputs,
                                           selected_scores_type);
    return true;
}

namespace nms_v4 {
using V4BoxEncoding = op::v4::NonMaxSuppression::BoxEncodingType;

struct InfoForNMS4 {
    int64_t max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
    float soft_nms_sigma;
    Shape out_shape;
    Shape boxes_shape;
    Shape scores_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    size_t out_shape_size;
    bool sort_result_descending;
    ngraph::element::Type output_type;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;

PartialShape infer_selected_indices_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                          int64_t max_output_boxes_per_class) {
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape result = {Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();

            result[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes * scores_ps[0].get_length();
        }
    }
    return result;
}

void normalize_corner(float* boxes, const Shape& boxes_shape) {
    size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float y1 = current_box[0];
        float x1 = current_box[1];
        float y2 = current_box[2];
        float x2 = current_box[3];

        float ymin = std::min(y1, y2);
        float ymax = std::max(y1, y2);
        float xmin = std::min(x1, x2);
        float xmax = std::max(x1, x2);

        current_box[0] = ymin;
        current_box[1] = xmin;
        current_box[2] = ymax;
        current_box[3] = xmax;
    }
}

void normalize_center(float* boxes, const Shape& boxes_shape) {
    size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float x_center = current_box[0];
        float y_center = current_box[1];
        float width = current_box[2];
        float height = current_box[3];

        float y1 = y_center - height / 2.0f;
        float x1 = x_center - width / 2.0f;
        float y2 = y_center + height / 2.0f;
        float x2 = x_center + width / 2.0f;

        current_box[0] = y1;
        current_box[1] = x1;
        current_box[2] = y2;
        current_box[3] = x2;
    }
}

void normalize_box_encoding(float* boxes, const Shape& boxes_shape, const V4BoxEncoding box_encoding) {
    if (box_encoding == V4BoxEncoding::CORNER) {
        normalize_corner(boxes, boxes_shape);
    } else {
        normalize_center(boxes, boxes_shape);
    }
}

std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes,
                                      const Shape& boxes_shape,
                                      const V4BoxEncoding box_encoding) {
    auto result = get_floats(boxes, boxes_shape);
    normalize_box_encoding(result.data(), boxes_shape, box_encoding);
    return result;
}

std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores, const Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

InfoForNMS4 get_info_for_nms4_eval(const std::shared_ptr<op::v4::NonMaxSuppression>& nms4,
                                   const std::vector<std::shared_ptr<HostTensor>>& inputs) {
    InfoForNMS4 result;

    result.max_output_boxes_per_class = inputs.size() > 2 ? get_integers(inputs[2], Shape({}))[0] : 0;
    result.iou_threshold = inputs.size() > 3 ? get_floats(inputs[3], Shape({}))[0] : 0.0f;
    result.score_threshold = inputs.size() > 4 ? get_floats(inputs[4], Shape({}))[0] : 0.0f;
    result.soft_nms_sigma = inputs.size() > 5 ? get_floats(inputs[5], Shape({}))[0] : 0.0f;

    auto selected_indices_shape = infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
    result.out_shape = selected_indices_shape.to_shape();

    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape, nms4->get_box_encoding());
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    result.out_shape_size = shape_size(result.out_shape);

    result.sort_result_descending = nms4->get_sort_result_descending();

    result.output_type = nms4->get_output_type();

    return result;
}
}  // namespace nms_v4

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v4::NonMaxSuppression>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    auto info = nms_v4::get_info_for_nms4_eval(op, inputs);

    std::vector<int64_t> selected_indices(info.out_shape_size);
    std::vector<float> selected_scores(info.out_shape_size);
    int64_t valid_outputs = 0;

    runtime::reference::non_max_suppression(info.boxes_data.data(),
                                            info.boxes_shape,
                                            info.scores_data.data(),
                                            info.scores_shape,
                                            info.max_output_boxes_per_class,
                                            info.iou_threshold,
                                            info.score_threshold,
                                            info.soft_nms_sigma,
                                            selected_indices.data(),
                                            info.out_shape,
                                            selected_scores.data(),
                                            info.out_shape,
                                            &valid_outputs,
                                            info.sort_result_descending);

    auto selected_scores_type = (inputs.size() < 4) ? element::f32 : inputs[3]->get_element_type();

    runtime::reference::nms_postprocessing(outputs,
                                           info.output_type,
                                           selected_indices,
                                           selected_scores,
                                           valid_outputs,
                                           selected_scores_type);
    return true;
}

namespace nms_v3 {
using V3BoxEncoding = op::v3::NonMaxSuppression::BoxEncodingType;

struct InfoForNMS3 {
    int64_t max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
    float soft_nms_sigma;
    Shape out_shape;
    Shape boxes_shape;
    Shape scores_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    size_t out_shape_size;
    bool sort_result_descending;
    ngraph::element::Type output_type;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;

PartialShape infer_selected_indices_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                          int64_t max_output_boxes_per_class) {
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape result = {Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();

            result[0] = std::min(num_boxes, max_output_boxes_per_class * num_classes);
        }
    }
    return result;
}

void normalize_corner(float* boxes, const Shape& boxes_shape) {
    size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float y1 = current_box[0];
        float x1 = current_box[1];
        float y2 = current_box[2];
        float x2 = current_box[3];

        float ymin = std::min(y1, y2);
        float ymax = std::max(y1, y2);
        float xmin = std::min(x1, x2);
        float xmax = std::max(x1, x2);

        current_box[0] = ymin;
        current_box[1] = xmin;
        current_box[2] = ymax;
        current_box[3] = xmax;
    }
}

void normalize_center(float* boxes, const Shape& boxes_shape) {
    size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float x_center = current_box[0];
        float y_center = current_box[1];
        float width = current_box[2];
        float height = current_box[3];

        float y1 = y_center - height / 2.0f;
        float x1 = x_center - width / 2.0f;
        float y2 = y_center + height / 2.0f;
        float x2 = x_center + width / 2.0f;

        current_box[0] = y1;
        current_box[1] = x1;
        current_box[2] = y2;
        current_box[3] = x2;
    }
}

void normalize_box_encoding(float* boxes, const Shape& boxes_shape, const V3BoxEncoding box_encoding) {
    if (box_encoding == V3BoxEncoding::CORNER) {
        normalize_corner(boxes, boxes_shape);
    } else {
        normalize_center(boxes, boxes_shape);
    }
}

std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes,
                                      const Shape& boxes_shape,
                                      const V3BoxEncoding box_encoding) {
    auto result = get_floats(boxes, boxes_shape);
    normalize_box_encoding(result.data(), boxes_shape, box_encoding);
    return result;
}

std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores, const Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

InfoForNMS3 get_info_for_nms3_eval(const std::shared_ptr<op::v3::NonMaxSuppression>& nms3,
                                   const std::vector<std::shared_ptr<HostTensor>>& inputs) {
    InfoForNMS3 result;

    result.max_output_boxes_per_class = inputs.size() > 2 ? get_integers(inputs[2], Shape({}))[0] : 0;
    result.iou_threshold = inputs.size() > 3 ? get_floats(inputs[3], Shape({}))[0] : 0.0f;
    result.score_threshold = inputs.size() > 4 ? get_floats(inputs[4], Shape({}))[0] : 0.0f;
    result.soft_nms_sigma = inputs.size() > 5 ? get_floats(inputs[5], Shape({}))[0] : 0.0f;

    auto selected_indices_shape = infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
    result.out_shape = selected_indices_shape.to_shape();

    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape, nms3->get_box_encoding());
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    result.out_shape_size = shape_size(result.out_shape);

    result.sort_result_descending = nms3->get_sort_result_descending();

    result.output_type = nms3->get_output_type();

    return result;
}
}  // namespace nms_v3

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v3::NonMaxSuppression>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    auto info = nms_v3::get_info_for_nms3_eval(op, inputs);

    std::vector<int64_t> selected_indices(info.out_shape_size);
    std::vector<float> selected_scores(info.out_shape_size);
    int64_t valid_outputs = 0;

    runtime::reference::non_max_suppression(info.boxes_data.data(),
                                            info.boxes_shape,
                                            info.scores_data.data(),
                                            info.scores_shape,
                                            info.max_output_boxes_per_class,
                                            info.iou_threshold,
                                            info.score_threshold,
                                            info.soft_nms_sigma,
                                            selected_indices.data(),
                                            info.out_shape,
                                            selected_scores.data(),
                                            info.out_shape,
                                            &valid_outputs,
                                            info.sort_result_descending);

    auto selected_scores_type = (inputs.size() < 4) ? element::f32 : inputs[3]->get_element_type();

    runtime::reference::nms_postprocessing(outputs,
                                           info.output_type,
                                           selected_indices,
                                           selected_scores,
                                           valid_outputs,
                                           selected_scores_type);
    return true;
}

namespace nms_v1 {
using V1BoxEncoding = op::v1::NonMaxSuppression::BoxEncodingType;

struct InfoForNMS1 {
    int64_t max_output_boxes_per_class;
    float iou_threshold;
    float score_threshold;
    float soft_nms_sigma;
    Shape out_shape;
    Shape boxes_shape;
    Shape scores_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    size_t out_shape_size;
    bool sort_result_descending;
    ngraph::element::Type output_type;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;

PartialShape infer_selected_indices_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                          int64_t max_output_boxes_per_class) {
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape result = {Dimension::dynamic(), 3};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();

            result[0] = std::min(num_boxes, max_output_boxes_per_class * num_classes);
        }
    }
    return result;
}

void normalize_corner(float* boxes, const Shape& boxes_shape) {
    size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float y1 = current_box[0];
        float x1 = current_box[1];
        float y2 = current_box[2];
        float x2 = current_box[3];

        float ymin = std::min(y1, y2);
        float ymax = std::max(y1, y2);
        float xmin = std::min(x1, x2);
        float xmax = std::max(x1, x2);

        current_box[0] = ymin;
        current_box[1] = xmin;
        current_box[2] = ymax;
        current_box[3] = xmax;
    }
}

void normalize_center(float* boxes, const Shape& boxes_shape) {
    size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
    for (size_t i = 0; i < total_num_of_boxes; ++i) {
        float* current_box = boxes + 4 * i;

        float x_center = current_box[0];
        float y_center = current_box[1];
        float width = current_box[2];
        float height = current_box[3];

        float y1 = y_center - height / 2.0f;
        float x1 = x_center - width / 2.0f;
        float y2 = y_center + height / 2.0f;
        float x2 = x_center + width / 2.0f;

        current_box[0] = y1;
        current_box[1] = x1;
        current_box[2] = y2;
        current_box[3] = x2;
    }
}

void normalize_box_encoding(float* boxes, const Shape& boxes_shape, const V1BoxEncoding box_encoding) {
    if (box_encoding == V1BoxEncoding::CORNER) {
        normalize_corner(boxes, boxes_shape);
    } else {
        normalize_center(boxes, boxes_shape);
    }
}

std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes,
                                      const Shape& boxes_shape,
                                      const V1BoxEncoding box_encoding) {
    auto result = get_floats(boxes, boxes_shape);
    normalize_box_encoding(result.data(), boxes_shape, box_encoding);
    return result;
}

std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores, const Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

InfoForNMS1 get_info_for_nms1_eval(const std::shared_ptr<op::v1::NonMaxSuppression>& nms1,
                                   const std::vector<std::shared_ptr<HostTensor>>& inputs) {
    InfoForNMS1 result;

    result.max_output_boxes_per_class = inputs.size() > 2 ? get_integers(inputs[2], Shape({}))[0] : 0;
    result.iou_threshold = inputs.size() > 3 ? get_floats(inputs[3], Shape({}))[0] : 0.0f;
    result.score_threshold = inputs.size() > 4 ? get_floats(inputs[4], Shape({}))[0] : 0.0f;
    result.soft_nms_sigma = inputs.size() > 5 ? get_floats(inputs[5], Shape({}))[0] : 0.0f;

    auto selected_indices_shape = infer_selected_indices_shape(inputs, result.max_output_boxes_per_class);
    result.out_shape = selected_indices_shape.to_shape();

    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape, nms1->get_box_encoding());
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    result.out_shape_size = shape_size(result.out_shape);

    result.sort_result_descending = nms1->get_sort_result_descending();

    result.output_type = ov::element::i64;

    return result;
}
}  // namespace nms_v1

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v1::NonMaxSuppression>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    auto info = nms_v1::get_info_for_nms1_eval(op, inputs);

    std::vector<int64_t> selected_indices(info.out_shape_size);
    std::vector<float> selected_scores(info.out_shape_size);
    int64_t valid_outputs = 0;

    runtime::reference::non_max_suppression(info.boxes_data.data(),
                                            info.boxes_shape,
                                            info.scores_data.data(),
                                            info.scores_shape,
                                            info.max_output_boxes_per_class,
                                            info.iou_threshold,
                                            info.score_threshold,
                                            info.soft_nms_sigma,
                                            selected_indices.data(),
                                            info.out_shape,
                                            selected_scores.data(),
                                            info.out_shape,
                                            &valid_outputs,
                                            info.sort_result_descending);

    auto selected_scores_type = (inputs.size() < 4) ? element::f32 : inputs[3]->get_element_type();

    runtime::reference::nms_postprocessing(outputs,
                                           info.output_type,
                                           selected_indices,
                                           selected_scores,
                                           valid_outputs,
                                           selected_scores_type);
    return true;
}

namespace matrix_nms_v8 {
using SortResultType = op::v8::MatrixNms::SortResultType;
struct InfoForNMS {
    Shape selected_outputs_shape;
    Shape selected_indices_shape;
    Shape boxes_shape;
    Shape scores_shape;
    std::vector<float> boxes_data;
    std::vector<float> scores_data;
    size_t selected_outputs_shape_size;
    size_t selected_indices_shape_size;
};

constexpr size_t boxes_port = 0;
constexpr size_t scores_port = 1;

PartialShape infer_selected_outputs_shape(const std::vector<std::shared_ptr<HostTensor>>& inputs,
                                          int nms_top_k,
                                          int keep_top_k) {
    const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
    const auto scores_ps = inputs[scores_port]->get_partial_shape();

    PartialShape result = {Dimension::dynamic(), 6};

    if (boxes_ps.rank().is_static() && scores_ps.rank().is_static()) {
        const auto num_boxes_boxes = boxes_ps[1];
        if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static()) {
            const auto num_boxes = num_boxes_boxes.get_length();
            const auto num_classes = scores_ps[1].get_length();
            int64_t max_output_boxes_per_class = 0;
            if (nms_top_k >= 0)
                max_output_boxes_per_class = std::min(num_boxes, (int64_t)nms_top_k);
            else
                max_output_boxes_per_class = num_boxes;

            auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
            if (keep_top_k >= 0)
                max_output_boxes_per_batch = std::min(max_output_boxes_per_batch, (int64_t)keep_top_k);

            result[0] = max_output_boxes_per_batch * scores_ps[0].get_length();
        }
    }

    return result;
}

std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes, const Shape& boxes_shape) {
    auto result = get_floats(boxes, boxes_shape);
    return result;
}

std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores, const Shape& scores_shape) {
    auto result = get_floats(scores, scores_shape);
    return result;
}

InfoForNMS get_info_for_nms_eval(const std::shared_ptr<op::v8::MatrixNms>& nms,
                                 const std::vector<std::shared_ptr<HostTensor>>& inputs) {
    InfoForNMS result;
    const auto& nms_attrs = nms->get_attrs();
    const auto nms_top_k = nms_attrs.nms_top_k;
    const auto keep_top_k = nms_attrs.keep_top_k;

    auto selected_outputs_shape = infer_selected_outputs_shape(inputs, nms_top_k, keep_top_k);
    result.selected_outputs_shape = selected_outputs_shape.to_shape();
    result.selected_indices_shape = {result.selected_outputs_shape[0], 1};

    result.boxes_shape = inputs[boxes_port]->get_shape();
    result.scores_shape = inputs[scores_port]->get_shape();

    result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape);
    result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

    result.selected_outputs_shape_size = shape_size(result.selected_outputs_shape);
    result.selected_indices_shape_size = shape_size(result.selected_indices_shape);

    return result;
}
}  // namespace matrix_nms_v8

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v8::MatrixNms>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    auto info = matrix_nms_v8::get_info_for_nms_eval(op, inputs);

    std::vector<float> selected_outputs(info.selected_outputs_shape_size);
    std::vector<int64_t> selected_indices(info.selected_indices_shape_size);
    std::vector<int64_t> valid_outputs(info.boxes_shape[0]);

    runtime::reference::matrix_nms(info.boxes_data.data(),
                                   info.boxes_shape,
                                   info.scores_data.data(),
                                   info.scores_shape,
                                   op->get_attrs(),
                                   selected_outputs.data(),
                                   info.selected_outputs_shape,
                                   selected_indices.data(),
                                   info.selected_indices_shape,
                                   valid_outputs.data());

    void* pscores = nullptr;
    void* pselected_num = nullptr;
    void* prois;
    size_t num_selected = static_cast<size_t>(std::accumulate(valid_outputs.begin(), valid_outputs.end(), size_t(0)));

    outputs[0]->set_shape({num_selected, 6});
    prois = outputs[0]->get_data_ptr();

    if (outputs.size() >= 2) {
        outputs[1]->set_shape({num_selected, 1});
        pscores = outputs[1]->get_data_ptr();
    }
    if (outputs.size() >= 3) {
        pselected_num = outputs[2]->get_data_ptr();
    }

    runtime::reference::nms_common::nms_common_postprocessing(prois,
                                                              pscores,
                                                              pselected_num,
                                                              op->get_attrs().output_type,
                                                              selected_outputs,
                                                              selected_indices,
                                                              valid_outputs,
                                                              op->get_input_element_type(0));
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v8::MulticlassNms>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    auto info = runtime::reference::multiclass_nms_impl::get_info_for_nms_eval(op, inputs);

    std::vector<float> selected_outputs(info.selected_outputs_shape_size);
    std::vector<int64_t> selected_indices(info.selected_indices_shape_size);
    std::vector<int64_t> valid_outputs(info.selected_numrois_shape_size);

    runtime::reference::multiclass_nms(info.boxes_data.data(),
                                       info.boxes_shape,
                                       info.scores_data.data(),
                                       info.scores_shape,
                                       nullptr,
                                       Shape(),  // won't be used
                                       op->get_attrs(),
                                       selected_outputs.data(),
                                       info.selected_outputs_shape,
                                       selected_indices.data(),
                                       info.selected_indices_shape,
                                       valid_outputs.data());

    void* pscores = nullptr;
    void* pselected_num = nullptr;
    void* prois;
    size_t num_selected = static_cast<size_t>(std::accumulate(valid_outputs.begin(), valid_outputs.end(), int64_t(0)));

    outputs[0]->set_shape({num_selected, 6});
    prois = outputs[0]->get_data_ptr();

    if (outputs.size() >= 2) {
        outputs[1]->set_shape({num_selected, 1});
        pscores = outputs[1]->get_data_ptr();
    }
    if (outputs.size() >= 3) {
        pselected_num = outputs[2]->get_data_ptr();
    }

    runtime::reference::nms_common::nms_common_postprocessing(prois,
                                                              pscores,
                                                              pselected_num,
                                                              op->get_attrs().output_type,
                                                              selected_outputs,
                                                              selected_indices,
                                                              valid_outputs,
                                                              op->get_input_element_type(0));

    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v9::MulticlassNms>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    auto info = runtime::reference::multiclass_nms_impl::get_info_for_nms_eval(op, inputs);

    std::vector<float> selected_outputs(info.selected_outputs_shape_size);
    std::vector<int64_t> selected_indices(info.selected_indices_shape_size);
    std::vector<int64_t> valid_outputs(info.selected_numrois_shape_size);

    runtime::reference::multiclass_nms(info.boxes_data.data(),
                                       info.boxes_shape,
                                       info.scores_data.data(),
                                       info.scores_shape,
                                       info.roisnum_data.data(),
                                       info.roisnum_shape,
                                       op->get_attrs(),
                                       selected_outputs.data(),
                                       info.selected_outputs_shape,
                                       selected_indices.data(),
                                       info.selected_indices_shape,
                                       valid_outputs.data());

    void* pscores = nullptr;
    void* pselected_num = nullptr;
    void* prois;
    size_t num_selected = static_cast<size_t>(std::accumulate(valid_outputs.begin(), valid_outputs.end(), 0));

    outputs[0]->set_shape({num_selected, 6});
    prois = outputs[0]->get_data_ptr();

    if (outputs.size() >= 2) {
        outputs[1]->set_shape({num_selected, 1});
        pscores = outputs[1]->get_data_ptr();
    }
    if (outputs.size() >= 3) {
        pselected_num = outputs[2]->get_data_ptr();
    }

    runtime::reference::nms_common::nms_common_postprocessing(prois,
                                                              pscores,
                                                              pselected_num,
                                                              op->get_attrs().output_type,
                                                              selected_outputs,
                                                              selected_indices,
                                                              valid_outputs,
                                                              op->get_input_element_type(0));

    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::LRN>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
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
bool evaluate(const shared_ptr<op::v1::Mod>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::mod<T>(inputs[0]->get_data_ptr<T>(),
                               inputs[1]->get_data_ptr<T>(),
                               outputs[0]->get_data_ptr<T>(),
                               inputs[0]->get_shape(),
                               inputs[1]->get_shape(),
                               op->get_autob());
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

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v0::Log>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::log<T>(inputs[0]->get_data_ptr<T>(),
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
bool evaluate(const shared_ptr<op::v1::Pad>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
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
bool evaluate(const shared_ptr<op::v0::NormalizeL2>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    runtime::reference::normalize_l2<T>(inputs[0]->get_data_ptr<const T>(),
                                        outputs[0]->get_data_ptr<T>(),
                                        op->get_input_shape(0),
                                        op->get_reduction_axes(),
                                        op->get_eps(),
                                        op->get_eps_mode());
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
bool evaluate(const shared_ptr<op::v5::LogSoftmax>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    using T = typename element_type_traits<ET>::value_type;
    int64_t i_axis = op->get_axis();
    if (i_axis < 0) {
        i_axis += inputs[0]->get_partial_shape().rank().get_length();
    }
    runtime::reference::log_softmax<T>(inputs[0]->get_data_ptr<const T>(),
                                       outputs[0]->get_data_ptr<T>(),
                                       op->get_output_shape(0),
                                       AxisSet{(size_t)i_axis});
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
bool evaluate(const shared_ptr<op::v0::Interpolate>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    element::Type input_et = op->get_input_element_type(0);
    switch (input_et) {
    case element::Type_t::f64:
        ngraph::runtime::reference::interpolate<double>(inputs[0]->get_data_ptr<double>(),
                                                        op->get_input_partial_shape(0),
                                                        outputs[0]->get_data_ptr<double>(),
                                                        op->get_output_shape(0),
                                                        op->get_attrs());
        break;
    case element::Type_t::f32:
        ngraph::runtime::reference::interpolate<float>(inputs[0]->get_data_ptr<float>(),
                                                       op->get_input_partial_shape(0),
                                                       outputs[0]->get_data_ptr<float>(),
                                                       op->get_output_shape(0),
                                                       op->get_attrs());
        break;
    case element::Type_t::f16:
        ngraph::runtime::reference::interpolate<float16>(inputs[0]->get_data_ptr<float16>(),
                                                         op->get_input_partial_shape(0),
                                                         outputs[0]->get_data_ptr<float16>(),
                                                         op->get_output_shape(0),
                                                         op->get_attrs());
        break;
    case element::Type_t::bf16:
        ngraph::runtime::reference::interpolate<bfloat16>(inputs[0]->get_data_ptr<bfloat16>(),
                                                          op->get_input_partial_shape(0),
                                                          outputs[0]->get_data_ptr<bfloat16>(),
                                                          op->get_output_shape(0),
                                                          op->get_attrs());
        break;
    default:;
    }
    return true;
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

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v10::IsFinite>& op,
              const HostTensorVector& outputs,
              const HostTensorVector& inputs) {
    element::Type input_et = op->get_input_element_type(0);
    switch (input_et) {
    case element::Type_t::f64:
        ngraph::runtime::reference::is_finite<double>(inputs[0]->get_data_ptr<double>(),
                                                      outputs[0]->get_data_ptr<element::Type_t::boolean>(),
                                                      shape_size(inputs[0]->get_shape()));
        break;
    case element::Type_t::f32:
        ngraph::runtime::reference::is_finite<float>(inputs[0]->get_data_ptr<float>(),
                                                     outputs[0]->get_data_ptr<element::Type_t::boolean>(),
                                                     shape_size(inputs[0]->get_shape()));
        break;
    case element::Type_t::f16:
        ngraph::runtime::reference::is_finite<float16>(inputs[0]->get_data_ptr<float16>(),
                                                       outputs[0]->get_data_ptr<element::Type_t::boolean>(),
                                                       shape_size(inputs[0]->get_shape()));
        break;
    case element::Type_t::bf16:
        ngraph::runtime::reference::is_finite<bfloat16>(inputs[0]->get_data_ptr<bfloat16>(),
                                                        outputs[0]->get_data_ptr<element::Type_t::boolean>(),
                                                        shape_size(inputs[0]->get_shape()));
        break;
    default:
        return false;
    }
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v10::IsInf>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    element::Type input_et = op->get_input_element_type(0);
    switch (input_et) {
    case element::Type_t::f64:
        ngraph::runtime::reference::is_inf(inputs[0]->get_data_ptr<double>(),
                                           outputs[0]->get_data_ptr<element::Type_t::boolean>(),
                                           shape_size(inputs[0]->get_shape()),
                                           op->get_attributes());
        break;
    case element::Type_t::f32:
        ngraph::runtime::reference::is_inf(inputs[0]->get_data_ptr<float>(),
                                           outputs[0]->get_data_ptr<element::Type_t::boolean>(),
                                           shape_size(inputs[0]->get_shape()),
                                           op->get_attributes());
        break;
    case element::Type_t::f16:
        ngraph::runtime::reference::is_inf(inputs[0]->get_data_ptr<float16>(),
                                           outputs[0]->get_data_ptr<element::Type_t::boolean>(),
                                           shape_size(inputs[0]->get_shape()),
                                           op->get_attributes());
        break;
    case element::Type_t::bf16:
        ngraph::runtime::reference::is_inf(inputs[0]->get_data_ptr<bfloat16>(),
                                           outputs[0]->get_data_ptr<element::Type_t::boolean>(),
                                           shape_size(inputs[0]->get_shape()),
                                           op->get_attributes());
        break;
    default:
        return false;
    }
    return true;
}

template <element::Type_t ET>
bool evaluate(const shared_ptr<op::v10::IsNaN>& op, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    element::Type input_et = op->get_input_element_type(0);
    switch (input_et) {
    case element::Type_t::f64:
        ngraph::runtime::reference::is_nan(inputs[0]->get_data_ptr<double>(),
                                           outputs[0]->get_data_ptr<element::Type_t::boolean>(),
                                           shape_size(inputs[0]->get_shape()));
        break;
    case element::Type_t::f32:
        ngraph::runtime::reference::is_nan(inputs[0]->get_data_ptr<float>(),
                                           outputs[0]->get_data_ptr<element::Type_t::boolean>(),
                                           shape_size(inputs[0]->get_shape()));
        break;
    case element::Type_t::f16:
        ngraph::runtime::reference::is_nan(inputs[0]->get_data_ptr<float16>(),
                                           outputs[0]->get_data_ptr<element::Type_t::boolean>(),
                                           shape_size(inputs[0]->get_shape()));
        break;
    case element::Type_t::bf16:
        ngraph::runtime::reference::is_nan(inputs[0]->get_data_ptr<bfloat16>(),
                                           outputs[0]->get_data_ptr<element::Type_t::boolean>(),
                                           shape_size(inputs[0]->get_shape()));
        break;
    default:
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
