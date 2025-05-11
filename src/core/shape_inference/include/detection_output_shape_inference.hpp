// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/detection_output.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace util {

template <typename T, typename V = typename std::iterator_traits<typename T::iterator>::value_type::value_type>
void compute_num_classes(const DetectionOutputBase* op,
                         const DetectionOutputBase::AttributesBase& attrs,
                         const std::vector<T>& input_shapes,
                         V& num_classes,
                         V& num_prior_boxes) {
    const T& box_logits_pshape = input_shapes[0];
    const T& class_preds_pshape = input_shapes[1];
    const T& proposals_pshape = input_shapes[2];
    T ad_class_preds_shape{};
    T ad_box_preds_shape{};
    bool have_five_inputs = false;

    NODE_VALIDATION_CHECK(op,
                          box_logits_pshape.rank().compatible(2),
                          "Box logits rank must be 2. Got ",
                          box_logits_pshape.rank().get_length());

    NODE_VALIDATION_CHECK(op,
                          class_preds_pshape.rank().compatible(2),
                          "Class predictions rank must be 2. Got ",
                          class_preds_pshape.rank().get_length());
    NODE_VALIDATION_CHECK(op,
                          proposals_pshape.rank().compatible(3),
                          "Proposals rank must be 3. Got ",
                          proposals_pshape.rank().get_length());
    if (input_shapes.size() == 5) {
        ad_class_preds_shape = input_shapes[3];
        NODE_VALIDATION_CHECK(op,
                              ad_class_preds_shape.rank().compatible(2),
                              "Additional class predictions rank must be 2. Got ",
                              ad_class_preds_shape.rank().get_length());
        ad_box_preds_shape = input_shapes[4];
        NODE_VALIDATION_CHECK(op,
                              ad_box_preds_shape.rank().compatible(2),
                              "Additional box predictions rank must be 2. Got ",
                              ad_box_preds_shape.rank().get_length());
        have_five_inputs = true;
    }

    int64_t prior_box_size = attrs.normalized ? 4 : 5;

    // try to deduce a number of prior boxes
    if (num_prior_boxes == 0 && proposals_pshape.rank().is_static() && proposals_pshape[2].is_static()) {
        NODE_VALIDATION_CHECK(op,
                              (proposals_pshape[2].get_length()) % prior_box_size == 0,
                              "Proposals' third dimension must be a multiply of prior_box_size (",
                              prior_box_size,
                              "). Current value is: ",
                              proposals_pshape[2].get_length(),
                              ".");
        num_prior_boxes = (proposals_pshape[2].get_length()) / prior_box_size;
        NODE_VALIDATION_CHECK(op,
                              num_prior_boxes > 0,
                              "A number of prior boxes must be greater zero. Got: ",
                              num_prior_boxes);
    }
    if (num_prior_boxes == 0 && have_five_inputs && ad_class_preds_shape.rank().is_static() &&
        ad_class_preds_shape[1].is_static()) {
        NODE_VALIDATION_CHECK(
            op,
            (ad_class_preds_shape[1].get_length()) % 2 == 0,
            "Additional class predictions second dimension must be a multiply of 2. Current value is: ",
            ad_class_preds_shape[1].get_length(),
            ".");
        num_prior_boxes = (ad_class_preds_shape[1].get_length()) / 2;
        NODE_VALIDATION_CHECK(op,
                              num_prior_boxes > 0,
                              "A number of prior boxes must be greater zero. Got: ",
                              num_prior_boxes);
    }

    // try to deduce a number of classes
    if (num_classes == 0 && num_prior_boxes > 0 && class_preds_pshape.rank().is_static() &&
        class_preds_pshape[1].is_static()) {
        NODE_VALIDATION_CHECK(op,
                              (class_preds_pshape[1].get_length()) % num_prior_boxes == 0,
                              "Class predictions second dimension must be a multiply of num_prior_boxes (",
                              num_prior_boxes,
                              "). Current value is: ",
                              class_preds_pshape[1].get_length(),
                              ".");
        num_classes = (class_preds_pshape[1].get_length()) / num_prior_boxes;
    }
    if (num_classes == 0 && num_prior_boxes > 0 && box_logits_pshape.rank().is_static() &&
        box_logits_pshape[1].is_static() && !attrs.share_location) {
        NODE_VALIDATION_CHECK(op,
                              (box_logits_pshape[1].get_length()) % (num_prior_boxes * 4) == 0,
                              "Box logits second dimension must be a multiply of num_prior_boxes * 4 (",
                              num_prior_boxes * 4,
                              "). Current value is: ",
                              box_logits_pshape[1].get_length(),
                              ".");
        num_classes = (box_logits_pshape[1].get_length()) / (num_prior_boxes * 4);
    }
    if (num_classes == 0 && num_prior_boxes > 0 && have_five_inputs && ad_box_preds_shape.rank().is_static() &&
        ad_box_preds_shape[1].is_static() && !attrs.share_location) {
        NODE_VALIDATION_CHECK(op,
                              (ad_box_preds_shape[1].get_length()) % (num_prior_boxes * 4) == 0,
                              "Additional box predictions second dimension must be a multiply of num_prior_boxes * 4 (",
                              num_prior_boxes * 4,
                              "). Current value is: ",
                              ad_box_preds_shape[1].get_length(),
                              ".");
        num_classes = ad_box_preds_shape[1].get_length() / (num_prior_boxes * 4);
    }
}

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer_base(const DetectionOutputBase* op,
                                      const DetectionOutputBase::AttributesBase& attrs,
                                      const std::vector<T>& input_shapes,
                                      int64_t attribute_num_classes) {
    using dim_t = typename T::value_type;
    using val_type = typename dim_t::value_type;

    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 3 || input_shapes.size() == 5));

    auto output_shapes = std::vector<TRShape>(1);
    auto& ret_output_shape = output_shapes[0];
    ret_output_shape.resize(4);

    const auto& box_logits_pshape = input_shapes[0];
    const auto& class_preds_pshape = input_shapes[1];
    const auto& proposals_pshape = input_shapes[2];

    val_type num_classes = 0;
    val_type num_prior_boxes = 0;
    dim_t dim_num_images{};
    bool dim_num_images_updated = false;

    if (attribute_num_classes == -1) {
        ov::op::util::compute_num_classes(op, attrs, input_shapes, num_classes, num_prior_boxes);
    } else {
        num_classes = static_cast<val_type>(attribute_num_classes);
    }

    const val_type num_loc_classes = attrs.share_location ? 1 : num_classes;
    const val_type prior_box_size = attrs.normalized ? 4 : 5;

    if (box_logits_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              box_logits_pshape.size() == 2,
                              "Box logits rank must be 2. Got ",
                              box_logits_pshape.size());
        dim_num_images = box_logits_pshape[0];
        dim_num_images_updated = true;
        if (!num_prior_boxes && box_logits_pshape[1].is_static()) {
            auto box_logits_pshape_2nd_dim = box_logits_pshape[1].get_length();
            NODE_VALIDATION_CHECK(op,
                                  num_loc_classes != 0 && (box_logits_pshape_2nd_dim % (num_loc_classes * 4)) == 0,
                                  "Box logits' second dimension must be a multiply of num_loc_classes * 4 (",
                                  num_loc_classes * 4,
                                  "). Current value is: ",
                                  box_logits_pshape_2nd_dim,
                                  ".");
            num_prior_boxes = box_logits_pshape_2nd_dim / (num_loc_classes * 4);
        }

        if (num_prior_boxes > 0 && num_loc_classes > 0) {
            NODE_SHAPE_INFER_CHECK(
                op,
                input_shapes,
                box_logits_pshape[1].compatible(num_prior_boxes * num_loc_classes * 4),
                "The second dimension of the first input (box logits) is not compatible. Current value: ",
                box_logits_pshape[1],
                ", expected value: ",
                num_prior_boxes * num_loc_classes * 4);
        }
    }
    if (class_preds_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              class_preds_pshape.size() == 2,
                              "Class predictions rank must be 2. Got ",
                              class_preds_pshape.size());
        if ((!dim_num_images_updated || dim_num_images.is_dynamic()) && class_preds_pshape[0].is_static()) {
            dim_num_images = class_preds_pshape[0];
            dim_num_images_updated = true;
        } else {
            NODE_VALIDATION_CHECK(
                op,
                class_preds_pshape[0].compatible(dim_num_images),
                "Class predictions' first dimension is not compatible with batch size.  Current value is: ",
                class_preds_pshape[0],
                ", expected: ",
                dim_num_images,
                ".");
        }
        if (class_preds_pshape[1].is_static() && num_classes) {
            auto class_preds_pshape_2nd_dim = class_preds_pshape[1].get_length();
            if (!num_prior_boxes) {
                NODE_VALIDATION_CHECK(op,
                                      class_preds_pshape_2nd_dim % num_classes == 0,
                                      "Class predictions' second dimension must be a multiply of num_classes (",
                                      num_classes,
                                      "). Current value is: ",
                                      class_preds_pshape_2nd_dim,
                                      ".");
                num_prior_boxes = class_preds_pshape_2nd_dim / num_classes;
            } else {
                NODE_VALIDATION_CHECK(op,
                                      class_preds_pshape_2nd_dim == num_prior_boxes * num_classes,
                                      "Class predictions' second dimension must be equal to num_prior_boxes * "
                                      "num_classes (",
                                      num_prior_boxes * num_classes,
                                      "). Current value is: ",
                                      class_preds_pshape_2nd_dim,
                                      ".");
            }
        }
    }
    if (proposals_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              proposals_pshape.size() == 3,
                              "Proposals rank must be 3. Got ",
                              proposals_pshape.size());
        NODE_VALIDATION_CHECK(op,
                              proposals_pshape[0].compatible(1) || proposals_pshape[0].compatible(dim_num_images),
                              "Proposals' first dimension is must be equal to either batch size (",
                              dim_num_images,
                              ") or 1. Got: ",
                              proposals_pshape[0],
                              ".");

        size_t proposals_expected_2nd_dim = attrs.variance_encoded_in_target ? 1 : 2;
        NODE_VALIDATION_CHECK(op,
                              proposals_pshape[1].compatible(proposals_expected_2nd_dim),
                              "Proposals' second dimension is mismatched. Current value is: ",
                              proposals_pshape[1],
                              ", expected: ",
                              proposals_expected_2nd_dim,
                              ".");

        if (proposals_pshape[2].is_static()) {
            auto proposals_pshape_3rd_dim = proposals_pshape[2].get_length();
            if (!num_prior_boxes) {
                NODE_VALIDATION_CHECK(op,
                                      proposals_pshape_3rd_dim % prior_box_size == 0,
                                      "Proposals' third dimension must be a multiply of prior_box_size (",
                                      prior_box_size,
                                      "). Current value is: ",
                                      proposals_pshape_3rd_dim,
                                      ".");
                num_prior_boxes = proposals_pshape_3rd_dim / prior_box_size;
            } else {
                NODE_VALIDATION_CHECK(op,
                                      proposals_pshape_3rd_dim == num_prior_boxes * prior_box_size,
                                      "Proposals' third dimension must be equal to num_prior_boxes "
                                      "* prior_box_size (",
                                      num_prior_boxes * prior_box_size,
                                      "). Current value is: ",
                                      proposals_pshape_3rd_dim,
                                      ".");
            }
        }
    }

    if (input_shapes.size() == 5) {
        const auto& aux_class_preds_pshape = input_shapes[3];
        const auto& aux_box_preds_pshape = input_shapes[4];
        if (aux_class_preds_pshape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  aux_class_preds_pshape.size() == 2,
                                  "additional class predictions rank must be 2. Got ",
                                  aux_class_preds_pshape.size());
            NODE_VALIDATION_CHECK(op,
                                  aux_class_preds_pshape[0].compatible(dim_num_images),
                                  "Additional class predictions' first dimension must be "
                                  "compatible with batch size. Current value is: ",
                                  aux_class_preds_pshape[0],
                                  ", expected: ",
                                  dim_num_images,
                                  ".");
            if (num_prior_boxes) {
                NODE_VALIDATION_CHECK(op,
                                      aux_class_preds_pshape[1].compatible(num_prior_boxes * 2),
                                      "Additional class predictions' second dimension must be compatible with "
                                      "num_prior_boxes * 2. Current value is: ",
                                      aux_class_preds_pshape[1],
                                      ", expected: ",
                                      num_prior_boxes * 2,
                                      ".");
            }

            if (aux_class_preds_pshape[1].is_static())
                num_prior_boxes = aux_class_preds_pshape[1].get_length() / 2;
        }
        NODE_VALIDATION_CHECK(
            op,
            aux_box_preds_pshape.compatible(box_logits_pshape),
            "Additional box predictions' shape must be compatible with box logits shape. Current value is: ",
            aux_box_preds_pshape,
            ", expected: ",
            box_logits_pshape,
            ".");
    }

    ret_output_shape[0] = 1;
    ret_output_shape[1] = 1;
    ret_output_shape[3] = 7;

    const dim_t dim_num_prior_boxes = num_prior_boxes ? dim_t{num_prior_boxes} : Dimension();
    const dim_t dim_num_classes = num_classes ? dim_t{num_classes} : Dimension();

    if (attrs.keep_top_k[0] > 0) {
        ret_output_shape[2] = dim_num_images * attrs.keep_top_k[0];
    } else if (attrs.keep_top_k[0] == -1 && attrs.top_k > 0) {
        ret_output_shape[2] = dim_num_images * attrs.top_k * dim_num_classes;
    } else {
        ret_output_shape[2] = dim_num_images * dim_num_prior_boxes * dim_num_classes;
    }
    return output_shapes;
}

}  // namespace util
}  // namespace op
}  // namespace ov

namespace ov {
namespace op {
namespace v0 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const DetectionOutput* op, const std::vector<TShape>& input_shapes) {
    const auto& attrs = op->get_attrs();
    return ov::op::util::shape_infer_base(op, attrs, input_shapes, attrs.num_classes);
}
}  // namespace v0
}  // namespace op
}  // namespace ov

namespace ov {
namespace op {
namespace v8 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const DetectionOutput* op, const std::vector<TShape>& input_shapes) {
    const auto& attrs = op->get_attrs();
    return ov::op::util::shape_infer_base(op, attrs, input_shapes, -1);
}
}  // namespace v8
}  // namespace op
}  // namespace ov
