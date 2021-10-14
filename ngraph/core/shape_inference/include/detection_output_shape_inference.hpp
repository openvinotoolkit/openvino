// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/detection_output.hpp>

namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const DetectionOutput* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    using dim_t = typename std::decay<decltype((input_shapes[0])[0])>::type;
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 3 || input_shapes.size() == 5) && output_shapes.size() == 1);

    auto& ret_output_shape = output_shapes[0];
    ret_output_shape.resize(4);

    const auto& box_logits_pshape = input_shapes[0];
    const auto& class_preds_pshape = input_shapes[1];
    const auto& proposals_pshape = input_shapes[2];

    const int& num_loc_classes = op->m_attrs.share_location ? 1 : op->m_attrs.num_classes;
    const int& prior_box_size = op->m_attrs.normalized ? 4 : 5;

    dim_t num_images{};
    dim_t num_prior_boxes{};

    if (box_logits_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              box_logits_pshape.size() == 2,
                              "Box logits rank must be 2. Got " + std::to_string(box_logits_pshape.size()));
        num_images = box_logits_pshape[0];
        if (box_logits_pshape[1].is_static()) {
            auto box_logits_pshape_2nd_dim = box_logits_pshape[1].get_length();
            NODE_VALIDATION_CHECK(op,
                                  (box_logits_pshape_2nd_dim % (num_loc_classes * 4)) == 0,
                                  "Box logits' second dimension must be a multiply of num_loc_classes * 4 (" +
                                      std::to_string(num_loc_classes * 4) + "). Current value is: ",
                                  box_logits_pshape_2nd_dim,
                                  ".");
            num_prior_boxes = box_logits_pshape_2nd_dim / (num_loc_classes * 4);
        }
    }
    if (class_preds_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              class_preds_pshape.size() == 2,
                              "Class predictions rank must be 2. Got " + std::to_string(class_preds_pshape.size()));
        if (num_images.is_dynamic() && class_preds_pshape[0].is_static()) {
            num_images = class_preds_pshape[0];
        } else {
            NODE_VALIDATION_CHECK(op,
                                  class_preds_pshape[0].compatible(num_images),
                                  "Class predictions' first dimension is not compatible with batch size.");
        }
        if (class_preds_pshape[1].is_static()) {
            auto class_preds_pshape_2nd_dim = class_preds_pshape[1].get_length();
            if (num_prior_boxes.is_dynamic()) {
                NODE_VALIDATION_CHECK(op,
                                      class_preds_pshape_2nd_dim % op->m_attrs.num_classes == 0,
                                      "Class predictions' second dimension must be a multiply of num_classes (" +
                                          std::to_string(op->m_attrs.num_classes) + "). Current value is: ",
                                      class_preds_pshape_2nd_dim,
                                      ".");
                num_prior_boxes = class_preds_pshape_2nd_dim / op->m_attrs.num_classes;
            } else {
                int num_prior_boxes_val = num_prior_boxes.get_length();
                NODE_VALIDATION_CHECK(op,
                                      class_preds_pshape_2nd_dim == num_prior_boxes_val * op->m_attrs.num_classes,
                                      "Class predictions' second dimension must be equal to num_prior_boxes * "
                                      "num_classes (" +
                                          std::to_string(num_prior_boxes_val * op->m_attrs.num_classes) +
                                          "). Current value is: ",
                                      class_preds_pshape_2nd_dim,
                                      ".");
            }
        }
    }
    if (proposals_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              proposals_pshape.size() == 3,
                              "Proposals rank must be 3. Got " + std::to_string(proposals_pshape.size()));
        if (num_images.is_static() && proposals_pshape[0].is_static()) {
            int64_t proposals_1st_dim = proposals_pshape[0].get_length();
            int64_t num_images_val = num_images.get_length();
            NODE_VALIDATION_CHECK(op,
                                  proposals_1st_dim == 1 || proposals_1st_dim == num_images_val,
                                  "Proposals' first dimension is must be equal to either batch size (" +
                                      std::to_string(num_images_val) +
                                      ") or 1. Got: " + std::to_string(proposals_1st_dim) + ".");
        }

        size_t proposals_expected_2nd_dim = op->m_attrs.variance_encoded_in_target ? 1 : 2;
        NODE_VALIDATION_CHECK(op,
                              proposals_pshape[1].compatible(proposals_expected_2nd_dim),
                              "Proposals' second dimension is mismatched. Current value is: ",
                              proposals_pshape[1],
                              ", expected: ",
                              proposals_expected_2nd_dim,
                              ".");

        if (proposals_pshape[2].is_static()) {
            auto proposals_pshape_3rd_dim = proposals_pshape[2].get_length();
            if (num_prior_boxes.is_dynamic()) {
                NODE_VALIDATION_CHECK(op,
                                      proposals_pshape_3rd_dim % prior_box_size == 0,
                                      "Proposals' third dimension must be a multiply of prior_box_size (" +
                                          std::to_string(prior_box_size) + "). Current value is: ",
                                      proposals_pshape_3rd_dim,
                                      ".");
                num_prior_boxes = proposals_pshape_3rd_dim / prior_box_size;
            } else {
                int num_prior_boxes_val = num_prior_boxes.get_length();
                NODE_VALIDATION_CHECK(op,
                                      proposals_pshape_3rd_dim == num_prior_boxes_val * prior_box_size,
                                      "Proposals' third dimension must be equal to num_prior_boxes "
                                      "* prior_box_size (" +
                                          std::to_string(num_prior_boxes_val * prior_box_size) +
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
                                  aux_class_preds_pshape[0].compatible(num_images),
                                  "Additional class predictions' first dimension must be "
                                  "compatible with batch size.");
            if (num_prior_boxes.is_static()) {
                int num_prior_boxes_val = num_prior_boxes.get_length();
                NODE_VALIDATION_CHECK(op,
                                      aux_class_preds_pshape[1].compatible(num_prior_boxes_val * 2),
                                      "Additional class predictions' second dimension must be compatible with "
                                      "num_prior_boxes * 2 ");
            }
        }
        NODE_VALIDATION_CHECK(op,
                              aux_box_preds_pshape.compatible(box_logits_pshape),
                              "Additional box predictions' shape must be compatible with box logits shape.");
    }

    ret_output_shape[0] = 1;
    ret_output_shape[1] = 1;
    ret_output_shape[3] = 7;

    if (op->m_attrs.keep_top_k[0] > 0) {
        ret_output_shape[2] = num_images * op->m_attrs.keep_top_k[0];
    } else if (op->m_attrs.top_k > 0) {
        ret_output_shape[2] = num_images * op->m_attrs.top_k * op->m_attrs.num_classes;
    } else {
        ret_output_shape[2] = num_images * num_prior_boxes * op->m_attrs.num_classes;
    }
}

}  // namespace v0
}  // namespace op
}  // namespace ov
