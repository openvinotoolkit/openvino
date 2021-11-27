// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/detection_output_base.hpp"

#include <ngraph/validation_util.hpp>

#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/squeeze.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ov::op::util;

BWDCMP_RTTI_DEFINITION(DetectionOutputBase);
DetectionOutputBase::DetectionOutputBase(OutputVector args) : Op(args) {}

ov::Dimension DetectionOutputBase::compute_num_classes(const AttributesBase& attrs) {
    Dimension num_classes = Dimension::dynamic();

    NODE_VALIDATION_CHECK(this,
                          3 <= get_input_size() && get_input_size() <= 5,
                          "A number of arguments must be greater than or equal to 3 and less than or equal to 5. Got " +
                              std::to_string(get_input_size()));

    const ov::PartialShape& box_logits_pshape = get_input_partial_shape(0);
    const ov::PartialShape& class_preds_pshape = get_input_partial_shape(1);
    const ov::PartialShape& proposals_pshape = get_input_partial_shape(2);
    ov::PartialShape ad_class_preds_shape = ov::PartialShape::dynamic();
    ov::PartialShape ad_box_preds_shape = ov::PartialShape::dynamic();

    if (box_logits_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(
            this,
            box_logits_pshape.rank().get_length() == 2,
            "Box logits rank must be 2. Got " + std::to_string(box_logits_pshape.rank().get_length()));
    }
    if (class_preds_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(
            this,
            class_preds_pshape.rank().get_length() == 2,
            "Class predictions rank must be 2. Got " + std::to_string(class_preds_pshape.rank().get_length()));
    }
    if (proposals_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this,
                              proposals_pshape.rank().get_length() == 3,
                              "Proposals rank must be 3. Got " + std::to_string(proposals_pshape.rank().get_length()));
    }
    if (get_input_size() >= 4) {
        ad_class_preds_shape = get_input_partial_shape(3);
        if (ad_class_preds_shape.rank().is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  ad_class_preds_shape.rank().get_length() == 2,
                                  "Additional class predictions rank must be 2. Got " +
                                      std::to_string(ad_class_preds_shape.rank().get_length()));
        }
    }
    if (get_input_size() == 5) {
        ad_box_preds_shape = get_input_partial_shape(4);
        if (ad_box_preds_shape.rank().is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  ad_box_preds_shape.rank().get_length() == 2,
                                  "Additional box predictions rank must be 2. Got " +
                                      std::to_string(ad_box_preds_shape.rank().get_length()));
        }
    }

    int prior_box_size = attrs.normalized ? 4 : 5;
    Dimension num_prior_boxes = Dimension::dynamic();

    // try to deduce a number of prior boxes
    if (num_prior_boxes.is_dynamic() && proposals_pshape.rank().is_static() && proposals_pshape[2].is_static()) {
        NODE_VALIDATION_CHECK(this,
                              proposals_pshape[2].get_length() % prior_box_size == 0,
                              "Proposals' third dimension must be a multiply of prior_box_size (" +
                                  std::to_string(prior_box_size) + "). Current value is: ",
                              proposals_pshape[2].get_length(),
                              ".");
        num_prior_boxes = proposals_pshape[2].get_length() / prior_box_size;
        NODE_VALIDATION_CHECK(
            this,
            num_prior_boxes.get_length() > 0,
            "A number of prior boxes must be greater zero. Got: " + std::to_string(num_prior_boxes.get_length()));
    }
    if (num_prior_boxes.is_dynamic() && ad_class_preds_shape.rank().is_static() &&
        ad_class_preds_shape[1].is_static()) {
        NODE_VALIDATION_CHECK(
            this,
            ad_class_preds_shape[1].get_length() % 2 == 0,
            "Additional class predictions second dimension must be a multiply of 2. Current value is: ",
            ad_class_preds_shape[1].get_length(),
            ".");
        num_prior_boxes = ad_class_preds_shape[1].get_length() / 2;
        NODE_VALIDATION_CHECK(
            this,
            num_prior_boxes.get_length() > 0,
            "A number of prior boxes must be greater zero. Got: " + std::to_string(num_prior_boxes.get_length()));
    }

    // try to deduce a number of classes
    if (num_classes.is_dynamic() && num_prior_boxes.is_static() && class_preds_pshape.rank().is_static() &&
        class_preds_pshape[1].is_static()) {
        NODE_VALIDATION_CHECK(this,
                              class_preds_pshape[1].get_length() % num_prior_boxes.get_length() == 0,
                              "Class predictions second dimension must be a multiply of num_prior_boxes (" +
                                  std::to_string(num_prior_boxes.get_length()) + "). Current value is: ",
                              class_preds_pshape[1].get_length(),
                              ".");
        num_classes = class_preds_pshape[1].get_length() / num_prior_boxes.get_length();
    }
    if (num_classes.is_dynamic() && num_prior_boxes.is_static() && box_logits_pshape.rank().is_static() &&
        box_logits_pshape[1].is_static() && !attrs.share_location) {
        NODE_VALIDATION_CHECK(this,
                              box_logits_pshape[1].get_length() % (num_prior_boxes.get_length() * 4) == 0,
                              "Box logits second dimension must be a multiply of num_prior_boxes * 4 (" +
                                  std::to_string(num_prior_boxes.get_length() * 4) + "). Current value is: ",
                              box_logits_pshape[1].get_length(),
                              ".");
        num_classes = box_logits_pshape[1].get_length() / (num_prior_boxes.get_length() * 4);
    }
    if (num_classes.is_dynamic() && num_prior_boxes.is_static() && ad_box_preds_shape.is_static() &&
        ad_box_preds_shape[1].is_static() && !attrs.share_location) {
        NODE_VALIDATION_CHECK(
            this,
            ad_box_preds_shape[1].get_length() % (num_prior_boxes.get_length() * 4) == 0,
            "Additional box predictions second dimension must be a multiply of num_prior_boxes * 4 (" +
                std::to_string(num_prior_boxes.get_length() * 4) + "). Current value is: ",
            ad_box_preds_shape[1].get_length(),
            ".");
        num_classes = ad_box_preds_shape[1].get_length() / (num_prior_boxes.get_length() * 4);
    }

    return num_classes;
}

void DetectionOutputBase::validate_and_infer_types_base(const DetectionOutputBase::AttributesBase& attrs,
                                                        Dimension num_classes) {
    NODE_VALIDATION_CHECK(this, attrs.keep_top_k.size() > 0, "keep_top_k attribute must be provided");

    NODE_VALIDATION_CHECK(
        this,
        attrs.code_type == "caffe.PriorBoxParameter.CORNER" || attrs.code_type == "caffe.PriorBoxParameter.CENTER_SIZE",
        "code_type must be either \"caffe.PriorBoxParameter.CORNER\" or "
        "\"caffe.PriorBoxParameter.CENTER_SIZE\"");

    auto box_logits_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          box_logits_et.is_real(),
                          "Box logits' data type must be floating point. Got " + box_logits_et.get_type_name());
    auto class_preds_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          class_preds_et == box_logits_et,
                          "Class predictions' data type must be the same as box logits type (" +
                              box_logits_et.get_type_name() + "). Got " + class_preds_et.get_type_name());
    auto proposals_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          proposals_et.is_real(),
                          "Proposals' data type must be floating point. Got " + proposals_et.get_type_name());

    const ov::PartialShape& box_logits_pshape = get_input_partial_shape(0);
    const ov::PartialShape& class_preds_pshape = get_input_partial_shape(1);
    const ov::PartialShape& proposals_pshape = get_input_partial_shape(2);

    // deduce a number of classes for DetectionOutput-8
    if (num_classes.is_dynamic()) {
        num_classes = compute_num_classes(attrs);
    }

    Dimension num_loc_classes = attrs.share_location ? 1 : num_classes;
    int prior_box_size = attrs.normalized ? 4 : 5;

    Dimension num_images = Dimension::dynamic();
    Dimension num_prior_boxes = Dimension::dynamic();
    if (box_logits_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(
            this,
            box_logits_pshape.rank().get_length() == 2,
            "Box logits rank must be 2. Got " + std::to_string(box_logits_pshape.rank().get_length()));
        num_images = box_logits_pshape[0];
        if (box_logits_pshape[1].is_static() && num_loc_classes.is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  (box_logits_pshape[1].get_length() % (num_loc_classes.get_length() * 4)) == 0,
                                  "Box logits' second dimension must be a multiply of num_loc_classes * 4 (" +
                                      std::to_string(num_loc_classes.get_length() * 4) + "). Current value is: ",
                                  box_logits_pshape[1].get_length(),
                                  ".");
            num_prior_boxes = box_logits_pshape[1].get_length() / (num_loc_classes.get_length() * 4);
        }
    }
    if (class_preds_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(
            this,
            class_preds_pshape.rank().get_length() == 2,
            "Class predictions rank must be 2. Got " + std::to_string(class_preds_pshape.rank().get_length()));
        if (num_images.is_dynamic() && class_preds_pshape[0].is_static()) {
            num_images = class_preds_pshape[0];
        } else {
            NODE_VALIDATION_CHECK(this,
                                  class_preds_pshape[0].compatible(num_images),
                                  "Class predictions' first dimension is not compatible with batch size.");
        }
        if (class_preds_pshape[1].is_static()) {
            if (num_prior_boxes.is_dynamic() && num_classes.is_static()) {
                NODE_VALIDATION_CHECK(this,
                                      class_preds_pshape[1].get_length() % num_classes.get_length() == 0,
                                      "Class predictions' second dimension must be a multiply of num_classes (" +
                                          std::to_string(num_classes.get_length()) + "). Current value is: ",
                                      class_preds_pshape[1].get_length(),
                                      ".");
                num_prior_boxes = class_preds_pshape[1].get_length() / num_classes.get_length();
            } else if (num_classes.is_static()) {
                int num_prior_boxes_val = num_prior_boxes.get_length();
                NODE_VALIDATION_CHECK(
                    this,
                    class_preds_pshape[1].get_length() == num_prior_boxes_val * num_classes.get_length(),
                    "Class predictions' second dimension must be equal to num_prior_boxes * "
                    "num_classes (" +
                        std::to_string(num_prior_boxes_val * num_classes.get_length()) + "). Current value is: ",
                    class_preds_pshape[1].get_length(),
                    ".");
            }
        }
    }
    if (proposals_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this,
                              proposals_pshape.rank().get_length() == 3,
                              "Proposals rank must be 3. Got " + std::to_string(proposals_pshape.rank().get_length()));
        if (num_images.is_static() && proposals_pshape[0].is_static()) {
            int64_t proposals_1st_dim = proposals_pshape[0].get_length();
            int64_t num_images_val = num_images.get_length();
            NODE_VALIDATION_CHECK(this,
                                  proposals_1st_dim == 1 || proposals_1st_dim == num_images_val,
                                  "Proposals' first dimension is must be equal to either batch size (" +
                                      std::to_string(num_images_val) +
                                      ") or 1. Got: " + std::to_string(proposals_1st_dim) + ".");
        }
        if (proposals_pshape[1].is_static()) {
            size_t proposals_expected_2nd_dim = attrs.variance_encoded_in_target ? 1 : 2;
            NODE_VALIDATION_CHECK(this,
                                  proposals_pshape[1].compatible(proposals_expected_2nd_dim),
                                  "Proposals' second dimension is mismatched. Current value is: ",
                                  proposals_pshape[1].get_length(),
                                  ", expected: ",
                                  proposals_expected_2nd_dim,
                                  ".");
        }
        if (proposals_pshape[2].is_static()) {
            if (num_prior_boxes.is_dynamic()) {
                NODE_VALIDATION_CHECK(this,
                                      proposals_pshape[2].get_length() % prior_box_size == 0,
                                      "Proposals' third dimension must be a multiply of prior_box_size (" +
                                          std::to_string(prior_box_size) + "). Current value is: ",
                                      proposals_pshape[2].get_length(),
                                      ".");
                num_prior_boxes = proposals_pshape[2].get_length() / prior_box_size;
            } else {
                int num_prior_boxes_val = num_prior_boxes.get_length();
                NODE_VALIDATION_CHECK(this,
                                      proposals_pshape[2].get_length() == num_prior_boxes_val * prior_box_size,
                                      "Proposals' third dimension must be equal to num_prior_boxes "
                                      "* prior_box_size (" +
                                          std::to_string(num_prior_boxes_val * prior_box_size) +
                                          "). Current value is: ",
                                      proposals_pshape[2].get_length(),
                                      ".");
            }
        }
    }

    if (get_input_size() > 3) {
        auto aux_class_preds_et = get_input_element_type(3);
        NODE_VALIDATION_CHECK(this,
                              aux_class_preds_et == class_preds_et,
                              "Additional class predictions' data type must be the same as class "
                              "predictions data type (" +
                                  class_preds_et.get_type_name() + "). Got " + aux_class_preds_et.get_type_name());
        auto aux_box_preds_et = get_input_element_type(4);
        NODE_VALIDATION_CHECK(this,
                              aux_box_preds_et == box_logits_et,
                              "Additional box predictions' data type must be the same as box logits data type (" +
                                  box_logits_et.get_type_name() + "). Got " + aux_box_preds_et.get_type_name());

        const ov::PartialShape& aux_class_preds_pshape = get_input_partial_shape(3);
        const ov::PartialShape& aux_box_preds_pshape = get_input_partial_shape(4);
        if (aux_class_preds_pshape.rank().is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  aux_class_preds_pshape[0].compatible(num_images),
                                  "Additional class predictions' first dimension must be "
                                  "compatible with batch size.");
            if (num_prior_boxes.is_static()) {
                int num_prior_boxes_val = num_prior_boxes.get_length();
                NODE_VALIDATION_CHECK(this,
                                      aux_class_preds_pshape[1].get_length() == num_prior_boxes_val * 2,
                                      "Additional class predictions' second dimension must be equal to "
                                      "num_prior_boxes * 2 (" +
                                          std::to_string(num_prior_boxes_val * 2) + "). Got " +
                                          std::to_string(aux_class_preds_pshape[1].get_length()) + ".");
            }
        }
        NODE_VALIDATION_CHECK(this,
                              aux_box_preds_pshape.compatible(box_logits_pshape),
                              "Additional box predictions' shape must be compatible with box logits shape.");
    }

    std::vector<Dimension> output_shape{1, 1};
    if (attrs.keep_top_k[0] > 0) {
        output_shape.push_back(num_images * attrs.keep_top_k[0]);
    } else if (attrs.top_k > 0 && num_classes.is_static()) {
        output_shape.push_back(num_images * attrs.top_k * num_classes);
    } else if (num_classes.is_static()) {
        output_shape.push_back(num_images * num_prior_boxes * num_classes);
    } else {
        output_shape.push_back(Dimension::dynamic());
    }
    output_shape.emplace_back(7);

    set_output_type(0, box_logits_et, output_shape);
}

bool ov::op::util::DetectionOutputBase::visit_attributes_base(AttributeVisitor& visitor,
                                                              DetectionOutputBase::AttributesBase& attrs) {
    visitor.on_attribute("background_label_id", attrs.background_label_id);
    visitor.on_attribute("top_k", attrs.top_k);
    visitor.on_attribute("variance_encoded_in_target", attrs.variance_encoded_in_target);
    visitor.on_attribute("keep_top_k", attrs.keep_top_k);
    visitor.on_attribute("code_type", attrs.code_type);
    visitor.on_attribute("share_location", attrs.share_location);
    visitor.on_attribute("nms_threshold", attrs.nms_threshold);
    visitor.on_attribute("confidence_threshold", attrs.confidence_threshold);
    visitor.on_attribute("clip_after_nms", attrs.clip_after_nms);
    visitor.on_attribute("clip_before_nms", attrs.clip_before_nms);
    visitor.on_attribute("decrease_label_id", attrs.decrease_label_id);
    visitor.on_attribute("normalized", attrs.normalized);
    visitor.on_attribute("input_height", attrs.input_height);
    visitor.on_attribute("input_width", attrs.input_width);
    visitor.on_attribute("objectness_score", attrs.objectness_score);
    return true;
}
