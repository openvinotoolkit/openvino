// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/detection_output_upgrade.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;
using namespace testing;

namespace {
void create_attributes_vectors(std::vector<opset1::DetectionOutput::Attributes>& attrs_v1_vector,
                               std::vector<opset8::DetectionOutput::Attributes>& attrs_v8_vector) {
    // initialize attributes affecting shape inference
    // others remain by default
    for (int keep_top_k : {10, -1}) {
        for (int top_k : {5, -1}) {
            for (bool variance_encoded_in_target : {true, false}) {
                for (bool share_location : {true, false}) {
                    for (bool normalized : {true, false}) {
                        opset1::DetectionOutput::Attributes attrs_v1;
                        opset8::DetectionOutput::Attributes attrs_v8;
                        attrs_v1.top_k = attrs_v8.top_k = top_k;
                        attrs_v1.keep_top_k = attrs_v8.keep_top_k = {keep_top_k};
                        attrs_v1.variance_encoded_in_target = attrs_v8.variance_encoded_in_target =
                            variance_encoded_in_target;
                        attrs_v1.share_location = attrs_v8.share_location = share_location;
                        attrs_v1.normalized = attrs_v8.normalized = normalized;
                        attrs_v1.nms_threshold = attrs_v8.nms_threshold = 0.5f;
                        attrs_v1_vector.push_back(attrs_v1);
                        attrs_v8_vector.push_back(attrs_v8);
                    }
                }
            }
        }
    }
}
}  // namespace

TEST(TransformationTests, DetectionOutput1ToDetectionOutput8) {
    std::vector<opset1::DetectionOutput::Attributes> attrs_v1_vector;
    std::vector<opset8::DetectionOutput::Attributes> attrs_v8_vector;
    Dimension N = 5;
    Dimension num_prior_boxes = 100;
    Dimension priors_batch_size = N;
    Dimension num_classes = 23;

    create_attributes_vectors(attrs_v1_vector, attrs_v8_vector);
    ASSERT_TRUE(attrs_v1_vector.size() == attrs_v8_vector.size()) << "Sizes of attribute test vectors must be equal";
    for (size_t ind = 0; ind < attrs_v1_vector.size(); ++ind) {
        std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
        // this case covers deducing a number of classes value
        // since this value is not saved in attributes
        opset8::DetectionOutput::Attributes attributes_v8 = attrs_v8_vector[ind];
        opset1::DetectionOutput::Attributes attributes_v1 = attrs_v1_vector[ind];
        if (num_classes.is_static()) {
            attributes_v1.num_classes = (int)num_classes.get_length();
        }

        Dimension num_loc_classes = attributes_v8.share_location ? 1 : num_classes;
        Dimension prior_box_size = attributes_v8.normalized ? 4 : 5;

        PartialShape box_logits_shape = {N, num_prior_boxes * num_loc_classes * 4};
        PartialShape class_preds_shape = {N, num_prior_boxes * num_classes};
        PartialShape proposals_shape = {priors_batch_size,
                                        attributes_v8.variance_encoded_in_target ? 1 : 2,
                                        num_prior_boxes * prior_box_size};

        {
            auto box_logits = std::make_shared<opset1::Parameter>(element::f32, box_logits_shape);
            auto class_preds = std::make_shared<opset1::Parameter>(element::f32, class_preds_shape);
            auto proposals = std::make_shared<opset1::Parameter>(element::f32, proposals_shape);

            auto detection_output_v1 =
                std::make_shared<opset1::DetectionOutput>(box_logits, class_preds, proposals, attributes_v1);

            f = std::make_shared<ov::Model>(NodeVector{detection_output_v1},
                                            ParameterVector{box_logits, class_preds, proposals});

            pass::Manager manager;
            manager.register_pass<ov::pass::ConvertDetectionOutput1ToDetectionOutput8>();
            manager.run_passes(f);
        }

        {
            auto box_logits = std::make_shared<opset1::Parameter>(element::f32, box_logits_shape);
            auto class_preds = std::make_shared<opset1::Parameter>(element::f32, class_preds_shape);
            auto proposals = std::make_shared<opset1::Parameter>(element::f32, proposals_shape);

            auto detection_output_v8 =
                std::make_shared<opset8::DetectionOutput>(box_logits, class_preds, proposals, attributes_v8);

            f_ref = std::make_shared<ov::Model>(NodeVector{detection_output_v8},
                                                ParameterVector{box_logits, class_preds, proposals});
        }
        const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
        auto res = fc(f, f_ref);
        ASSERT_TRUE(res.valid) << res.message;
    }
}

TEST(TransformationTests, DetectionOutput1ToDetectionOutput8FiveArguments) {
    // In this case num_classes attribute value is deduced using inputs shapes
    std::vector<opset1::DetectionOutput::Attributes> attrs_v1_vector;
    std::vector<opset8::DetectionOutput::Attributes> attrs_v8_vector;
    Dimension N = 5;
    Dimension num_prior_boxes = 15;
    Dimension priors_batch_size = N;
    Dimension num_classes = 23;

    create_attributes_vectors(attrs_v1_vector, attrs_v8_vector);
    ASSERT_TRUE(attrs_v1_vector.size() == attrs_v8_vector.size()) << "Sizes of attribute test vectors must be equal";
    for (size_t ind = 0; ind < attrs_v1_vector.size(); ++ind) {
        std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
        opset8::DetectionOutput::Attributes attributes_v8 = attrs_v8_vector[ind];
        opset1::DetectionOutput::Attributes attributes_v1 = attrs_v1_vector[ind];
        if (num_classes.is_static()) {
            attributes_v1.num_classes = (int)num_classes.get_length();
        }

        Dimension num_loc_classes = attributes_v8.share_location ? 1 : num_classes;
        Dimension prior_box_size = attributes_v8.normalized ? 4 : 5;

        PartialShape box_logits_shape = {N, num_prior_boxes * num_loc_classes * 4};
        PartialShape class_preds_shape = {N, num_prior_boxes * num_classes};
        PartialShape proposals_shape = {priors_batch_size,
                                        attributes_v8.variance_encoded_in_target ? 1 : 2,
                                        num_prior_boxes * prior_box_size};
        PartialShape ad_class_preds_shape = {N, num_prior_boxes * 2};
        PartialShape ad_box_preds_shape = {N, num_prior_boxes * num_loc_classes * 4};

        {
            auto box_logits = std::make_shared<opset1::Parameter>(element::f32, box_logits_shape);
            auto class_preds = std::make_shared<opset1::Parameter>(element::f32, class_preds_shape);
            auto proposals = std::make_shared<opset1::Parameter>(element::f32, proposals_shape);
            auto ad_class_preds = std::make_shared<opset1::Parameter>(element::f32, ad_class_preds_shape);
            auto ad_box_preds = std::make_shared<opset1::Parameter>(element::f32, ad_box_preds_shape);

            auto detection_output_v1 = std::make_shared<opset1::DetectionOutput>(box_logits,
                                                                                 class_preds,
                                                                                 proposals,
                                                                                 ad_class_preds,
                                                                                 ad_box_preds,
                                                                                 attributes_v1);

            f = std::make_shared<ov::Model>(
                NodeVector{detection_output_v1},
                ParameterVector{box_logits, class_preds, proposals, ad_class_preds, ad_box_preds});

            pass::Manager manager;
            manager.register_pass<ov::pass::ConvertDetectionOutput1ToDetectionOutput8>();
            manager.run_passes(f);
        }

        {
            auto box_logits = std::make_shared<opset1::Parameter>(element::f32, box_logits_shape);
            auto class_preds = std::make_shared<opset1::Parameter>(element::f32, class_preds_shape);
            auto proposals = std::make_shared<opset1::Parameter>(element::f32, proposals_shape);
            auto ad_class_preds = std::make_shared<opset1::Parameter>(element::f32, ad_class_preds_shape);
            auto ad_box_preds = std::make_shared<opset1::Parameter>(element::f32, ad_box_preds_shape);

            auto detection_output_v8 = std::make_shared<opset8::DetectionOutput>(box_logits,
                                                                                 class_preds,
                                                                                 proposals,
                                                                                 ad_class_preds,
                                                                                 ad_box_preds,
                                                                                 attributes_v8);

            f_ref = std::make_shared<ov::Model>(
                NodeVector{detection_output_v8},
                ParameterVector{box_logits, class_preds, proposals, ad_class_preds, ad_box_preds});
        }
        const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
        auto res = fc(f, f_ref);
        ASSERT_TRUE(res.valid) << res.message;
    }
}
