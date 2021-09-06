// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Layer which performs non-max suppression to
/// generate detection output using location and confidence predictions
class OPENVINO_API DetectionOutput : public Op {
public:
    struct Attributes {
        int num_classes;
        int background_label_id = 0;
        int top_k = -1;
        bool variance_encoded_in_target = false;
        std::vector<int> keep_top_k;
        std::string code_type = std::string{"caffe.PriorBoxParameter.CORNER"};
        bool share_location = true;
        float nms_threshold;
        float confidence_threshold = 0;
        bool clip_after_nms = false;
        bool clip_before_nms = false;
        bool decrease_label_id = false;
        bool normalized = false;
        size_t input_height = 1;
        size_t input_width = 1;
        float objectness_score = 0;
    };

    OPENVINO_RTTI_DECLARATION;

    DetectionOutput() = default;
    /// \brief Constructs a DetectionOutput operation
    ///
    /// \param box_logits			Box logits
    /// \param class_preds			Class predictions
    /// \param proposals			Proposals
    /// \param aux_class_preds		Auxilary class predictions
    /// \param aux_box_preds		Auxilary box predictions
    /// \param attrs				Detection Output attributes
    DetectionOutput(const Output<Node>& box_logits,
                    const Output<Node>& class_preds,
                    const Output<Node>& proposals,
                    const Output<Node>& aux_class_preds,
                    const Output<Node>& aux_box_preds,
                    const Attributes& attrs);

    /// \brief Constructs a DetectionOutput operation
    ///
    /// \param box_logits			Box logits
    /// \param class_preds			Class predictions
    /// \param proposals			Proposals
    /// \param attrs				Detection Output attributes
    DetectionOutput(const Output<Node>& box_logits,
                    const Output<Node>& class_preds,
                    const Output<Node>& proposals,
                    const Attributes& attrs);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    const Attributes& get_attrs() const {
        return m_attrs;
    }
    bool visit_attributes(AttributeVisitor& visitor) override;

private:
    Attributes m_attrs;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
