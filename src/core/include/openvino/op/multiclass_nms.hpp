// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/nms_base.hpp"

namespace ov {
namespace op {
namespace v8 {
/// \brief MulticlassNms operation
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API MulticlassNms : public util::NmsBase {
public:
    OPENVINO_OP("MulticlassNms", "opset8", op::util::NmsBase);
    BWDCMP_RTTI_DECLARATION;

    /// \brief Structure that specifies attributes of the operation
    struct Attributes {
        // specifies order of output elements
        SortResultType sort_result_type = SortResultType::NONE;
        // specifies whenever it is necessary to sort selected boxes across batches or
        // not
        bool sort_result_across_batch = false;
        // specifies the output tensor type
        ov::element::Type output_type = ov::element::i64;
        // specifies intersection over union threshold
        float iou_threshold = 0.0f;
        // specifies minimum score to consider box for the processing
        float score_threshold = 0.0f;
        // specifies maximum number of boxes to be selected per class, -1 meaning to
        // keep all boxes
        int nms_top_k = -1;
        // specifies maximum number of boxes to be selected per batch element, -1
        // meaning to keep all boxes
        int keep_top_k = -1;
        // specifies the background class id, -1 meaning to keep all classes
        int background_class = -1;
        // specifies eta parameter for adpative NMS, in close range [0, 1.0]
        float nms_eta = 1.0f;
        // specifies whether boxes are normalized or not
        bool normalized = true;
    };

    MulticlassNms();

    /// \brief Constructs a MulticlassNms operation
    ///
    /// \param boxes Node producing the box coordinates
    /// \param scores Node producing the box scores
    /// \param attrs Attributes of the operation
    MulticlassNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs);

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \brief Returns attributes of the operation MulticlassNms
    const Attributes& get_attrs() const {
        return m_attrs;
    }

protected:
    Attributes m_attrs;
    void validate() override;
};
}  // namespace v8
}  // namespace op
}  // namespace ov
