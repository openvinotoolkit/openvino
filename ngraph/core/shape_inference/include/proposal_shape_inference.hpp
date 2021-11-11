// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/proposal.hpp>

namespace ov {
namespace op {
namespace v0 {

template <typename T>
inline void proposal_build_dynamic_dimension(T& outdim) {
    OPENVINO_UNREACHABLE("This code should be executed only for ov::Dimension class");
}

template <>
inline void proposal_build_dynamic_dimension<ov::Dimension>(ov::Dimension& outdim) {
    outdim = Dimension::dynamic();
}

template <class T>
void shape_infer(const ov::op::v0::Proposal* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 && output_shapes.size() == 1);
    const auto& class_probs_ps = input_shapes[0];
    const auto& bbox_deltas_ps = input_shapes[1];
    const auto& image_shape_ps = input_shapes[2];

    NODE_VALIDATION_CHECK(op,
                          class_probs_ps.rank().compatible(4),
                          "Proposal layer shape class_probs should be rank 4 compatible (",
                          class_probs_ps,
                          ").");

    NODE_VALIDATION_CHECK(op,
                          bbox_deltas_ps.rank().compatible(4),
                          "Proposal layer shape bbox_deltas should be rank 4 compatible (",
                          bbox_deltas_ps,
                          ").");

    NODE_VALIDATION_CHECK(op,
                          image_shape_ps.rank().compatible(1),
                          "Proposal layer shape image_shape should be rank 1 compatible (",
                          image_shape_ps,
                          ").");
    if (bbox_deltas_ps.is_static() && class_probs_ps.is_static()) {
        // class probs and bbox deltas shapes are static, check anchor count and batch number
        // consistency
        NODE_VALIDATION_CHECK(op,
                              class_probs_ps[1].get_length() * 2 == bbox_deltas_ps[1].get_length(),
                              "Anchor number inconsistent between class_probs (",
                              class_probs_ps[1].get_length() / 2,
                              "), and bbox_deltas (",
                              bbox_deltas_ps[1].get_length() / 4,
                              ").");

        NODE_VALIDATION_CHECK(op,
                              class_probs_ps[0] == bbox_deltas_ps[0],
                              "Batch size inconsistent between class_probs (",
                              class_probs_ps[0],
                              ") and bbox deltas (",
                              bbox_deltas_ps[0],
                              ").");
    }

    if (image_shape_ps.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              image_shape_ps[0].get_length() >= 3 && image_shape_ps[0].get_length() <= 4,
                              "Image_shape 1D tensor must have => 3 and <= 4 elements (image_shape_shape[0]",
                              image_shape_ps[0],
                              ").");
    }

    auto out_dim = DimType{};

    if (class_probs_ps.rank().is_static() && bbox_deltas_ps.rank().is_static()) {
        out_dim = (class_probs_ps[0] & bbox_deltas_ps[0]);
    } else if (class_probs_ps.rank().is_static()) {
        out_dim = class_probs_ps[0];
    } else if (bbox_deltas_ps.rank().is_static()) {
        out_dim = bbox_deltas_ps[0];
    } else {
        proposal_build_dynamic_dimension(out_dim);
    }

    /*
        const auto& class_probs_ps = get_input_partial_shape(0);
        const auto& bbox_deltas_ps = get_input_partial_shape(1);
        const auto& image_shape_ps = get_input_partial_shape(2);
        Dimension out_dim = Dimension::dynamic();
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(0).is_real(),
                              "Proposal layer input class_probs should have floating point type (",
                              get_input_element_type(0),
                              ").");

        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(1).is_real(),
                              "Proposal layer input bbox_deltas should have floating point type (",
                              get_input_element_type(1),
                              ").");

        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(2).is_real(),
                              "Proposal layer input image_shape should have floating point type (",
                              get_input_element_type(2),
                              ").");

        NODE_VALIDATION_CHECK(this,
                              class_probs_ps.rank().compatible(4),
                              "Proposal layer shape class_probs should be rank 4 compatible (",
                              class_probs_ps,
                              ").");

        NODE_VALIDATION_CHECK(this,
                              bbox_deltas_ps.rank().compatible(4),
                              "Proposal layer shape bbox_deltas should be rank 4 compatible (",
                              bbox_deltas_ps,
                              ").");

        NODE_VALIDATION_CHECK(this,
                              image_shape_ps.rank().compatible(1),
                              "Proposal layer shape image_shape should be rank 1 compatible (",
                              image_shape_ps,
                              ").");

        if (bbox_deltas_ps.is_static() && class_probs_ps.is_static()) {
            // class probs and bbox deltas shapes are static, check anchor count and batch number
            // consistency
            NODE_VALIDATION_CHECK(this,
                                  class_probs_ps[1].get_length() * 2 == bbox_deltas_ps[1].get_length(),
                                  "Anchor number inconsistent between class_probs (",
                                  class_probs_ps[1].get_length() / 2,
                                  "), and bbox_deltas (",
                                  bbox_deltas_ps[1].get_length() / 4,
                                  ").");

            NODE_VALIDATION_CHECK(this,
                                  class_probs_ps[0] == bbox_deltas_ps[0],
                                  "Batch size inconsistent between class_probs (",
                                  class_probs_ps[0],
                                  ") and bbox deltas (",
                                  bbox_deltas_ps[0],
                                  ").");
        }

        if (image_shape_ps.is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  image_shape_ps[0].get_length() >= 3 && image_shape_ps[0].get_length() <= 4,
                                  "Image_shape 1D tensor must have => 3 and <= 4 elements (image_shape_shape[0]",
                                  image_shape_ps[0],
                                  ").");
        }

        if (class_probs_ps.rank().is_static() && bbox_deltas_ps.rank().is_static()) {
            out_dim = (class_probs_ps[0] & bbox_deltas_ps[0]);
        } else if (class_probs_ps.rank().is_static()) {
            out_dim = class_probs_ps[0];
        } else if (bbox_deltas_ps.rank().is_static()) {
            out_dim = bbox_deltas_ps[0];
        }

        // intersect the batch size
        set_output_type(0, get_input_element_type(0), ov::PartialShape{out_dim * m_attrs.post_nms_topn, 5});
        */
    output_shapes[0].resize(2);
    (output_shapes[0])[0] = out_dim * op->get_attrs().post_nms_topn;
    (output_shapes[0])[1] = 5;
}

}  // namespace v0
}  // namespace op
}  // namespace ov

namespace ov {
namespace op {
namespace v4 {

template <class T>
inline void proposal_set_full_dynamic(T& output_shape) {
    OPENVINO_UNREACHABLE("Shape Infer can't set partial shape");
}

template <>
inline void proposal_set_full_dynamic<ov::PartialShape>(ov::PartialShape& output_shape) {
    output_shape = ov::PartialShape::dynamic();
}

template <class T>
void shape_infer(const ov::op::v4::Proposal* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 && output_shapes.size() == 2);

#if 0    
    NGRAPH_OP_SCOPE(v4_Proposal_validate_and_infer_types);
    v0::Proposal::validate_and_infer_types();
    // Output shape was inferred in v0's validate_and_infer_types
    const auto proposals_ps = get_output_partial_shape(0);
    auto out_ps = ov::PartialShape{Dimension::dynamic()};
    if (proposals_ps.rank().is_static() && proposals_ps.rank().compatible(2)) {
        out_ps = ov::PartialShape{proposals_ps[0]};
    }
    set_output_type(1, get_input_element_type(0), out_ps);
#endif
    const auto proposal_v0_op = dynamic_cast<const ov::op::v0::Proposal*>(op);
    auto output_vector_v0 = std::vector<T>{output_shapes[0]};
    shape_infer(proposal_v0_op, input_shapes, output_vector_v0);
    output_shapes[0] = output_vector_v0[0];

    const auto& proposals_ps = output_vector_v0[0];
    auto& out_ps = output_shapes[1];
    if (proposals_ps.rank().is_static() && proposals_ps.size() == 2) {
        out_ps = T{proposals_ps[0]};
    } else {
        proposal_set_full_dynamic(out_ps);
    }
}

}  // namespace v4
}  // namespace op
}  // namespace ov
