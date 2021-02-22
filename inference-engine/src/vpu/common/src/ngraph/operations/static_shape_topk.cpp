// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/ngraph/operations/static_shape_topk.hpp>
#include <ngraph/validation_util.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::vpu::op::StaticShapeTopK, "StaticShapeTopK", 0);

ngraph::vpu::op::StaticShapeTopK::StaticShapeTopK(
        const Output<Node>& data,
        const Output<Node>& k,
        const int64_t axis,
        const std::string& mode,
        const std::string& sort,
        const element::Type& index_element_type)
        : ngraph::op::v3::TopK{data, k, axis, mode, sort, index_element_type} {
    constructor_validate_and_infer_types();
}

ngraph::vpu::op::StaticShapeTopK::StaticShapeTopK(
        const ngraph::Output<ngraph::Node> &data,
        const ngraph::Output<ngraph::Node> &k,
        const int64_t axis,
        const ngraph::vpu::op::StaticShapeTopK::Mode mode,
        const ngraph::vpu::op::StaticShapeTopK::SortType sort,
        const ngraph::element::Type &index_element_type)
        : ngraph::op::v3::TopK{data, k, axis, mode, sort, index_element_type} {
    constructor_validate_and_infer_types();
}

void ngraph::vpu::op::StaticShapeTopK::validate_and_infer_types() {
    if (get_output_partial_shape(0).is_static() && get_output_partial_shape(1).is_static()) {
        return;
    }

    ngraph::op::v3::TopK::validate_and_infer_types();
    auto outputShape = get_output_partial_shape(0);
    NODE_VALIDATION_CHECK(this, outputShape.rank().is_static(), "StaticShapeTopK (", get_friendly_name(), ") ",
                          "output is expected to be of static rank");
    for (size_t i = 0; i < outputShape.rank().get_length(); i++) {
        outputShape[i] = outputShape[i].get_max_length();
    }
    NODE_VALIDATION_CHECK(this, outputShape.is_static(),
                          "StaticShapeTopK (", get_friendly_name(), ") can't evaluate output shape");

    set_output_type(0, get_input_element_type(0), outputShape);
    set_output_type(1, m_index_element_type, outputShape);
}
