// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/ngraph/operations/static_shape_topk.hpp>
#include <ngraph/validation_util.hpp>

constexpr ngraph::NodeTypeInfo ngraph::vpu::op::StaticShapeTopK::type_info;

static const std::uint64_t UNKNOWN_NORMALIZED_AXIS = std::numeric_limits<uint64_t>::max();

ngraph::vpu::op::StaticShapeTopK::StaticShapeTopK(
        const Output<Node>& data,
        const Output<Node>& k,
        const int64_t axis,
        const std::string& mode,
        const std::string& sort,
        const element::Type& index_element_type)
        : Op{{data, k}}
        , m_axis{axis}
        , m_maximumK{-1}
        , m_normalized_axis{0}
        , m_mode{as_enum<Mode>(mode)}
        , m_sort{as_enum<SortType>(sort)}
        , m_index_element_type{index_element_type} {
    constructor_validate_and_infer_types();
}

ngraph::vpu::op::StaticShapeTopK::StaticShapeTopK(
        const ngraph::Output<ngraph::Node> &data,
        const ngraph::Output<ngraph::Node> &k,
        const int64_t axis,
        const ngraph::vpu::op::StaticShapeTopK::Mode mode,
        const ngraph::vpu::op::StaticShapeTopK::SortType sort,
        const ngraph::element::Type &index_element_type)
        : Op{{data, k}}
        , m_axis{axis}
        , m_maximumK{-1}
        , m_normalized_axis{0}
        , m_mode{mode}
        , m_sort{sort}
        , m_index_element_type{index_element_type} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ngraph::vpu::op::StaticShapeTopK::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    auto new_v1_topk = std::make_shared<ngraph::vpu::op::StaticShapeTopK>(
            new_args.at(0), new_args.at(1), m_axis, m_mode, m_sort);

    new_v1_topk->set_index_element_type(m_index_element_type);

    return std::move(new_v1_topk);
}

void ngraph::vpu::op::StaticShapeTopK::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_integral_number(),
                          "K input has to be an integer type, which does match the provided one:",
                          get_input_element_type(1));
    const auto& input_partial_shape = get_input_partial_shape(0);
    const auto input_rank = input_partial_shape.rank();

    NODE_VALIDATION_CHECK(this,
                          input_rank.is_static() && input_rank.get_length() > 0,
                          "Input rank must be greater than 0.");

    const auto& k_partial_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(
            this, k_partial_shape.rank().compatible(0), "The 'K' input must be a scalar.");

    size_t k = 0;
    if (auto constant = ngraph::as_type_ptr<ngraph::opset3::Constant>(input_value(1).get_node_shared_ptr())) {
        const auto value = constant->cast_vector<int64_t>();
        NODE_VALIDATION_CHECK(this,
                              value.size() == 1,
                              "Only one value (scalar) should be provided as the 'K' input to TopK",
                              " (got ",
                              value.size(),
                              " elements).");

        NODE_VALIDATION_CHECK(this,
                              value[0] > 0,
                              "The value of 'K' must be a positive number.",
                              " (got ",
                              value[0],
                              ").");
        k = static_cast<size_t>(value[0]);
    }

    PartialShape output_shape{input_partial_shape};
    m_normalized_axis = ngraph::normalize_axis(this->description(), m_axis, output_shape.rank());
    if (k != 0) {
        output_shape[m_normalized_axis] = k;
    } else {
        auto max_k = maximum_value(input_value(1));
        const auto is_max_value_calculated = max_k.first;
        const auto calculated_max_value = max_k.second;
        if (is_max_value_calculated) {
            m_maximumK = calculated_max_value;
        }
    }

    output_shape[m_normalized_axis] = m_maximumK;

    NODE_VALIDATION_CHECK(this, output_shape.is_static(),
            "StaticShapeTopK output shape is not fully defined: ", output_shape);

    set_output_size(2);
    set_output_type(0, get_input_element_type(0), output_shape);
    set_output_type(1, m_index_element_type, output_shape);
}

void ngraph::vpu::op::StaticShapeTopK::set_axis(const int64_t axis) {
    const auto input_rank = get_input_partial_shape(0).rank();
    if (input_rank.is_static()) {
        m_normalized_axis = ngraph::normalize_axis(this->description(), axis, input_rank);
    } else {
        m_normalized_axis = UNKNOWN_NORMALIZED_AXIS;
    }
    m_axis = axis;
}

uint64_t ngraph::vpu::op::StaticShapeTopK::get_axis() const {
    NODE_VALIDATION_CHECK(
            this, m_normalized_axis != UNKNOWN_NORMALIZED_AXIS, "Normalized axis of TopK is unknown");

    return m_normalized_axis;
}

bool ngraph::vpu::op::StaticShapeTopK::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("mode", m_mode);
    visitor.on_attribute("sort", m_sort);
    visitor.on_attribute("index_element_type", m_index_element_type);
    return true;
}