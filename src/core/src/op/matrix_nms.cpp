// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/matrix_nms.hpp"

#include <cstring>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/matrix_nms.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
BWDCMP_RTTI_DEFINITION(ov::op::v8::MatrixNms);

op::v8::MatrixNms::MatrixNms() : NmsBase(m_attrs.output_type, m_attrs.nms_top_k, m_attrs.keep_top_k) {}

op::v8::MatrixNms::MatrixNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs)
    : NmsBase({boxes, scores}, m_attrs.output_type, m_attrs.nms_top_k, m_attrs.keep_top_k),
      m_attrs{attrs} {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v8::MatrixNms::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v8_MatrixNms_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2, "Number of inputs must be 2");

    return std::make_shared<op::v8::MatrixNms>(new_args.at(0), new_args.at(1), m_attrs);
}

bool op::v8::MatrixNms::validate() {
    NGRAPH_OP_SCOPE(v8_MatrixNms_validate);
    const auto validated = NmsBase::validate();

    NODE_VALIDATION_CHECK(this,
                          m_attrs.background_class >= -1,
                          "The 'background_class' must be great or equal -1. Got:",
                          m_attrs.background_class);
    return validated;
}

bool ngraph::op::v8::MatrixNms::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v8_MatrixNms_visit_attributes);

    visitor.on_attribute("sort_result_type", m_attrs.sort_result_type);
    visitor.on_attribute("output_type", m_attrs.output_type);
    visitor.on_attribute("nms_top_k", m_attrs.nms_top_k);
    visitor.on_attribute("keep_top_k", m_attrs.keep_top_k);
    visitor.on_attribute("sort_result_across_batch", m_attrs.sort_result_across_batch);
    visitor.on_attribute("score_threshold", m_attrs.score_threshold);
    visitor.on_attribute("background_class", m_attrs.background_class);
    visitor.on_attribute("decay_function", m_attrs.decay_function);
    visitor.on_attribute("gaussian_sigma", m_attrs.gaussian_sigma);
    visitor.on_attribute("post_threshold", m_attrs.post_threshold);
    visitor.on_attribute("normalized", m_attrs.normalized);

    return true;
}

std::ostream& ov::operator<<(std::ostream& s, const op::v8::MatrixNms::DecayFunction& type) {
    return s << as_string(type);
}

namespace ov {
template <>
NGRAPH_API EnumNames<ngraph::op::v8::MatrixNms::DecayFunction>&
EnumNames<ngraph::op::v8::MatrixNms::DecayFunction>::get() {
    static auto enum_names = EnumNames<ngraph::op::v8::MatrixNms::DecayFunction>(
        "op::v8::MatrixNms::DecayFunction",
        {{"gaussian", ngraph::op::v8::MatrixNms::DecayFunction::GAUSSIAN},
         {"linear", ngraph::op::v8::MatrixNms::DecayFunction::LINEAR}});
    return enum_names;
}

BWDCMP_RTTI_DEFINITION(AttributeAdapter<op::v8::MatrixNms::DecayFunction>);

}  // namespace ov
