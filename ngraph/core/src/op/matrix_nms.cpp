// Copyright (C) 2018-2021 Intel Corporation
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

NGRAPH_RTTI_DEFINITION(op::v8::MatrixNms, "MatrixNms", 8);

op::v8::MatrixNms::MatrixNms(const Output<Node>& boxes,
                             const Output<Node>& scores,
                             const SortResultType sort_result_type,
                             const bool sort_result_across_batch,
                             const ngraph::element::Type& output_type,
                             const float score_threshold,
                             const int nms_top_k,
                             const int keep_top_k,
                             const int background_class,
                             const DecayFunction decay_function,
                             const float gaussian_sigma,
                             const float post_threshold,
                             const bool normalized)
    : NmsBase(boxes, scores, sort_result_type, output_type, nms_top_k, keep_top_k)
    , m_sort_result_across_batch{sort_result_across_batch}
    , m_score_threshold{score_threshold}
    , m_background_class{background_class}
    , m_decay_function{decay_function}
    , m_gaussian_sigma{gaussian_sigma}
    , m_post_threshold{post_threshold}
    , m_normalized{normalized}
{
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v8::MatrixNms::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v8_MatrixNms_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2, "Number of inputs must be 2");

    return std::make_shared<op::v8::MatrixNms>(new_args.at(0),
                                               new_args.at(1),
                                               m_sort_result_type,
                                               m_sort_result_across_batch,
                                               m_output_type,
                                               m_score_threshold,
                                               m_nms_top_k,
                                               m_keep_top_k,
                                               m_background_class,
                                               m_decay_function,
                                               m_gaussian_sigma,
                                               m_post_threshold,
                                               m_normalized);
}

void op::v8::MatrixNms::validate()
{
    NmsBase::validate();

    NODE_VALIDATION_CHECK(this,
                          m_background_class >= -1,
                          "The 'background_class' must be great or equal -1. Got:",
                          m_background_class);
}

bool ngraph::op::v8::MatrixNms::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v8_MatrixNms_visit_attributes);
    NmsBase::visit_attributes(visitor);

    visitor.on_attribute("sort_result_across_batch", m_sort_result_across_batch);
    visitor.on_attribute("score_threshold", m_score_threshold);
    visitor.on_attribute("background_class", m_background_class);
    visitor.on_attribute("decay_function", m_decay_function);
    visitor.on_attribute("gaussian_sigma", m_gaussian_sigma);
    visitor.on_attribute("post_threshold", m_post_threshold);
    visitor.on_attribute("normalized", m_normalized);

    return true;
}

namespace ngraph
{
    template <>
    EnumNames<op::v8::MatrixNms::DecayFunction>& EnumNames<op::v8::MatrixNms::DecayFunction>::get()
    {
        static auto enum_names = EnumNames<op::v8::MatrixNms::DecayFunction>(
            "op::v8::MatrixNms::DecayFunction",
            {{"gaussian", op::v8::MatrixNms::DecayFunction::GAUSSIAN},
             {"linear", op::v8::MatrixNms::DecayFunction::LINEAR}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::v8::MatrixNms::DecayFunction>::type_info;

    std::ostream& operator<<(std::ostream& s, const op::v8::MatrixNms::DecayFunction& type)
    {
        return s << as_string(type);
    }
} // namespace ngraph
