//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/experimental_detectron_generate_proposals.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v6::ExperimentalDetectronGenerateProposalsSingleImage,
                       "ExperimentalDetectronGenerateProposalsSingleImage",
                       6);

op::v6::ExperimentalDetectronGenerateProposalsSingleImage::
    ExperimentalDetectronGenerateProposalsSingleImage(const Output<Node>& im_info,
                                                      const Output<Node>& anchors,
                                                      const Output<Node>& deltas,
                                                      const Output<Node>& scores,
                                                      const Attributes& attrs)
    : Op({im_info, anchors, deltas, scores})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v6::ExperimentalDetectronGenerateProposalsSingleImage::clone_with_new_inputs(
    const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v6_ExperimentalDetectronGenerateProposalsSingleImage_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v6::ExperimentalDetectronGenerateProposalsSingleImage>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_attrs);
}

bool op::v6::ExperimentalDetectronGenerateProposalsSingleImage::visit_attributes(
    AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v6_ExperimentalDetectronGenerateProposalsSingleImage_visit_attributes);
    visitor.on_attribute("min_size", m_attrs.min_size);
    visitor.on_attribute("nms_threshold", m_attrs.nms_threshold);
    visitor.on_attribute("post_nms_count", m_attrs.post_nms_count);
    visitor.on_attribute("pre_nms_count", m_attrs.pre_nms_count);
    return true;
}

void op::v6::ExperimentalDetectronGenerateProposalsSingleImage::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v6_ExperimentalDetectronGenerateProposalsSingleImage_validate_and_infer_types);
    size_t post_nms_count = static_cast<size_t>(m_attrs.post_nms_count);
    auto input_et = get_input_element_type(0);
    set_output_type(0, input_et, Shape{post_nms_count, 4});
    set_output_type(1, input_et, Shape{post_nms_count});
}
