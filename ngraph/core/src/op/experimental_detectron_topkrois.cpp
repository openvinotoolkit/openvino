//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "ngraph/op/experimental_detectron_topkrois.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v6::ExperimentalDetectronTopKROIs, "ExperimentalDetectronTopKROIs", 6);

op::v6::ExperimentalDetectronTopKROIs::ExperimentalDetectronTopKROIs(const Output<Node>& input_rois,
                                                                     const Output<Node>& rois_probs,
                                                                     size_t max_rois)
    : Op({input_rois, rois_probs})
    , m_max_rois(max_rois)
{
    constructor_validate_and_infer_types();
}

bool op::v6::ExperimentalDetectronTopKROIs::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v6_ExperimentalDetectronTopKROIs_visit_attributes);
    visitor.on_attribute("max_rois", m_max_rois);
    return true;
}

shared_ptr<Node>
    op::v6::ExperimentalDetectronTopKROIs::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v6_ExperimentalDetectronTopKROIs_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v6::ExperimentalDetectronTopKROIs>(
        new_args.at(0), new_args.at(1), m_max_rois);
}

void op::v6::ExperimentalDetectronTopKROIs::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v6_ExperimentalDetectronTopKROIs_validate_and_infer_types);
    const auto input_rois_shape = get_input_partial_shape(0);
    const auto rois_probs_shape = get_input_partial_shape(1);

    set_output_type(0, get_input_element_type(0), Shape{m_max_rois, 4});

    if (input_rois_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              input_rois_shape.rank().get_length() == 2,
                              "The 'input_rois' input is expected to be a 2D. Got: ",
                              input_rois_shape);

        NODE_VALIDATION_CHECK(this,
                              input_rois_shape[1] == 4,
                              "The second dimension of 'input_rois' should be 4. Got: ",
                              input_rois_shape[1]);
    }
    if (rois_probs_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              rois_probs_shape.rank().get_length() == 1,
                              "The 'rois_probs' input is expected to be a 1D. Got: ",
                              rois_probs_shape);
    }
    if (input_rois_shape.rank().is_static() && rois_probs_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              input_rois_shape[0] == rois_probs_shape[0],
                              "Number of rois and number of probabilities should be equal. Got: ",
                              input_rois_shape[0],
                              rois_probs_shape[0]);
    }
}
