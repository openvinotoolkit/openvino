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

#include <algorithm>
#include <memory>
#include <utility>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/experimental_detectron_roi_feature.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v6::ExperimentalDetectronROIFeatureExtractor,
                       "ExperimentalDetectronROIFeatureExtractor",
                       6);

op::v6::ExperimentalDetectronROIFeatureExtractor::ExperimentalDetectronROIFeatureExtractor(
    const OutputVector& args, const Attributes& attrs)
    : Op(args)
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

op::v6::ExperimentalDetectronROIFeatureExtractor::ExperimentalDetectronROIFeatureExtractor(
    const NodeVector& args, const Attributes& attrs)
    : ExperimentalDetectronROIFeatureExtractor(as_output_vector(args), attrs)
{
}

bool op::v6::ExperimentalDetectronROIFeatureExtractor::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v6_ExperimentalDetectronROIFeatureExtractor_visit_attributes);
    visitor.on_attribute("output_size", m_attrs.output_size);
    visitor.on_attribute("sampling_ratio", m_attrs.sampling_ratio);
    visitor.on_attribute("pyramid_scales", m_attrs.pyramid_scales);
    visitor.on_attribute("aligned", m_attrs.aligned);
    return true;
}

void op::v6::ExperimentalDetectronROIFeatureExtractor::validate_and_infer_types()
{
    NGRAPH_OP_SCOPE(v6_ExperimentalDetectronROIFeatureExtractor_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() >= 2, "At least two argument required.");

    auto rois_shape = get_input_partial_shape(0);
    auto input_et = get_input_element_type(0);

    PartialShape out_shape = {
        Dimension::dynamic(), Dimension::dynamic(), m_attrs.output_size, m_attrs.output_size};

    PartialShape out_rois_shape = {Dimension::dynamic(), 4};
    if (rois_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(
            this, rois_shape.rank().get_length() == 2, "Input rois rank must be equal to 2.");

        auto input_rois_last_dim_intersection_with_4 = rois_shape[1] & Dimension(4);
        NODE_VALIDATION_CHECK(this,
                              !input_rois_last_dim_intersection_with_4.get_interval().empty(),
                              "The last dimension of the 'input_rois' input must be equal to 4. "
                              "Got: ",
                              rois_shape[1]);

        out_shape[0] = rois_shape[0];
        out_rois_shape[0] = rois_shape[0];
    }

    size_t num_of_inputs = get_input_size();
    Dimension channels_intersection;

    for (size_t i = 1; i < num_of_inputs; i++)
    {
        auto current_shape = get_input_partial_shape(i);
        auto current_rank = current_shape.rank();

        if (current_rank.is_static())
        {
            NODE_VALIDATION_CHECK(this,
                                  current_rank.get_length() == 4,
                                  "Rank of each element of the pyramid must be equal to 4. Got: ",
                                  current_rank);

            auto first_dim_intersection_with_1 = current_shape[0] & Dimension(1);
            NODE_VALIDATION_CHECK(this,
                                  !first_dim_intersection_with_1.get_interval().empty(),
                                  "The first dimension of each pyramid element must be equal to 1. "
                                  "Got: ",
                                  current_shape[0]);

            channels_intersection &= current_shape[1];
        }
    }

    NODE_VALIDATION_CHECK(this,
                          !channels_intersection.get_interval().empty(),
                          "The number of channels must be the same for all layers of the pyramid.");

    out_shape[1] = channels_intersection;

    set_output_type(0, input_et, out_shape);
    set_output_type(1, input_et, out_rois_shape);
}

shared_ptr<Node> op::v6::ExperimentalDetectronROIFeatureExtractor::clone_with_new_inputs(
    const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v6_ExperimentalDetectronROIFeatureExtractor_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v6::ExperimentalDetectronROIFeatureExtractor>(new_args, m_attrs);
}
