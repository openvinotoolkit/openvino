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

#include <memory>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/experimental_detectron_roi_feature.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v6::ExperimentalDetectronROIFeatureExtractor,
                       "ExperimentalDetectronROIFeatureExtractor",
                       6);

op::v6::ExperimentalDetectronROIFeatureExtractor::ExperimentalDetectronROIFeatureExtractor(
    const OutputVector& args,
    const Attributes& attrs)
    : Op(args)
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

bool op::v6::ExperimentalDetectronROIFeatureExtractor::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("distribute_rois_between_levels", m_attrs.distribute_rois_between_levels);
    visitor.on_attribute("image_id", m_attrs.image_id);
    visitor.on_attribute("output_size", m_attrs.output_size);
    visitor.on_attribute("preserve_rois_order", m_attrs.preserve_rois_order);
    visitor.on_attribute("sampling_ratio", m_attrs.sampling_ratio);
    visitor.on_attribute("pyramid_scales", m_attrs.pyramid_scales);
    visitor.on_attribute("aligned", m_attrs.aligned);
    return true;
}
