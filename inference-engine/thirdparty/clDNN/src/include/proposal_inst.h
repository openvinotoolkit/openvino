/*
// Copyright (c) 2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/CPP/proposal.hpp"
#include "primitive_inst.h"
#include <string>
#include <vector>

namespace cldnn {

template <>
struct typed_program_node<proposal> : public typed_program_node_base<proposal> {
    using parent = typed_program_node_base<proposal>;
    using parent::parent;

    program_node& cls_score() const { return get_dependency(0); }
    program_node& bbox_pred() const { return get_dependency(1); }
    program_node& image_info() const { return get_dependency(2); }
};

using proposal_node = typed_program_node<proposal>;

template <>
class typed_primitive_inst<proposal> : public typed_primitive_inst_base<proposal> {
    using parent = typed_primitive_inst_base<proposal>;

public:
    struct anchor {
        float start_x;
        float start_y;
        float end_x;
        float end_y;

        anchor() { start_x = start_y = end_x = end_y = 0.0f; }

        anchor(float s_x, float s_y, float e_x, float e_y) {
            start_x = s_x;
            start_y = s_y;
            end_x = e_x;
            end_y = e_y;
        }
    };

    // indices of the memory objects used by the layer
    enum input_index { cls_scores_index, bbox_pred_index, image_info_index, proposal_probabilities_out };

    // TODO(ruv): missign validation?? for image_info dimensions? also faster r-cnn expected it to be dim3 while the new
    // networks expect dim 6!!! ([5] being unused)
    // TODO(ruv): we should be able to set dims[3]=dim[4]=1 if not provided

    // indices of the image info parameters inside the image_info memory object (the object
    // is an integer array of these parameters)
    enum image_info_size_index {
        image_info_height_index,
        image_info_width_index,
        image_info_depth_index,
        image_info_scale_min_bbox_y,
        image_info_scale_min_bbox_x,
        image_info_scale_depth_index,
    };

    static layout calc_output_layout(proposal_node const& node);
    static std::string to_string(proposal_node const& node);

public:
    typed_primitive_inst(network_impl& network, proposal_node const& desc);

    const std::vector<anchor>& get_anchors() const { return _anchors; }

private:
    std::vector<anchor> _anchors;
};

using proposal_inst = typed_primitive_inst<proposal>;

}  // namespace cldnn
