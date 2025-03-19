// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/proposal.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
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
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using proposal_node = typed_program_node<proposal>;

template <>
class typed_primitive_inst<proposal> : public typed_primitive_inst_base<proposal> {
    using parent = typed_primitive_inst_base<proposal>;
    using parent::parent;

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

    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(proposal_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(proposal_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(proposal_node const& node);

    typed_primitive_inst(network& network, proposal_node const& desc);

    const std::vector<anchor>& get_anchors() const { return _anchors; }

private:
    std::vector<anchor> _anchors;
};

using proposal_inst = typed_primitive_inst<proposal>;

}  // namespace cldnn
