// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/experimental_detectron_roi_feature_extractor.hpp"
#include "primitive_inst.h"

namespace cldnn {

using experimental_detectron_roi_feature_extractor_node = typed_program_node<experimental_detectron_roi_feature_extractor>;

template <>
struct typed_primitive_inst<experimental_detectron_roi_feature_extractor> : public typed_primitive_inst_base<experimental_detectron_roi_feature_extractor> {
    using parent = typed_primitive_inst_base<experimental_detectron_roi_feature_extractor>;
    using parent::parent;

public:
    size_t inputs_memory_count() const;
    void copy_rois_input_to_second_output() const;

    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(experimental_detectron_roi_feature_extractor_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(experimental_detectron_roi_feature_extractor_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(experimental_detectron_roi_feature_extractor_node const& node);

private:
    memory::ptr second_output_memory() const;
    memory::ptr rois_memory() const;
};

using experimental_detectron_roi_feature_extractor_inst = typed_primitive_inst<experimental_detectron_roi_feature_extractor>;
}  // namespace cldnn
