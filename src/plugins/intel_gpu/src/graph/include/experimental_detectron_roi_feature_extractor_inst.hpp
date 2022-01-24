// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "intel_gpu/primitives/experimental_detectron_roi_feature_extractor.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<experimental_detectron_roi_feature_extractor> : public typed_program_node_base<experimental_detectron_roi_feature_extractor> {
    using parent = typed_program_node_base<experimental_detectron_roi_feature_extractor>;
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
};

using experimental_detectron_roi_feature_extractor_node = typed_program_node<experimental_detectron_roi_feature_extractor>;

template <>
struct typed_primitive_inst<experimental_detectron_roi_feature_extractor> : public typed_primitive_inst_base<experimental_detectron_roi_feature_extractor> {
    using parent = typed_primitive_inst_base<experimental_detectron_roi_feature_extractor>;
    using parent::parent;

public:
    size_t inputs_memory_count() const;
    void copy_rois_input_to_second_output() const;

    static layout calc_output_layout(experimental_detectron_roi_feature_extractor_node const& node);
    static std::string to_string(experimental_detectron_roi_feature_extractor_node const& node);

private:
    memory::ptr second_output_memory() const;
    memory::ptr rois_memory() const;
};

using experimental_detectron_roi_feature_extractor_inst = typed_primitive_inst<experimental_detectron_roi_feature_extractor>;
}  // namespace cldnn
