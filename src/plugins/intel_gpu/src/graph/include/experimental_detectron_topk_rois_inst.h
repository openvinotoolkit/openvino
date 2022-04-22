// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/experimental_detectron_topk_rois.hpp"
#include "primitive_inst.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {

template<>
struct typed_program_node<experimental_detectron_topk_rois> : public typed_program_node_base<experimental_detectron_topk_rois> {
    using parent = typed_program_node_base<experimental_detectron_topk_rois>;
public:
    using parent::parent;

    const program_node &input(std::size_t index = 0) const { return *get_dependency(index).first; }
};

using experimental_detectron_topk_rois_node = typed_program_node<experimental_detectron_topk_rois>;

template<>
class typed_primitive_inst<experimental_detectron_topk_rois> : public typed_primitive_inst_base<experimental_detectron_topk_rois> {
    using parent = typed_primitive_inst_base<experimental_detectron_topk_rois>;

public:
    static layout calc_output_layout(experimental_detectron_topk_rois_node const &node);

    static std::string to_string(experimental_detectron_topk_rois_node const &node);

public:
    typed_primitive_inst(network &network, experimental_detectron_topk_rois_node const &desc);
};

using experimental_detectron_topk_rois_inst = typed_primitive_inst<experimental_detectron_topk_rois>;

} // namespace cldnn
