// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "intel_gpu/primitives/qsa.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<qsa> : public typed_program_node_base<qsa> {
    using parent = typed_program_node_base<qsa>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const {
        return get_dependency(index);
    }

    // past_lens / subsequence_begins must be readable on the host so that the CM
    // implementation can size its internal buffers and build the Q1 work-group map.
    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {ov::intel_gpu::op::QSA::InputIdx::PAST_LENS, ov::intel_gpu::op::QSA::InputIdx::SUBSEQUENCE_BEGINS};
    }
};
using qsa_node = typed_program_node<qsa>;

template <>
class typed_primitive_inst<qsa> : public typed_primitive_inst_base<qsa> {
    using parent = typed_primitive_inst_base<qsa>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const qsa_node& node, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const qsa_node& node, const kernel_impl_params& impl_params);

    static std::string to_string(const qsa_node& node);
    typed_primitive_inst(network& network, const qsa_node& node);
};

using qsa_inst = typed_primitive_inst<qsa>;

}  // namespace cldnn
