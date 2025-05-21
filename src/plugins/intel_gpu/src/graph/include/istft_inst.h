// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <intel_gpu/primitives/istft.hpp>

#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<ISTFT> : public typed_program_node_base<ISTFT> {
    using parent = typed_program_node_base<ISTFT>;
    typed_program_node(const std::shared_ptr<ISTFT> prim, program& prog) : parent(prim, prog) {}

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const {
        return get_dependency(idx);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override {
        if (this->get_dependencies().size() == 5)
            return {2, 3, 4};
        else
            return {2, 3};
    }
};

using ISTFT_node = typed_program_node<ISTFT>;

template <>
class typed_primitive_inst<ISTFT> : public typed_primitive_inst_base<ISTFT> {
    using parent = typed_primitive_inst_base<ISTFT>;
    using parent::parent;

public:
    typed_primitive_inst(network& network, ISTFT_node const& desc);
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(ISTFT_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(ISTFT_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(ISTFT_node const& node);
    bool need_reset_output_memory() const override {
        return true;
    }
};

using ISTFT_inst = typed_primitive_inst<ISTFT>;

}  // namespace cldnn
