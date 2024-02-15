// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <intel_gpu/primitives/dft.hpp>

#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<dft> : public typed_program_node_base<dft> {
    using parent = typed_program_node_base<dft>;

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    std::vector<size_t> get_shape_infer_dependencies() const override {
        if (this->get_dependencies().size() == 3)
            return {1, 2};
        else
            return {1};
    }
};

using dft_node = typed_program_node<dft>;

template <>
class typed_primitive_inst<dft> : public typed_primitive_inst_base<dft> {
    using parent = typed_primitive_inst_base<dft>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(dft_node const& /*node*/, kernel_impl_params const& impl_param);
    static layout calc_output_layout(dft_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(dft_node const& node);

    typed_primitive_inst(network& network, dft_node const& node);
};

using dft_inst = typed_primitive_inst<dft>;

}  // namespace cldnn
