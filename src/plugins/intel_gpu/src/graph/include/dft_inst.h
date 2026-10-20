// Copyright (C) 2018-2026 Intel Corporation
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

    program_node& input(size_t idx = 0) const {
        return get_dependency(idx);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override {
        if (this->get_dependencies().size() == 3) {
            return {1, 2};
        }
        return {1};
    }
};

using dft_node = typed_program_node<dft>;

template <>
class typed_primitive_inst<dft> : public typed_primitive_inst_base<dft> {
    using parent = typed_primitive_inst_base<dft>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const dft_node& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const dft_node& node, const kernel_impl_params& impl_param);
    static std::string to_string(const dft_node& node);

    typed_primitive_inst(network& network, const dft_node& node);
};

using dft_inst = typed_primitive_inst<dft>;

}  // namespace cldnn
