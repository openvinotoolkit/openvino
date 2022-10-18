// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <intel_gpu/primitives/dft.hpp>

#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<dft> : public typed_program_node_base<dft> {
    using typed_program_node_base::typed_program_node_base;

    program_node& input() const {
        return get_dependency(0);
    }
};

using dft_node = typed_program_node<dft>;

template <>
class typed_primitive_inst<dft> : public typed_primitive_inst_base<dft> {
public:
    using typed_primitive_inst_base::typed_primitive_inst_base;

    static layout calc_output_layout(const dft_node& node, const kernel_impl_params& impl_param);
    static std::string to_string(const dft_node& node);
};

using dft_inst = typed_primitive_inst<dft>;

}  // namespace cldnn
