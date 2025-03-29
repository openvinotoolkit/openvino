// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <intel_gpu/primitives/stft.hpp>

#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<STFT> : public typed_program_node_base<STFT> {
    using parent = typed_program_node_base<STFT>;
    typed_program_node(const std::shared_ptr<STFT> prim, program& prog) : parent(prim, prog) {}

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const {
        return get_dependency(idx);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {2, 3};
    }
};

using STFT_node = typed_program_node<STFT>;

template <>
class typed_primitive_inst<STFT> : public typed_primitive_inst_base<STFT> {
    using parent = typed_primitive_inst_base<STFT>;
    using parent::parent;

public:
    typed_primitive_inst(network& network, STFT_node const& desc);
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(STFT_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(STFT_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(STFT_node const& node);
};

using STFT_inst = typed_primitive_inst<STFT>;

}  // namespace cldnn
