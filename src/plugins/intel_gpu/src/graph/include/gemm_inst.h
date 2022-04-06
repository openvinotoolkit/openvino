// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/gemm.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<gemm> : public typed_program_node_base<gemm> {
    using parent = typed_program_node_base<gemm>;

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return *get_dependency(idx).first; }
    size_t inputs_count() const { return this->get_primitive()->input_size(); }
};

using gemm_node = typed_program_node<gemm>;

template <>
class typed_primitive_inst<gemm> : public typed_primitive_inst_base<gemm> {
    using parent = typed_primitive_inst_base<gemm>;

public:
    static layout calc_output_layout(gemm_node const& node);
    static std::string to_string(gemm_node const& node);

public:
    typed_primitive_inst(network& network, gemm_node const& node);
};

using gemm_inst = typed_primitive_inst<gemm>;

}  // namespace cldnn
