// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/permute.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<permute> : public typed_program_node_base<permute> {
    using parent = typed_program_node_base<permute>;
    typed_program_node(const std::shared_ptr<permute> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    std::vector<uint16_t> get_permute_order() const { return get_primitive()->permute_order; }
    bool is_rotating_except_batch() const {
        // Target transform: Rotate feature dim to back to be taken as inner-most axis
        // ex) 0(b), 4(f), 1(z), 2(y), 3(x)
        // ex) 0(b), 3(f), 1(y), 2(x)
        auto& order = get_primitive()->permute_order;
        if ((int32_t) order[1] != order.size() - 1) return false;
        if ((int32_t) order[0] != 0) return false;
        for (int32_t i = 2; i < (int32_t) order.size(); ++i) {
            if ((int32_t)order[i] !=  (i - 1)) return false;
        }
        return true;
    }
};

using permute_node = typed_program_node<permute>;

template <>
class typed_primitive_inst<permute> : public typed_primitive_inst_base<permute> {
    using parent = typed_primitive_inst_base<permute>;

public:
    static layout calc_output_layout(permute_node const& node);
    static std::string to_string(permute_node const& node);

public:
    typed_primitive_inst(network& network, permute_node const& node);
};

using permute_inst = typed_primitive_inst<permute>;

}  // namespace cldnn
