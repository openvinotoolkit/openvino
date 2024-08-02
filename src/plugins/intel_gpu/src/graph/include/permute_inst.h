// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
        // ex) 0(b), 2(f), 3(x), 4(y), 1(z)
        // ex) 0(b), 2(f), 3(x), 1(y)
        auto& order = get_primitive()->permute_order;
        if ((int32_t) order[order.size() - 2] != order.size() - 1) return false;
        if ((int32_t) order[0] != 0) return false;
        for (int32_t i = 2; i < (int32_t) order.size(); ++i) {
            if ((int32_t)order[i - 1] != i) return false;
        }
        return true;
    }

    bool is_reverse_rotating_except_batch() const {
        // Target transform: Rotate feature dim to front to be taken as second outer axis
        // ex) 0(b), 4(f), 1(x), 2(y), 3(z)
        // ex) 0(b), 3(f), 1(x), 2(y)
        auto& order = get_primitive()->permute_order;
        if ((int32_t) order[order.size() - 2] != 1) return false;
        if ((int32_t) order[0] != 0) return false;
        for (int32_t i = 2; i < (int32_t) order.size(); ++i) {
            if ((int32_t)order[i] != i - 1) return false;
        }
        return true;
    }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using permute_node = typed_program_node<permute>;

template <>
class typed_primitive_inst<permute> : public typed_primitive_inst_base<permute> {
    using parent = typed_primitive_inst_base<permute>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(permute_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(permute_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(permute_node const& node);

public:
    typed_primitive_inst(network& network, permute_node const& node);
    void update_output_memory() override;

private:
    void on_execute() override;
};

using permute_inst = typed_primitive_inst<permute>;

}  // namespace cldnn
