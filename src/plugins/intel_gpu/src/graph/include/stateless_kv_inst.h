// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/stateless_kv.hpp"
#include "primitive_inst.h"

#include <optional>

namespace cldnn {

template <>
struct typed_program_node<stateless_kv> : public typed_program_node_base<stateless_kv> {
private:
    using parent = typed_program_node_base<stateless_kv>;

public:
    using parent::parent;

    program_node& input() const {
        return get_dependency(0);
    }

    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {2};
    }
};

using stateless_kv_node = typed_program_node<stateless_kv>;

template<>
class typed_primitive_inst<stateless_kv> : public typed_primitive_inst_base<stateless_kv> {
    using parent = typed_primitive_inst_base<stateless_kv>;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const stateless_kv_node& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const stateless_kv_node& node, const kernel_impl_params& impl_param);

    static std::string to_string(const stateless_kv_node& node);

    void update_output_memory() override;

    bool get_is_inplace() const {
        return m_is_inplace;
    }

    static std::optional<int64_t> compute_update_offset(const kernel_impl_params& impl_param, const stateless_kv& desc);
    void update_shape_info_tensor(const kernel_impl_params& params) override;

    typed_primitive_inst(network& network, const stateless_kv_node& desc);
    typed_primitive_inst(network& network) : parent(network) {}

private:
    void on_execute() override;
    bool m_is_inplace = false;
};

using stateless_kv_inst = typed_primitive_inst<stateless_kv>;

} // namespace cldnn
