// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/primitive.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "kernel_selector_params.h"
#include "meta_utils.h"

namespace cldnn {
struct fused_primitive_desc {
    explicit fused_primitive_desc(std::shared_ptr<const primitive> prim) : desc(prim) {}

    template <class PType>
    bool is_type() const {
        static_assert(meta::is_primitive<PType>::value,
            "Type argument fused_primitive_desc::is_type should be a non-const, non-volatile type derived from primitive");
        return desc->type == PType::type_id();
    }

    template <class PType>
    std::shared_ptr<const PType> typed_desc() const { return std::static_pointer_cast<const PType>(desc); }

    template<typename T>
    std::shared_ptr<T> get_typed_fuse_params() const {
        auto p = std::dynamic_pointer_cast<T>(f_param);
        if (!p)
            throw std::runtime_error("Invalid dynamic cast of fused parameters!");
        return p;
    }

    std::shared_ptr<const primitive> desc;

    layout input_layout = layout(data_types::f32, format::bfyx, tensor());
    layout output_layout = layout(data_types::f32, format::bfyx, tensor());

    std::shared_ptr<kernel_selector::fuse_params> f_param;

    std::vector<std::pair<primitive_id, size_t>> deps;
    std::map<primitive_id, size_t> fused_deps;
    size_t dep_start_idx;
    size_t total_num_deps = 0;

    activation_func activation;
    activation_additional_params activation_params = { 0.f, 0.f };
};
} // namespace cldnn
