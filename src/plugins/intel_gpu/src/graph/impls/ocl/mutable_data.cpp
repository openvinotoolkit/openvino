// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "mutable_data_inst.h"

namespace cldnn {
namespace ocl {

struct mutable_data_impl : public typed_primitive_impl_ocl<mutable_data> {
    using parent = typed_primitive_impl_ocl<mutable_data>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::mutable_data_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<mutable_data_impl>(*this);
    }

public:
    static std::unique_ptr<primitive_impl> create(mutable_data_node const& arg, const kernel_impl_params&) {
        return std::make_unique<mutable_data_impl>(kernel_selector::kernel_data{});
    }
};

namespace detail {

attach_mutable_data_impl::attach_mutable_data_impl() {
    implementation_map<mutable_data>::add(impl_types::ocl, mutable_data_impl::create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::mutable_data_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::mutable_data)
