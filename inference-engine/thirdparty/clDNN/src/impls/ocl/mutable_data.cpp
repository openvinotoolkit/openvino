// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mutable_data_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"

namespace cldnn {
namespace ocl {

struct mutable_data_impl : public typed_primitive_impl_ocl<mutable_data> {
    using parent = typed_primitive_impl_ocl<mutable_data>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<mutable_data_impl>(*this);
    }

public:
    static primitive_impl* create(mutable_data_node const& arg) { return new mutable_data_impl(arg, {}); }
};

namespace detail {

attach_mutable_data_impl::attach_mutable_data_impl() {
    implementation_map<mutable_data>::add(impl_types::ocl, mutable_data_impl::create, {});
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
