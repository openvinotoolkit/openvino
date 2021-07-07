// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mutable_data_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"

namespace cldnn {
namespace gpu {

struct mutable_data_gpu : public typed_primitive_gpu_impl<mutable_data> {
    using parent = typed_primitive_gpu_impl<mutable_data>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<mutable_data_gpu>(*this);
    }

public:
    static primitive_impl* create(mutable_data_node const& arg) { return new mutable_data_gpu(arg, {}); }
};

namespace detail {

attach_mutable_data_gpu::attach_mutable_data_gpu() {
    implementation_map<mutable_data>::add({{engine_types::ocl, mutable_data_gpu::create}});
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
