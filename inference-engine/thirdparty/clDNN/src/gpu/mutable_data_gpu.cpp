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

public:
    bool validate_impl(const typed_primitive_inst<mutable_data>& instance) const override {
        bool is_primary = instance.get_network().is_primary_stream();

        auto net_id = instance.get_network().get_id();
        auto mem_net_id = instance.output_memory().get_net_id();

        bool res = is_primary || net_id == mem_net_id;
        return res;
    }

    static primitive_impl* create(mutable_data_node const& arg) { return new mutable_data_gpu(arg, {}); }
};

namespace detail {

attach_mutable_data_gpu::attach_mutable_data_gpu() {
    implementation_map<mutable_data>::add({{engine_types::ocl, mutable_data_gpu::create}});
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
