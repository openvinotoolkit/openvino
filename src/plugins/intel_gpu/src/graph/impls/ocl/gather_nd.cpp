// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "gather_nd.hpp"
#include "gather_nd_inst.h"
#include "gather/gather_nd_kernel_selector.h"
#include "gather/gather_nd_kernel_ref.h"

namespace cldnn {
namespace ocl {

struct gather_nd_impl : typed_primitive_impl_ocl<gather_nd> {
    using parent = typed_primitive_impl_ocl<gather_nd>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::gather_nd_kernel_selector;
    using kernel_params_t = kernel_selector::gather_nd_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::gather_nd_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<gather_nd_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<gather_nd>();
        auto params = get_default_params<kernel_selector::gather_nd_params>(impl_param);

        params.indices_rank = primitive->indices_rank;
        params.batch_dims = primitive->batch_dims;
        params.batch_merged_output = primitive->batch_merged_output;

        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param));
        }

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }
};

std::unique_ptr<primitive_impl> GatherNDImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<gather_nd>());
    return typed_primitive_impl_ocl<gather_nd>::create<gather_nd_impl>(static_cast<const gather_nd_node&>(node), params);
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::gather_nd_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gather_nd)
