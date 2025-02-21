// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "lora_inst.h"
#include "lora.hpp"
#include "lora/lora_kernel_selector.h"
#include "lora/lora_kernel_base.h"

namespace cldnn {
namespace ocl {

struct lora_impl : typed_primitive_impl_ocl<lora> {
    using parent = typed_primitive_impl_ocl<lora>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::lora_kernel_selector;
    using kernel_params_t = kernel_selector::lora_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::lora_impl);

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<lora_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic() && _kernel_data.kernelName.length() != 0) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->GetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        auto params = get_default_params<kernel_selector::lora_params>(impl_param, is_shape_agnostic);

        for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }
        return params;
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        // If model loaded from cache, params are not initialized, so we create a new object and reuse it in the future
        if (_kernel_data.params == nullptr) {
            _kernel_data.params = std::make_shared<kernel_params_t>(get_kernel_params(impl_param, true));
        }

        update_shapes(*_kernel_data.params, impl_param);
        (_kernel_data.update_dispatch_data_func)(*_kernel_data.params, _kernel_data);
    }
};

std::unique_ptr<primitive_impl> LoraImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<lora>());
    return typed_primitive_impl_ocl<lora>::create<lora_impl>(static_cast<const lora_node&>(node), params);
}

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::lora_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::lora)
