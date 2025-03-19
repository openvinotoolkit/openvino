// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "permute_inst.h"
#include "permute/permute_kernel_selector.h"
#include "permute/permute_kernel_ref.h"

namespace cldnn {
namespace ocl {

namespace {
// This helper function is needed to convert permute order from OV format (bfyx) into cldnn format (bfxy)
inline std::vector<uint16_t> convert_permute_order(const std::vector<uint16_t>& ie_order, size_t rank = 0) {
    std::vector<uint16_t> ie_order_aligned = ie_order;
    // if order size is less than 4 - fill the rest with just copy
    rank = std::max(rank, (size_t)4);
    for (auto o = ie_order_aligned.size(); o < rank; o++)
        ie_order_aligned.push_back((uint16_t)o);

    std::vector<uint16_t> cldnn_order;
    // 1. Switch permute order values for spatial dims
    for (auto const& o : ie_order_aligned) {
        if (o >= 2)
            cldnn_order.push_back(1 + static_cast<uint16_t>(ie_order_aligned.size()) - o);
        else
            cldnn_order.push_back(o);
    }

    // 2. Swap spatial positions
    for (int i = 0; i < (static_cast<int>(cldnn_order.size()) - 2) / 2; i++) {
        std::swap(cldnn_order[2 + i], cldnn_order[1 + cldnn_order.size() - (2 + i)]);
    }

    return cldnn_order;
}
}  // namespace

struct permute_impl : typed_primitive_impl_ocl<permute> {
    using parent = typed_primitive_impl_ocl<permute>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::permute_kernel_selector;
    using kernel_params_t = kernel_selector::permute_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::permute_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<permute_impl, kernel_params_t>(*this);
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
        const auto& primitive = impl_param.typed_desc<permute>();
        auto params = get_default_params<kernel_selector::permute_params>(impl_param, is_shape_agnostic);

        auto in_rank = impl_param.get_input_layout(0).get_rank();
        auto permute_order = convert_permute_order(primitive->permute_order, in_rank);
        params.order = permute_order;

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

namespace detail {

attach_permute_impl::attach_permute_impl() {
    implementation_map<permute>::add(impl_types::ocl, shape_types::static_shape, typed_primitive_impl_ocl<permute>::create<permute_impl>, {});

    auto dyn_types = {
        data_types::f32,
        data_types::f16,
        data_types::i8,
        data_types::u8,
        data_types::i32
    };

    auto dyn_formats = {
        format::bfyx,
        format::bfzyx,
        format::bfwzyx,
        format::bfuwzyx,
        format::bfvuwzyx,
    };

    implementation_map<permute>::add(impl_types::ocl,
                                     shape_types::dynamic_shape,
                                     typed_primitive_impl_ocl<permute>::create<permute_impl>,
                                     dyn_types,
                                     dyn_formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::permute_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::permute)
