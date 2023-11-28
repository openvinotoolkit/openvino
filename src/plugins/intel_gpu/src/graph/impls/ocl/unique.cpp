// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"
#include "unique/unique_kernel_ref.hpp"
#include "unique/unique_kernel_selector.hpp"
#include "unique_inst.hpp"

namespace cldnn {
namespace ocl {

struct unique_count_impl : typed_primitive_impl_ocl<unique_count> {
    using parent = typed_primitive_impl_ocl<unique_count>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::unique_count_kernel_selector;
    using kernel_params_t =
        std::pair<kernel_selector::unique_count_params, kernel_selector::unique_count_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::unique_count_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<unique_count_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->SetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<unique_count>();
        auto params = get_default_params<kernel_selector::unique_count_params>(impl_param, is_shape_agnostic);
        auto optional_params =
            get_default_optional_params<kernel_selector::unique_count_optional_params>(impl_param.get_program());

        params.flattened = primitive->flattened;
        params.axis = primitive->axis;

        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
    }
};

namespace detail {

attach_unique_count_impl::attach_unique_count_impl() {
    auto types = {
        data_types::u8,
        data_types::i8,
        data_types::f16,
        data_types::f32,
        data_types::i32,
        data_types::i64,
    };

    auto formats = {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv16_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,

        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,

        format::bfwzyx,
    };

    implementation_map<unique_count>::add(impl_types::ocl,
                                          shape_types::any,
                                          typed_primitive_impl_ocl<unique_count>::create<unique_count_impl>,
                                          types,
                                          formats);
}
}  // namespace detail

struct unique_gather_impl : typed_primitive_impl_ocl<unique_gather> {
    using parent = typed_primitive_impl_ocl<unique_gather>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::unique_gather_kernel_selector;
    using kernel_params_t =
        std::pair<kernel_selector::unique_gather_params, kernel_selector::unique_gather_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::unique_gather)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<unique_gather_impl>(*this);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        if (is_dynamic()) {
            auto& kernel_selector = kernel_selector_t::Instance();
            auto kernel_impl = kernel_selector.GetImplementation(_kernel_data.kernelName);
            kernel_impl->SetUpdateDispatchDataFunc(_kernel_data);
        }
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool is_shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<unique_gather>();
        auto params = get_default_params<kernel_selector::unique_gather_params>(impl_param, is_shape_agnostic);
        auto optional_params =
            get_default_optional_params<kernel_selector::unique_gather_optional_params>(impl_param.get_program());

        params.flattened = primitive->flattened;
        params.axis = primitive->axis;
        params.sorted = primitive->sorted;

        for (auto i = 1U; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts.at(i)));
        }

        for (auto i = 1U; i < impl_param.output_layouts.size(); ++i) {
            params.outputs.push_back(convert_data_tensor(impl_param.output_layouts.at(i)));
        }

        return {params, optional_params};
    }

    void update_dispatch_data(const kernel_impl_params& impl_param) override {
        auto kernel_params = get_kernel_params(impl_param, true);
        (_kernel_data.update_dispatch_data_func)(kernel_params.first, _kernel_data);
    }
};

namespace detail {

attach_unique_gather_impl::attach_unique_gather_impl() {
    auto types = {
        data_types::u8,
        data_types::i8,
        data_types::f16,
        data_types::f32,
        data_types::i32,
        data_types::i64,
    };

    auto formats = {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv16_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,

        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,

        format::bfwzyx,
    };

    implementation_map<unique_gather>::add(impl_types::ocl,
                                           shape_types::any,
                                           typed_primitive_impl_ocl<unique_gather>::create<unique_gather_impl>,
                                           types,
                                           formats);
}
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::unique_count_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::unique_gather_impl)
