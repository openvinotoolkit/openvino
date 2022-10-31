// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalize_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "normalize/normalize_kernel_selector.h"
#include "normalize/normalize_kernel_base.h"

#include <algorithm>

using namespace cldnn;

namespace cldnn {
namespace ocl {

struct normalize_impl : typed_primitive_impl_ocl<normalize> {
    using parent = typed_primitive_impl_ocl<normalize>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::normalize_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::normalize_params, kernel_selector::normalize_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<normalize_impl>(*this);
    }

protected:
     kernel_arguments_data get_arguments(const typed_primitive_inst<normalize>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);
        args.scale_table = instance.scale_memory();
        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<normalize>();
        auto params = get_default_params<kernel_selector::normalize_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::normalize_optional_params>(impl_param.get_program());

        auto scale_layout = impl_param.get_input_layout(1);

        params.normMode = primitive->across_spatial ? kernel_selector::normalize_mode::ACROSS_SPATIAL
                                                    : kernel_selector::normalize_mode::WITHIN_SPATIAL;
        params.epsilon = primitive->epsilon;
        if (format::is_simple_data_format(scale_layout.format)) {
            params.scaleTable = convert_data_tensor(scale_layout).FlattenFeatureAndSpatials();
        } else {
            params.scaleTable = convert_data_tensor(scale_layout);
        }
        return {params, optional_params};
    }

    static std::unique_ptr<primitive_impl> create(const normalize_node& arg, const kernel_impl_params& impl_param) {
        auto kernel_params = get_kernel_params(impl_param);
        auto& kernel_selector = kernel_selector_t::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(kernel_params.first, kernel_params.second);

        return make_unique<normalize_impl>(arg, best_kernel);
    }
};

namespace detail {

attach_normalize_impl::attach_normalize_impl() {
    auto types = {data_types::f16, data_types::f32, data_types::i8, data_types::u8};
    auto formats = {
        format::bfyx,
        format::yxfb,
        format::byxf,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
    };
    implementation_map<normalize>::add(impl_types::ocl, normalize_impl::create, types, formats);
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::normalize_impl, cldnn::object_type::NORMALIZE_IMPL)
