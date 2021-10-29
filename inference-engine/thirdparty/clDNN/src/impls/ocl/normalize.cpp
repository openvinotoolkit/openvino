// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "normalize_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "cldnn/runtime/error_handler.hpp"
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

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<normalize_impl>(*this);
    }

protected:
     kernel_arguments_data get_arguments(typed_primitive_inst<normalize>& instance, int32_t split) const override {
        kernel_arguments_data args = parent::get_arguments(instance, split);
        args.scale_table = instance.scale_memory();
        return args;
    }

public:
    static primitive_impl* create(const normalize_node& arg) {
        auto norm_params = get_default_params<kernel_selector::normalize_params>(arg);
        auto norm_optional_params =
            get_default_optional_params<kernel_selector::normalize_optional_params>(arg.get_program());

        const auto& scale_layout = arg.scale().get_output_layout();

        norm_params.normMode = arg.get_primitive()->across_spatial ? kernel_selector::normalize_mode::ACROSS_SPATIAL
                                                                   : kernel_selector::normalize_mode::WITHIN_SPATIAL;
        norm_params.epsilon = arg.get_primitive()->epsilon;
        norm_params.scaleTable = convert_data_tensor(scale_layout).FlattenFeatureAndSpatials();

        auto& kernel_selector = kernel_selector::normalize_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(norm_params, norm_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto lrn = new normalize_impl(arg, best_kernels[0]);

        return lrn;
    }
};

namespace detail {

attach_normalize_impl::attach_normalize_impl() {
    implementation_map<normalize>::add(impl_types::ocl, normalize_impl::create, {
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::f32, format::yxfb),
        std::make_tuple(data_types::f16, format::yxfb),
        std::make_tuple(data_types::i8, format::yxfb),
        std::make_tuple(data_types::u8, format::yxfb),
        std::make_tuple(data_types::f32, format::byxf),
        std::make_tuple(data_types::f16, format::byxf),
        std::make_tuple(data_types::i8, format::byxf),
        std::make_tuple(data_types::u8, format::byxf),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
