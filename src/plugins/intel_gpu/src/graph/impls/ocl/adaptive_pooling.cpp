// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "adaptive_pooling_inst.h"
#include "adaptive_pooling/adaptive_pooling_kernel_ref.h"
#include "adaptive_pooling/adaptive_pooling_kernel_selector.h"

namespace cldnn {
namespace ocl {
struct adaptive_pooling_impl : public typed_primitive_impl_ocl<adaptive_pooling> {
    using parent = typed_primitive_impl_ocl<adaptive_pooling>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::adaptive_pooling_kernel_selector;
    using kernel_params_t = kernel_selector::adaptive_pooling_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::adaptive_pooling_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<adaptive_pooling_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<adaptive_pooling>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        auto desc = instance.get_typed_desc<adaptive_pooling>();

        // Legacy multi-output
        if (desc->num_outputs == 1 && desc->mode == adaptive_pooling_mode::max) {
            args.outputs.push_back(instance.dep_memory_ptr(2));
        }

        return args;
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<adaptive_pooling>();
        auto params = get_default_params<kernel_selector::adaptive_pooling_params>(impl_param);

        if (primitive->mode == adaptive_pooling_mode::average) {
            params.mode = kernel_selector::PoolType::AVG;
        } else {
            params.mode = kernel_selector::PoolType::MAX;
            params.poolIndexElementType = to_data_type(primitive->index_element_type);
            params.outputs_num = 2;
            if (primitive->num_outputs == 2) {
                params.outputs.push_back(convert_data_tensor(impl_param.get_output_layout(1)));
            } else {
                // Legacy multi-output
                params.outputs.push_back(convert_data_tensor(impl_param.get_input_layout(2)));
            }
        }

        return params;
    }
};

namespace detail {
attach_adaptive_pooling_impl::attach_adaptive_pooling_impl() {
    auto types = {data_types::f16, data_types::f32, data_types::i32, data_types::i64};
    auto formats = {
        format::bfyx,
        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::b_fs_zyx_fsv32,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
        format::bs_fs_zyx_bsv32_fsv16
    };

    implementation_map<adaptive_pooling>::add(impl_types::ocl, typed_primitive_impl_ocl<adaptive_pooling>::create<adaptive_pooling_impl>, types, formats);
}
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::adaptive_pooling_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::adaptive_pooling)
