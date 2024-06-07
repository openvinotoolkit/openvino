// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "extract_image_patches_inst.h"
#include "extract_image_patches/extract_image_patches_kernel_selector.h"
#include "extract_image_patches/extract_image_patches_kernel_ref.h"

namespace cldnn {
namespace ocl {
static inline std::string pad_to_string(ov::op::PadType pad) {
    switch (pad) {
        case ov::op::PadType::SAME_UPPER: return "same_upper";
        case ov::op::PadType::SAME_LOWER: return "same_lower";
        case ov::op::PadType::VALID: return "valid";
        default: OPENVINO_THROW("Unsupported pad type in ExtractImagePatches primitive ", pad);
    }

    return "";
}

struct extract_image_patches_impl : typed_primitive_impl_ocl<extract_image_patches> {
    using parent = typed_primitive_impl_ocl<extract_image_patches>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::extract_image_patches_kernel_selector;
    using kernel_params_t = kernel_selector::extract_image_patches_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::extract_image_patches_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<extract_image_patches_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<extract_image_patches>();
        auto params = get_default_params<kernel_selector::extract_image_patches_params>(impl_param);

        std::vector<uint32_t> sizes;
        std::vector<uint32_t> strides;
        std::vector<uint32_t> rates;
        for (auto size : primitive->sizes) {
            sizes.push_back(static_cast<uint32_t>(size));
        }
        for (auto stride : primitive->strides) {
            strides.push_back(static_cast<uint32_t>(stride));
        }
        for (auto rate : primitive->rates) {
            rates.push_back(static_cast<uint32_t>(rate));
        }

        params.sizes = sizes;
        params.strides = strides;
        params.rates = rates;
        params.auto_pad = pad_to_string(primitive->auto_pad);

        return params;
    }
};

namespace detail {

attach_extract_image_patches_impl::attach_extract_image_patches_impl() {
    implementation_map<extract_image_patches>::add(impl_types::ocl, typed_primitive_impl_ocl<extract_image_patches>::create<extract_image_patches_impl>, {
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i64, format::bfyx),
        std::make_tuple(data_types::i8, format::bfyx),
        std::make_tuple(data_types::u8, format::bfyx),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f16, format::bfyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::extract_image_patches_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::extract_image_patches)
