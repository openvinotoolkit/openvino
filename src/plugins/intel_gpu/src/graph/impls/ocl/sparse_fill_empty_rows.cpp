// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "sparse_fill_empty_rows_inst.h"
#include "sparse_fill_empty_rows/sparse_fill_empty_rows_kernel_ref.h"
#include "sparse_fill_empty_rows/sparse_fill_empty_rows_kernel_selector.h"

namespace cldnn::ocl {

struct sparse_fill_empty_rows_impl : typed_primitive_impl_ocl<sparse_fill_empty_rows> {
    using parent = typed_primitive_impl_ocl<sparse_fill_empty_rows>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::sparse_fill_empty_rows_kernel_selector;
    using kernel_params_t = kernel_selector::sparse_fill_empty_rows_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::sparse_fill_empty_rows_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<sparse_fill_empty_rows_impl, kernel_params_t>(*this);
    }

public:
        static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param, bool shape_agnostic = false) {
        const auto& primitive = impl_param.typed_desc<sparse_fill_empty_rows>();
        auto params = get_default_params<kernel_selector::sparse_fill_empty_rows_params>(impl_param, shape_agnostic);

        // Manually add all inputs/outputs except for the first ones
        for (size_t i = 1; i < impl_param.input_layouts.size(); ++i) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }
        for (size_t i = 1; i < impl_param.output_layouts.size(); ++i) {
            params.outputs.push_back(convert_data_tensor(impl_param.get_output_layout(i)));
        }
        return params;
    }

    // [NOTE]: Has to be added as a separete static function, since it is called via static dispatching in
    // typed_primitive_impl_ocl::create()..
    //static kernel_impl_params static_canonicalize_shapes(const kernel_impl_params& impl_params) {
    //    auto updated_impl_params = canonicalize_fused_shapes(impl_params);
    //    // Debug prints for input layouts
    //    for (size_t i = 0; i < updated_impl_params.input_layouts.size(); ++i) {
    //        const auto& layout = updated_impl_params.input_layouts[i];
    //        std::cout << "[sparse_fill_empty_rows] Setting input_layout[" << i << "]: "
    //              << "partial_shape=" << layout.get_partial_shape().to_string()
    //              << std::endl;
    //    }
//
    //    // Debug prints for output layouts
    //    for (size_t i = 0; i < updated_impl_params.output_layouts.size(); ++i) {
    //        const auto& layout = updated_impl_params.output_layouts[i];
    //        std::cout << "[sparse_fill_empty_rows] Setting output_layout[" << i << "]: "
    //              << "partial_shape=" << layout.get_partial_shape().to_string()
    //              << std::endl;
    //    }
    //    for (auto& input_layout : updated_impl_params.input_layouts) {
    //        input_layout.set_partial_shape(extend_shape_to_rank_from_begin(input_layout.get_partial_shape()));
    //    }
//
    //    for (auto& output_layout : updated_impl_params.output_layouts) {
    //        output_layout.set_partial_shape(extend_shape_to_rank_from_begin(output_layout.get_partial_shape()));
    //    }
//
    //    return updated_impl_params;
    //}
//
    //kernel_impl_params canonicalize_shapes(const kernel_impl_params& impl_params) const override {
    //    return static_canonicalize_shapes(impl_params);
    //}
};

namespace detail {

attach_sparse_fill_empty_rows_impl::attach_sparse_fill_empty_rows_impl() {
    auto types = {data_types::f16, data_types::f32, data_types::i8, data_types::u8, data_types::i32, data_types::i64};
    auto formats = {format::bfyx};
    implementation_map<sparse_fill_empty_rows>::add(
        impl_types::ocl,
        shape_types::static_shape,
        typed_primitive_impl_ocl<sparse_fill_empty_rows>::create<sparse_fill_empty_rows_impl>,
        types,
        formats);
}

}  // namespace detail
}  // namespace cldnn::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::sparse_fill_empty_rows_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::sparse_fill_empty_rows)
