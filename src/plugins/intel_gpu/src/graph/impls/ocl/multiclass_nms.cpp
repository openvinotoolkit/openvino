// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/multiclass_nms.hpp"
#include "primitive_base.hpp"

#include "multiclass_nms_inst.h"
#include "multiclass_nms/multiclass_nms_kernel_ref.h"
#include "multiclass_nms/multiclass_nms_kernel_selector.h"

namespace cldnn {
namespace ocl {
namespace {
kernel_selector::SortResultType get_sort_result_type(const ov::op::util::MulticlassNmsBase::SortResultType sort_result_type) {
    switch (sort_result_type) {
        case ov::op::util::MulticlassNmsBase::SortResultType::CLASSID:
            return kernel_selector::SortResultType::CLASSID;
            break;
        case ov::op::util::MulticlassNmsBase::SortResultType::SCORE:
            return kernel_selector::SortResultType::SCORE;
            break;
        case ov::op::util::MulticlassNmsBase::SortResultType::NONE:
            return kernel_selector::SortResultType::NONE;
            break;
        default:
            throw std::runtime_error{"Not supported sort result type"};
    }
    return kernel_selector::SortResultType::NONE;
}

}  // namespace

struct multiclass_nms_impl : public typed_primitive_impl_ocl<multiclass_nms> {
    using parent = typed_primitive_impl_ocl<multiclass_nms>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::multiclass_nms_kernel_selector;
    using kernel_params_t = kernel_selector::multiclass_nms_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::multiclass_nms_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<multiclass_nms_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<multiclass_nms>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);
        // Legacy multi-output
        if (instance.desc()->num_outputs == 1) {
            args.outputs.push_back(instance.output_indices_memory());
            args.outputs.push_back(instance.output_num_memory());
        }
        return args;
    }

public:
   static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<multiclass_nms>();
        auto params = get_default_params<kernel_selector::multiclass_nms_params>(impl_param);

        const auto& attrs = primitive->attrs;

        params.sort_result_type = get_sort_result_type(attrs.sort_result_type);
        params.sort_result_across_batch = attrs.sort_result_across_batch;
        params.iou_threshold = attrs.iou_threshold;
        params.score_threshold = attrs.score_threshold;
        params.nms_top_k = attrs.nms_top_k;
        params.keep_top_k = attrs.keep_top_k;
        params.background_class = attrs.background_class;
        params.normalized = attrs.normalized;
        params.nms_eta = attrs.nms_eta;
        params.has_roisnum = primitive->has_roisnum;

        size_t inputs_num = primitive->has_roisnum ? 3 : 2;

        params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[1]));
        if (inputs_num == 3) {
            params.inputs.push_back(convert_data_tensor(impl_param.input_layouts[2]));
            params.has_roisnum = true;
        }

        if (primitive->num_outputs == 3) {
            params.outputs.push_back(convert_data_tensor(impl_param.output_layouts[1]));
            params.outputs.push_back(convert_data_tensor(impl_param.output_layouts[2]));
        } else {
            // Legacy multi-output
            params.outputs.push_back(convert_data_tensor(impl_param.input_layouts[inputs_num + 0]));
            params.outputs.push_back(convert_data_tensor(impl_param.input_layouts[inputs_num + 1]));
        }

        return params;
    }
};

namespace detail {
attach_multiclass_nms_impl::attach_multiclass_nms_impl() {
    auto types = {data_types::f16, data_types::f32, data_types::i32, data_types::i64};
    auto formats = {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv16_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
    };

    implementation_map<multiclass_nms>::add(impl_types::ocl, typed_primitive_impl_ocl<multiclass_nms>::create<multiclass_nms_impl>, types, formats);
}
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::multiclass_nms_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::multiclass_nms)
