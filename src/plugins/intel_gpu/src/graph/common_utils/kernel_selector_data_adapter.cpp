// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_selector_data_adapter.hpp"

#include <type_traits>

#include "openvino/core/except.hpp"

namespace cldnn {

kernel_selector::Datatype to_data_type(data_types data_type) {
    switch (data_type) {
    case data_types::u2:
        return kernel_selector::Datatype::UINT2;
    case data_types::i4:
        return kernel_selector::Datatype::INT4;
    case data_types::u4:
        return kernel_selector::Datatype::UINT4;
    case data_types::i8:
        return kernel_selector::Datatype::INT8;
    case data_types::u8:
        return kernel_selector::Datatype::UINT8;
    case data_types::i16:
        return kernel_selector::Datatype::INT16;
    case data_types::u16:
        return kernel_selector::Datatype::UINT16;
    case data_types::i32:
        return kernel_selector::Datatype::INT32;
    case data_types::u32:
        return kernel_selector::Datatype::UINT32;
    case data_types::i64:
        return kernel_selector::Datatype::INT64;
    case data_types::f16:
        return kernel_selector::Datatype::F16;
    case data_types::f32:
        return kernel_selector::Datatype::F32;
    case data_types::bf16:
        return kernel_selector::Datatype::BF16;
    case data_types::f4e2m1:
        return kernel_selector::Datatype::F4E2M1;
    case data_types::f8e4m3:
        return kernel_selector::Datatype::F8E4M3;
    case data_types::f8e5m2:
        return kernel_selector::Datatype::F8E5M2;
    case data_types::f8e8m0:
        return kernel_selector::Datatype::F8E8M0;
    default:
        OPENVINO_THROW("[GPU] Unable to convert cldnn data type ", data_type, " to kernel_selector data type");
    }
}

kernel_selector::EngineInfo make_kernel_selector_engine_info(const device_info& info) {
    kernel_selector::EngineInfo result;
    result.supports_fp16 = info.supports_fp16;
    result.supports_fp64 = info.supports_fp64;
    result.supports_fp16_denorms = info.supports_fp16_denorms;
    result.supports_khr_subgroups = info.supports_khr_subgroups;
    result.supports_intel_subgroups = info.supports_intel_subgroups;
    result.supports_intel_subgroups_short = info.supports_intel_subgroups_short;
    result.supports_intel_subgroups_char = info.supports_intel_subgroups_char;
    result.supports_intel_required_subgroup_size = info.supports_intel_required_subgroup_size;
    result.supports_image = info.supports_image;
    result.supports_work_group_collective_functions = info.supports_work_group_collective_functions;
    result.supports_non_uniform_work_group = info.supports_non_uniform_work_group;
    result.supports_imad = info.supports_imad;
    result.supports_immad = info.supports_immad;
    result.enable_sub_groups_emulation = true;
    result.deviceType = info.dev_type == device_type::discrete_gpu ? kernel_selector::dev_type::discrete_gpu : kernel_selector::dev_type::integrated_gpu;
    result.maxWorkGroupSize = info.max_work_group_size;
    result.maxLocalMemSize = info.max_local_mem_size;
    result.maxImage2dWidth = info.max_image2d_width;
    result.maxImage2dHeight = info.max_image2d_height;
    result.computeUnitsCount = info.execution_units_count;
    result.maxThreadsPerExecutionUnit = info.num_threads_per_eu > 0 ? info.num_threads_per_eu : 7;
    result.maxThreadsPerDevice = result.maxThreadsPerExecutionUnit * info.execution_units_count;
    result.driverVersion = info.driver_version;
    result.supportedSimdSizes = info.supported_simd_sizes;
    result.vendor_id = info.vendor_id;
    result.ip_version = info.ip_version;
    result.arch = kernel_selector::gpu_arch(static_cast<std::underlying_type<gpu_arch>::type>(info.arch));
    return result;
}

kernel_selector::DataLayout to_data_layout(format data_format) {
    switch (data_format) {
    case format::bfyx:
        return kernel_selector::DataLayout::bfyx;
    case format::yxfb:
        return kernel_selector::DataLayout::yxfb;
    case format::byxf:
        return kernel_selector::DataLayout::byxf;
    case format::byfx:
        return kernel_selector::DataLayout::byfx;
    case format::bxfy:
        return kernel_selector::DataLayout::bxfy;
    case format::fbyx:
        return kernel_selector::DataLayout::fbyx;
    case format::fyxb:
        return kernel_selector::DataLayout::fyxb;
    case format::b_fs_yx_fsv2:
        return kernel_selector::DataLayout::b_fs_yx_fsv2;
    case format::b_fs_yx_fsv4:
        return kernel_selector::DataLayout::b_fs_yx_fsv4;
    case format::b_fs_yx_fsv8:
        return kernel_selector::DataLayout::b_fs_yx_fsv8;
    case format::b_fs_yx_fsv16:
        return kernel_selector::DataLayout::b_fs_yx_fsv16;
    case format::b_fs_yx_fsv32:
        return kernel_selector::DataLayout::b_fs_yx_fsv32;
    case format::b_fs_zyx_fsv2:
        return kernel_selector::DataLayout::b_fs_zyx_fsv2;
    case format::b_fs_zyx_fsv4:
        return kernel_selector::DataLayout::b_fs_zyx_fsv4;
    case format::b_fs_zyx_fsv8:
        return kernel_selector::DataLayout::b_fs_zyx_fsv8;
    case format::b_fs_zyx_fsv32:
        return kernel_selector::DataLayout::b_fs_zyx_fsv32;
    case format::bs_f_bsv16:
        return kernel_selector::DataLayout::bs_f_bsv16__af8;
    case format::bs_fs_fsv8_bsv8:
        return kernel_selector::DataLayout::bs_f_bsv8__af8;
    case format::winograd_2x3_s1_data:
        return kernel_selector::DataLayout::winograd_2x3_s1_data;
    case format::bfzyx:
        return kernel_selector::DataLayout::bfzyx;
    case format::bzyxf:
        return kernel_selector::DataLayout::bzyxf;
    case format::ybfx:
        return kernel_selector::DataLayout::ybfx;
    case format::fs_b_yx_fsv32:
        return kernel_selector::DataLayout::fs_b_yx_fsv32;
    case format::bfwzyx:
        return kernel_selector::DataLayout::bfwzyx;
    case format::bfuwzyx:
        return kernel_selector::DataLayout::bfuwzyx;
    case format::bfvuwzyx:
        return kernel_selector::DataLayout::bfvuwzyx;
    case format::b_fs_zyx_fsv16:
        return kernel_selector::DataLayout::b_fs_zyx_fsv16;
    case format::bs_fs_yx_bsv16_fsv32:
        return kernel_selector::DataLayout::bs_fs_yx_bsv16_fsv32;
    case format::bs_fs_zyx_bsv16_fsv32:
        return kernel_selector::DataLayout::bs_fs_zyx_bsv16_fsv32;
    case format::bs_fs_zyx_bsv16_fsv16:
        return kernel_selector::DataLayout::bs_fs_zyx_bsv16_fsv16;
    case format::bs_fs_yx_bsv16_fsv16:
        return kernel_selector::DataLayout::bs_fs_yx_bsv16_fsv16;
    case format::bs_fs_zyx_bsv32_fsv16:
        return kernel_selector::DataLayout::bs_fs_zyx_bsv32_fsv16;
    case format::bs_fs_yx_bsv32_fsv16:
        return kernel_selector::DataLayout::bs_fs_yx_bsv32_fsv16;
    case format::bs_fs_yx_bsv4_fsv4:
        return kernel_selector::DataLayout::bs_fs_yx_bsv4_fsv4;
    case format::bs_fs_yx_bsv8_fsv4:
        return kernel_selector::DataLayout::bs_fs_yx_bsv8_fsv4;
    case format::bs_fs_zyx_bsv8_fsv4:
        return kernel_selector::DataLayout::bs_fs_zyx_bsv8_fsv4;
    case format::bs_fs_yx_bsv16_fsv4:
        return kernel_selector::DataLayout::bs_fs_yx_bsv16_fsv4;
    case format::bs_fs_zyx_bsv16_fsv4:
        return kernel_selector::DataLayout::bs_fs_zyx_bsv16_fsv4;
    case format::bs_fs_yx_bsv16_fsv2:
        return kernel_selector::DataLayout::bs_fs_yx_bsv16_fsv2;
    case format::bs_fs_zyx_bsv16_fsv2:
        return kernel_selector::DataLayout::bs_fs_zyx_bsv16_fsv2;
    case format::bs_fs_yx_bsv16_fsv8:
        return kernel_selector::DataLayout::bs_fs_yx_bsv16_fsv8;
    case format::bs_fs_zyx_bsv16_fsv8:
        return kernel_selector::DataLayout::bs_fs_zyx_bsv16_fsv8;
    case format::bs_fs_yx_bsv8_fsv2:
        return kernel_selector::DataLayout::bs_fs_yx_bsv8_fsv2;
    case format::bs_fs_zyx_bsv8_fsv2:
        return kernel_selector::DataLayout::bs_fs_zyx_bsv8_fsv2;
    case format::bs_fs_yx_bsv4_fsv2:
        return kernel_selector::DataLayout::bs_fs_yx_bsv4_fsv2;
    case format::bs_fs_yx_bsv32_fsv32:
        return kernel_selector::DataLayout::bs_fs_yx_bsv32_fsv32;
    case format::bs_fs_zyx_bsv32_fsv32:
        return kernel_selector::DataLayout::bs_fs_zyx_bsv32_fsv32;
    case format::nv12:
        return kernel_selector::DataLayout::nv12;
    case format::image_2d_rgba:
        return kernel_selector::DataLayout::image_2d_rgba;
    default:
        OPENVINO_THROW("[GPU] Can't convert tensor format to kernel selector format as f=", data_format, " is not handled");
    }
}

kernel_selector::Tensor::NDims compute_tensor_dimensions(const layout& tensor_layout, size_t channel_count, tensor view_offset) {
    const auto& padding = tensor_layout.data_padding;
    const auto& dynamic_padding = layout::format_sizes(padding._dynamic_dims_mask, tensor_layout.format);
    const auto& original_shape = tensor_layout.get_partial_shape();
    const auto& view_offsets = view_offset.sizes(tensor_layout.format);

    ov::PartialShape ordered_shape;
    const auto& axis_order = tensor_layout.format.dims_order();
    for (size_t index = 0; index < axis_order.size(); ++index) {
        ordered_shape.push_back(axis_order[index] < original_shape.size() ? original_shape[axis_order[index]] : ov::Dimension(1));
    }

    const auto& lower_padding = layout::format_sizes(padding._lower_size, tensor_layout.format);
    const auto& upper_padding = layout::format_sizes(padding._upper_size, tensor_layout.format);
    kernel_selector::Tensor::NDims dimensions(channel_count);
    size_t pitch = 1;
    for (size_t index = 0; index < dimensions.size(); ++index) {
        const size_t tensor_index = dimensions.size() - 1 - index;
        const auto dimension = tensor_index < ordered_shape.size() ? ordered_shape[tensor_index] : ov::Dimension(1);
        const auto lower = lower_padding[tensor_index] + view_offsets[tensor_index];
        const auto upper = upper_padding[tensor_index];
        const auto reserved_elements = dimension.is_dynamic() ? 0 : dimension.get_length() - view_offsets[tensor_index];

        auto& result = dimensions[index];
        result.v = dimension.is_dynamic() ? 0 : static_cast<size_t>(dimension.get_length() - view_offsets[tensor_index]);
        result.pitch = pitch;
        result.pad.before = dynamic_padding[tensor_index] ? 0 : lower;
        result.pad.after = dynamic_padding[tensor_index] ? 0 : upper;
        result.pad.is_dynamic = dynamic_padding[tensor_index] != 0;
        result.is_dynamic = dimension.is_dynamic();
        pitch *= reserved_elements + lower + upper;
    }

    return dimensions;
}

kernel_selector::DataTensor convert_data_tensor(const layout& tensor_layout, tensor view_offset) {
    const auto data_layout = to_data_layout(tensor_layout.format);
    auto dimensions = compute_tensor_dimensions(tensor_layout, kernel_selector::DataTensor::ChannelsCount(data_layout), view_offset);
    return {dimensions, to_data_type(tensor_layout.data_type), data_layout};
}

}  // namespace cldnn
