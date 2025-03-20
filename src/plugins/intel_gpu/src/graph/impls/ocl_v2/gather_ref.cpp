// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_ref.hpp"

#include <cstddef>

#include "common_utils/dispatch_utils.hpp"
#include "common_utils/jitter.hpp"
#include "gather_inst.h"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "openvino/core/validation_util.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/fused_ops_jitter.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

using cldnn::layout;  // explicit using as symbol conflicts with namespace in core

size_t get_gather_axis_shape_info_idx(size_t axis, size_t rank) {
    auto shape_info_order = get_default_channels_order(layout::max_rank());
    auto order = get_default_channels_order(rank);
    return std::distance(shape_info_order.begin(), std::find(shape_info_order.begin(), shape_info_order.end(), order[axis]));
}

int64_t get_non_empty_dims_num(const layout& data_tensor) {
    if (data_tensor.is_dynamic() || data_tensor.count() != 1) {
        // Count the number of "one size" dimensions starting with X to Batch
        size_t one_size_dims = 0;
        const auto& shape_raw = data_tensor.get_partial_shape();
        ov::Shape shape;
        for (size_t i = 0; i < layout::max_rank(); i++) {
            int shape_raw_idx = get_channel_index(ChannelName{static_cast<uint8_t>(i)}, data_tensor.get_rank());
            if (shape_raw_idx >= 0 && shape_raw_idx < static_cast<int32_t>(shape_raw.size())) {
                shape.push_back(static_cast<size_t>(shape_raw[shape_raw_idx].is_dynamic() ? 0 : shape_raw[shape_raw_idx].get_length()));
            } else if (shape_raw_idx >= 0) {
                shape.push_back(1);
            }
        }
        for (auto& i : shape) {
            if (i == 1) {
                one_size_dims++;
            } else {
                break;
            }
        }
        return static_cast<int64_t>(data_tensor.get_rank() - one_size_dims);
    }
    return 1;
}

std::string get_order_string(const std::vector<std::string>& order) {
    std::string order_str = order[0];
    for (size_t i = 1; i < order.size(); i++) {
        order_str += ", " + order[i];
    }

    return order_str;
}

inline std::vector<std::string> get_idx_order(size_t rank) {
    std::vector<std::string> idx_order;
    if (rank <= 4) {
        idx_order = {"b", "f", "y", "x"};
    } else if (rank == 5) {
        idx_order = {"b", "f", "z", "y", "x"};
    } else if (rank == 6) {
        idx_order = {"b", "f", "w", "z", "y", "x"};
    }
    return idx_order;
}

inline std::vector<std::string> get_final_idx_order(size_t rank) {
    std::vector<std::string> idx_order;

    OPENVINO_ASSERT(rank > 4, "[GPU] Only support 5 or 6 dimensions");

    if (rank == 5) {
        idx_order = {"b", "f", "0", "z", "0"};
    } else if (rank == 6) {
        idx_order = {"b", "f", "0", "w", "z", "0"};
    }
    return idx_order;
}

std::string get_indices_idx_order(const layout& idx_l, const layout& out_l, size_t axis, int64_t batch_dim) {
    std::vector<std::string> idx_order;

    if ((axis == static_cast<size_t>(batch_dim)) && (axis > 1) && (idx_l.get_rank() > 4)) {
        idx_order = get_final_idx_order(out_l.get_rank());
    } else {
        idx_order = get_idx_order(out_l.get_rank());
        const auto* zero_val = "0";

        size_t indices_dims_num = get_non_empty_dims_num(idx_l);

        // Shift indices of Gather indices input related to output dims
        for (size_t i = batch_dim; i < indices_dims_num; i++) {
            idx_order[i] = idx_order[axis + i - batch_dim];
        }

        for (size_t i = indices_dims_num; i < idx_order.size(); i++) {
            idx_order[i] = zero_val;
        }

        // Fix size to inputs[1] dims size
        for (size_t i = 0; i < out_l.get_rank() - idx_l.get_rank(); i++) {
            idx_order.pop_back();
        }
    }

    return get_order_string(idx_order);
}

std::string get_dictionary_idx_order(const layout& data_l, const layout& out_l, size_t axis) {
    auto idx_order = get_idx_order(out_l.get_rank());
    const auto* input_axis_index_macro = "INPUT_AXIS_INDEX";
    const auto* zero_val = "0";

    size_t dictionary_dims_num = get_non_empty_dims_num(data_l);
    size_t indices_dims_num = get_non_empty_dims_num(out_l) - dictionary_dims_num + 1;

    // Shift indices of Gather dictionary input related to output dims
    for (size_t i = axis + 1; i < dictionary_dims_num; i++) {
        idx_order[i] = idx_order[i + indices_dims_num - 1];
    }

    for (size_t i = dictionary_dims_num; i < idx_order.size(); i++) {
        idx_order[i] = zero_val;
    }

    // Fix size to inputs[0] dims size
    if (out_l.get_rank() > data_l.get_rank()) {
        for (size_t i = 0; i < out_l.get_rank() - data_l.get_rank(); i++) {
            idx_order.pop_back();
        }
    }
    idx_order[axis] = input_axis_index_macro;

    return get_order_string(idx_order);
}

class GatherGenerator : public KernelGenerator {
public:
    GatherGenerator() : KernelGenerator("gather_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto& desc = params.typed_desc<gather>();
        const auto& in_offsets_map = params.in_port_to_shape_info_offset;
        const auto& data_l = params.input_layouts[0];
        const auto& idx_l = params.input_layouts[1];
        const auto& out_l = params.output_layouts[0];
        auto axis = ov::util::normalize_axis(desc->axis, static_cast<int64_t>(data_l.get_rank()));
        int64_t batch_dim = (desc->batch_dim != 0) ? get_non_empty_dims_num(idx_l) + desc->batch_dim : desc->batch_dim;

        jit.add({
            make_jit_constant("DICTIONARY_INDEX_ORDER", get_dictionary_idx_order(data_l, out_l, axis)),
            make_jit_constant("INDICES_INDEX_ORDER", get_indices_idx_order(idx_l, out_l, axis, batch_dim)),
        });

        if (params.has_fused_primitives()) {
            std::vector<std::string> idx_order = get_idx_order(data_l.get_rank());

            FusedOpsConfiguration conf = {"", idx_order, "val", data_l.data_type};
            jit.add(make_fused_ops_jit_constants(params, {conf}));
        }

        const auto& gather_dim = data_l.get_partial_shape()[static_cast<std::ptrdiff_t>(axis)];
        JitTerm shape_info{"shape_info"};
        if (gather_dim.is_static()) {
            jit.make("AXIS_DIM", gather_dim.get_length());
        } else {
            jit.make("AXIS_DIM", shape_info[get_gather_axis_shape_info_idx(axis, data_l.get_partial_shape().size())].str());
        }

        if (desc->support_neg_ind) {
            jit.make("WITH_NEGATIVE_IDX", true);
        }

        if (desc->decompression_scale.is_valid()) {
            const auto scale_l = params.input_layouts[2];
            const size_t scale_groups_num = scale_l.count();
            const size_t scale_group_size = data_l.count() / scale_groups_num;

            jit.make("COMPRESSED_WEIGHTS", true);
            jit.make("DECOMPRESSION_SCALE_TERM", true);
            jit.add(make_layout_jit_constants("DECOMPRESSION_SCALE", scale_l, in_offsets_map.at(2)));
            jit.make("DECOMPRESSION_SCALE_GROUPS_NUM", scale_groups_num);
            jit.make("DECOMPRESSION_SCALE_GROUP_SIZE", scale_group_size);
            if (one_of(data_l.data_type, {ov::element::i8, ov::element::u8})) {
                jit.make("COMPRESSED_WEIGHTS_INT8", true);
            } else if (one_of(data_l.data_type, {ov::element::i4, ov::element::u4})) {
                jit.make("COMPRESSED_WEIGHTS_INT4", true);
                jit.add(make_int4_packed_type_jit_constant("INT4_PACKED_TYPE", data_l.data_type, 2));
            }

            if (desc->decompression_zero_point.is_valid()) {
                const auto zp_l = params.input_layouts[3];
                const size_t zp_groups_num = zp_l.count();
                const size_t zp_group_size = data_l.count() / zp_groups_num;
                jit.make("DECOMPRESSION_ZP_TERM", true);
                jit.add(make_layout_jit_constants("DECOMPRESSION_ZP", zp_l, in_offsets_map.at(3)));
                jit.make("DECOMPRESSION_ZP_GROUPS_NUM", zp_groups_num);
                jit.make("DECOMPRESSION_ZP_GROUP_SIZE", zp_group_size);
            } else if (desc->decompression_zero_point_scalar.has_value()) {
                jit.make("DECOMPRESSION_ZP_TERM", true);
                jit.make("DECOMPRESSION_ZP_SCALAR", true);
                jit.make("DECOMPRESSION_ZP_VALUE", desc->decompression_zero_point_scalar.value());
            }
        }

        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        const auto& desc = params.typed_desc<gather>();

        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }

        args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // data
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // indices
        // Note that axis input is skipped on gather primitive creation, so decompression parameters has smaller idx
        if (desc->decompression_scale.is_valid()) {
            args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // scales
            if (desc->decompression_zero_point.is_valid()) {
                args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // zp
            }
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        add_fused_ops_arguments(args, params);

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            const auto& data_l = params.input_layouts[0];
            const auto& out_l = params.output_layouts[0];

            std::vector<std::vector<ChannelName>> dims_by_gws;

            size_t rank = out_l.get_rank();
            size_t b = extract_channel(ChannelName::BATCH, out_l);
            size_t f = extract_channel(ChannelName::FEATURE, out_l);
            size_t w = extract_channel(ChannelName::W, out_l);
            size_t z = extract_channel(ChannelName::Z, out_l);
            size_t y = extract_channel(ChannelName::Y, out_l);
            size_t x = extract_channel(ChannelName::X, out_l);

            if (rank == 4) {
                wgs.global = {x, y, f * b};
                dims_by_gws = {{ChannelName::X}, {ChannelName::Y}, {ChannelName::FEATURE, ChannelName::BATCH}};
            } else if (rank == 5) {
                wgs.global = {x, y * z, f * b};
                dims_by_gws = {{ChannelName::X}, {ChannelName::Y, ChannelName::Z}, {ChannelName::FEATURE, ChannelName::BATCH}};
            } else if (rank == 6) {
                wgs.global = {x * y, z * w, f * b};
                dims_by_gws = {{ChannelName::X, ChannelName::Y}, {ChannelName::Z, ChannelName::W}, {ChannelName::FEATURE, ChannelName::BATCH}};
            } else {
                OPENVINO_THROW("[GPU] Unexpected rank: ", rank);
            }

            wgs.local = ov::intel_gpu::get_optimal_lws(wgs.global, params.get_device_info(), data_l.format, out_l.format, dims_by_gws);
        }};
    }
};

class GatherRefImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GatherRefImpl)

    Stage::Ptr gather_kernel = make_stage<GatherGenerator>();

    GatherRefImpl() : PrimitiveImplOCL(GatherRef::get_type_info_static()) {}
    GatherRefImpl(const program_node& node, const RuntimeParams& params) : GatherRefImpl() {
        add_stage(gather_kernel, params);
    }
    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GatherRefImpl>(this);
    }

    [[nodiscard]] kernel_arguments_data get_arguments(const primitive_inst& instance) const override {
        kernel_arguments_data args = PrimitiveImplOCL::get_arguments(instance);
        const auto& desc = instance.get_typed_desc<gather>();

        if (desc->decompression_scale.is_valid()) {
            args.inputs.push_back(instance.dep_memory_ptr(2));
        }

        if (desc->decompression_zero_point.is_valid()) {
            args.inputs.push_back(instance.dep_memory_ptr(3));
        }

        return args;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GatherRef::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<gather>());
    return std::make_unique<GatherRefImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::gather)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GatherRefImpl)
