// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "histc.hpp"

#include <algorithm>

#include "common_utils/jitter.hpp"
#include "histc_inst.h"
#include "intel_gpu/primitives/histc.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

constexpr size_t histc_lws = 256;
constexpr size_t simd = 16;
constexpr size_t merge_lws = 64;
constexpr size_t min_items_per_work_item = 8;
// Interleaved copies of the local histogram reduce SLM atomic conflicts, which matter only for few bins.
constexpr size_t max_hist_copies = 16;
constexpr size_t hist_copies_slm_budget = 8 * 1024;
// Upper bound for the per-work-group partial histograms buffer (in elements).
constexpr size_t max_partial_hist_size = size_t{1} << 22;

bool range_from_data(const RuntimeParams& params) {
    const auto desc = params.typed_desc<histc>();
    return desc->min == desc->max;
}

size_t get_bins(const RuntimeParams& params) {
    return static_cast<size_t>(params.typed_desc<histc>()->bins);
}

size_t get_lws(const RuntimeParams& params) {
    return std::min<size_t>(histc_lws, params.get_device_info().max_work_group_size);
}

// Every work-group builds its own partial histogram, which are then summed by the merge stage.
size_t get_num_groups(const RuntimeParams& params) {
    const auto& info = params.get_device_info();
    size_t groups = std::clamp<size_t>(max_partial_hist_size / get_bins(params), 1, std::max<size_t>(info.execution_units_count, 1));
    const auto& in_layout = params.get_input_layout(0);
    if (in_layout.is_static()) {
        groups = std::clamp<size_t>(ceil_div(in_layout.count(), get_lws(params) * min_items_per_work_item), 1, groups);
    }
    return groups;
}

class HistcGeneratorBase : public KernelGenerator {
public:
    explicit HistcGeneratorBase(std::string_view suffix) : KernelGenerator("histc", suffix) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<histc>();
        const auto bins = get_bins(params);
        jit.make("BINS", bins);
        jit.make("RANGE_FROM_DATA", range_from_data(params));
        jit.make("SIMD", simd);
        // 16-byte block reads per work-item.
        jit.make("VEC_SIZE", 16 / ov::element::Type(params.get_input_layout(0).data_type).size());
        // Leave half of SLM free so that several work-groups can share a sub-slice.
        const bool hist_in_slm = bins * sizeof(uint32_t) <= params.get_device_info().max_local_mem_size / 2;
        size_t copies = 1;
        while (hist_in_slm && copies < max_hist_copies && bins * copies * 2 * sizeof(uint32_t) <= hist_copies_slm_budget) {
            copies *= 2;
        }
        jit.make("HIST_IN_SLM", hist_in_slm);
        jit.make("HIST_COPIES", copies);
        if (!range_from_data(params)) {
            jit.make("RANGE_MIN", static_cast<float>(desc->min));
            jit.make("RANGE_MAX", static_cast<float>(desc->max));
        }
        return jit;
    }

    // Bin edges must match the reference implementation, which relies on IEEE division.
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + " -cl-fp32-correctly-rounded-divide-sqrt";
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            const auto lws = get_lws(params);
            wgs.global = {get_num_groups(params) * lws, 1, 1};
            wgs.local = {lws, 1, 1};
        }};
    }
};

class HistcMinMaxGenerator : public HistcGeneratorBase {
public:
    HistcMinMaxGenerator() : HistcGeneratorBase("minmax") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = HistcGeneratorBase::get_jit_constants(params);
        jit.make("HISTC_STAGE_MINMAX", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        return args;
    }
};

class HistcCountGenerator : public HistcGeneratorBase {
public:
    HistcCountGenerator() : HistcGeneratorBase("count") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = HistcGeneratorBase::get_jit_constants(params);
        jit.make("HISTC_STAGE_COUNT", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        if (range_from_data(params)) {
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        }
        return args;
    }
};

class HistcMergeGenerator : public HistcGeneratorBase {
public:
    HistcMergeGenerator() : HistcGeneratorBase("merge") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = HistcGeneratorBase::get_jit_constants(params);
        jit.make("HISTC_STAGE_MERGE", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            wgs.global = {align_to(get_bins(params), merge_lws), 1, 1};
            wgs.local = {merge_lws, 1, 1};

            scalar_desc num_groups;
            num_groups.t = scalar_desc::Types::UINT32;
            num_groups.v.u32 = static_cast<uint32_t>(get_num_groups(params));
            kd.params.scalars = {num_groups};
        }};
    }
};

class HistcImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::HistcImpl)

    Stage::Ptr minmax = make_stage<HistcMinMaxGenerator>();
    Stage::Ptr count = make_stage<HistcCountGenerator>();
    Stage::Ptr merge = make_stage<HistcMergeGenerator>();

    HistcImpl() : PrimitiveImplOCL(Histc::get_type_info_static()) {}
    HistcImpl(const program_node& node, const RuntimeParams& params) : HistcImpl() {
        if (range_from_data(params)) {
            add_stage(minmax, params);
        }
        add_stage(count, params);
        add_stage(merge, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<HistcImpl>(this);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        const auto groups = get_num_groups(params);
        std::vector<BufferDescriptor> buffers = {BufferDescriptor{groups * get_bins(params), ov::element::u32}};
        if (range_from_data(params)) {
            buffers.emplace_back(groups * 2, ov::element::f32);
        }
        return buffers;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> Histc::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<histc>());
    return std::make_unique<HistcImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::histc)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::HistcImpl)
