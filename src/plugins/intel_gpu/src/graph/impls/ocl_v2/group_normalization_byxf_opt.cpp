// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_normalization_byxf_opt.hpp"

#include <algorithm>

#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "ocl_v2/utils/fused_ops_jitter.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

constexpr uint32_t scale_input_idx = 1;
constexpr uint32_t bias_input_idx = 2;

// Internal f32 buffers: per-(chunk, b, f) shifted sums and sums of squares, then per-(b, f) A followed by per-(b, f) C.
constexpr uint32_t sum_buffer_idx = 0;
constexpr uint32_t sq_sum_buffer_idx = 1;
constexpr uint32_t scale_shift_buffer_idx = 2;

// Up to 4 features per work-item: 8- and 16-feature vectors were 4-35 % slower on Xe3 (fewer work-items in flight).
size_t get_vec_size(size_t features) {
    for (size_t vec : {4, 2}) {
        if (features % vec == 0) {
            return vec;
        }
    }
    return 1;
}

size_t get_feature_num(const RuntimeParams& params) {
    return static_cast<size_t>(params.get_input_layout(0).get_partial_shape()[1].get_length());
}

struct DispatchParams {
    size_t lws0;         // work-items along F per work-group (stages 1 and 3)
    size_t gws0;         // work-items along F, >= F / vec
    size_t num_chunks;   // stage 1 interleaved pixel sets per (b, f), multiple of stats_lws1
    size_t stats_lws1;   // stage 1 work-group size along the chunk axis
    size_t group_lanes;  // stage 2 work-items per chunk along the features of a group
    size_t norm_chunks;  // stage 3 pixel chunks per (b, f), multiple of norm_lws1
    size_t norm_lws1;    // stage 3 work-group size along the chunk axis
};

// Kernels read the chunk counts back from the launch geometry, so this is the only place that defines them.
DispatchParams get_dispatch_params(const RuntimeParams& params, const ov::Shape& shape) {
    const auto& info = params.get_device_info();
    const size_t batch = shape[0];
    const size_t pixels = shape[2] * shape[3];
    const size_t fv = shape[1] / get_vec_size(shape[1]);
    const size_t group_size = shape[1] / static_cast<size_t>(params.typed_desc<group_normalization>()->num_groups);
    const size_t max_lws = std::min<size_t>(256, info.max_work_group_size);
    // 16 stats work-items per hardware thread: more did not add bandwidth on Xe3.
    const size_t target_work_items = 16 * info.execution_units_count * std::max<size_t>(info.num_threads_per_eu, 1);

    DispatchParams d{};
    d.lws0 = ceil_div(fv, ceil_div(fv, max_lws));
    d.gws0 = d.lws0 * ceil_div(fv, d.lws0);
    // >= 32 pixels per chunk keeps the partial sums <= 1/8 of the f16 input bytes.
    d.num_chunks = std::clamp<size_t>(ceil_div(target_work_items, batch * d.gws0), 1, std::min(max_lws, std::max<size_t>(1, pixels / 32)));
    // Stats work-groups also span chunks, so a small F still fills whole sub-groups.
    d.stats_lws1 = std::min(std::max<size_t>(1, max_lws / d.lws0), d.num_chunks);
    d.num_chunks -= d.num_chunks % d.stats_lws1;
    d.group_lanes = std::clamp<size_t>(max_lws / d.num_chunks, 1, group_size);
    const size_t norm_chunks = std::max<size_t>(1, ceil_div(pixels, 16));
    d.norm_lws1 = std::min(std::max<size_t>(1, max_lws / d.lws0), norm_chunks);
    d.norm_chunks = align_to(norm_chunks, d.norm_lws1);
    return d;
}

class GroupNormalizationByxfGeneratorBase : public KernelGenerator {
public:
    explicit GroupNormalizationByxfGeneratorBase(std::string_view suffix) : KernelGenerator("group_normalization_byxf_opt", suffix) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto features = get_feature_num(params);
        const auto vec = get_vec_size(features);
        jit.make("VEC", vec);
        jit.make("FV", features / vec);
        jit.make("GROUP_SIZE", features / static_cast<size_t>(params.typed_desc<group_normalization>()->num_groups));
        return jit;
    }
};

class GroupNormalizationByxfGeneratorCalcStats : public GroupNormalizationByxfGeneratorBase {
public:
    GroupNormalizationByxfGeneratorCalcStats() : GroupNormalizationByxfGeneratorBase("calc_stats") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GroupNormalizationByxfGeneratorBase::get_jit_constants(params);
        jit.make("GROUP_NORM_KERNEL_CALC_STATS", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, sum_buffer_idx});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, sq_sum_buffer_idx});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            const auto& shape = params.get_input_layout(0).get_shape();
            const auto d = get_dispatch_params(params, shape);
            auto& wgs = kd.params.workGroups;
            wgs.global = {d.gws0, d.num_chunks, shape[0]};
            wgs.local = {d.lws0, d.stats_lws1, 1};
        }};
    }
};

class GroupNormalizationByxfGeneratorFinalizeStats : public GroupNormalizationByxfGeneratorBase {
public:
    GroupNormalizationByxfGeneratorFinalizeStats() : GroupNormalizationByxfGeneratorBase("finalize_stats") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GroupNormalizationByxfGeneratorBase::get_jit_constants(params);
        jit.make("GROUP_NORM_KERNEL_FINALIZE_STATS", 1);
        jit.make("EPSILON", static_cast<float>(params.typed_desc<group_normalization>()->epsilon));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, scale_input_idx});
        args.push_back({ArgumentDescriptor::Types::INPUT, bias_input_idx});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, sum_buffer_idx});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, sq_sum_buffer_idx});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, scale_shift_buffer_idx});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            const auto& shape = params.get_input_layout(0).get_shape();
            const auto d = get_dispatch_params(params, shape);
            const auto num_groups = static_cast<size_t>(params.typed_desc<group_normalization>()->num_groups);
            auto& wgs = kd.params.workGroups;
            wgs.global = {d.group_lanes * num_groups, d.num_chunks, shape[0]};
            wgs.local = {d.group_lanes, d.num_chunks, 1};
        }};
    }
};

class GroupNormalizationByxfGeneratorNormalize : public GroupNormalizationByxfGeneratorBase {
public:
    GroupNormalizationByxfGeneratorNormalize() : GroupNormalizationByxfGeneratorBase("normalize") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GroupNormalizationByxfGeneratorBase::get_jit_constants(params);
        jit.make("GROUP_NORM_KERNEL_NORMALIZE", 1);
        if (params.has_fused_primitives()) {
            // Scalar fused ops: a vector config fails at JIT time for fused inputs in another layout.
            const FusedOpsConfiguration conf = {"", {"(b)", "(f)", "(y)", "(x)"}, "normalized", ov::element::f32};
            jit.add(make_fused_ops_jit_constants(params, {conf}));
        }
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, scale_shift_buffer_idx});
        add_fused_ops_arguments(args, params);
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            const auto& shape = params.get_input_layout(0).get_shape();
            const auto d = get_dispatch_params(params, shape);
            auto& wgs = kd.params.workGroups;
            wgs.global = {d.gws0, d.norm_chunks, shape[0]};
            wgs.local = {d.lws0, d.norm_lws1, 1};
        }};
    }
};

class GroupNormalizationByxfOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GroupNormalizationByxfOptImpl)
    Stage::Ptr calc_stats = make_stage<GroupNormalizationByxfGeneratorCalcStats>();
    Stage::Ptr finalize_stats = make_stage<GroupNormalizationByxfGeneratorFinalizeStats>();
    Stage::Ptr normalize = make_stage<GroupNormalizationByxfGeneratorNormalize>();

    GroupNormalizationByxfOptImpl() : PrimitiveImplOCL(GroupNormalizationByxfOpt::get_type_info_static()) {}
    GroupNormalizationByxfOptImpl(const program_node& node, const RuntimeParams& params) : GroupNormalizationByxfOptImpl() {
        add_stage(calc_stats, params);
        add_stage(finalize_stats, params);
        add_stage(normalize, params);
        // add_stage() drops a stage whose code generation throws; validate_impl must reject such configurations.
        OPENVINO_ASSERT(_order.size() == _stages.size(), "[GPU] Failed to generate byxf GroupNormalization kernels for ", node.id());
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GroupNormalizationByxfOptImpl>(this);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        // Called with static params at runtime; at build time only when the outputs could be allocated (static or upper-bounded shape).
        const auto& pshape = params.get_input_layout(0).get_partial_shape();
        OPENVINO_ASSERT(std::none_of(pshape.begin(),
                                     pshape.end(),
                                     [](const ov::Dimension& dim) {
                                         return dim.get_max_length() == -1;
                                     }),
                        "[GPU] byxf GroupNormalization internal buffers need an upper-bounded shape, got ",
                        pshape);
        const auto shape = pshape.get_max_shape();
        const auto d = get_dispatch_params(params, shape);
        const size_t batch_features = shape[0] * shape[1];
        const BufferDescriptor partial_sums{d.num_chunks * batch_features, ov::element::f32};
        return {partial_sums, partial_sums, BufferDescriptor{2 * batch_features, ov::element::f32}};
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GroupNormalizationByxfOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    OPENVINO_ASSERT(node.is_type<group_normalization>());
    return std::make_unique<GroupNormalizationByxfOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GroupNormalizationByxfOptImpl)
