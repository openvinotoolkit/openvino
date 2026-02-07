// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "group_normalization_fsv16_fused.hpp"

#include "common_utils/jitter.hpp"
#include "common_utils/kernel_generator_base.hpp"
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

constexpr size_t fsv = 16;

ov::element::Type get_activation_type(const RuntimeParams& params) {
    if (params.get_input_layout(0).data_type == ov::element::f16) {
        return ov::element::f16;
    }
    return ov::element::f32;
}

ov::element::Type get_accumulator_type(const RuntimeParams& params) {
    auto in_dt = params.get_input_layout(0).data_type;
    switch (in_dt) {
    case ov::element::u8:
    case ov::element::i8:
        return ov::element::i32;
    default:
        return ov::element::f32;
    }
}

class GroupNormalizationGeneratorBase : public KernelGenerator {
public:
    explicit GroupNormalizationGeneratorBase(std::string_view name, std::string_view suffix) : KernelGenerator(name, suffix) {}
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto get_max_simd_size = [](const RuntimeParams& params) {
            size_t max_simd_size = fsv;
            for (auto& simd_size : params.get_device_info().supported_simd_sizes) {
                max_simd_size = std::max(max_simd_size, simd_size);
            }
            return max_simd_size;
        };

        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<group_normalization>();
        jit.make("EPSILON", static_cast<float>(desc->epsilon));
        jit.make("NUM_GROUPS", desc->num_groups);
        jit.make("SIMD", get_max_simd_size(params));
        jit.make("FSV", fsv);

        if (params.is_dynamic()) {
            jit.make("GWS0", "get_global_size(0)");
            jit.make("LWS0", "get_local_size(0)");
            jit.make("SLM_SIZE", params.get_device_info().max_work_group_size);
        } else {
            KernelData kd;
            get_dispatch_data_func()(params, kd);
            const auto& wgs = kd.params.workGroups;

            jit.make("GWS0", wgs.global[0]);
            jit.make("LWS0", wgs.local[0]);
            jit.make("SLM_SIZE", wgs.local[0]);
        }

        auto activation_type = get_activation_type(params);
        jit.add(make_type_jit_constants("ACTIVATION", activation_type));
        jit.add(make_type_jit_constants("ACCUMULATOR", get_accumulator_type(params)));

        return jit;
    }
};

class GroupNormalizationGeneratorCalcSQRMean : public GroupNormalizationGeneratorBase {
public:
    GroupNormalizationGeneratorCalcSQRMean() : GroupNormalizationGeneratorBase("group_normalization_fsv16_fused", "calc_sqr_mean") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = GroupNormalizationGeneratorBase::get_jit_constants(params);
        jit.make("GROUP_NORM_KERNEL_FEATURE_MEAN_SQR_MEAN", 1);

        if (params.has_fused_primitives()) {
            const auto& out_l = params.get_output_layout(0);
            FusedOpsConfiguration conf = {"", std::vector<std::string>{"(b)", "(f)", "(y)", "(x)"}, "normalized", out_l.data_type};
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
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});
        add_fused_ops_arguments(args, params);
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;

            auto padded_dims = params.input_layouts[0].get_padded_dims();
            auto x = padded_dims[3];
            auto y = padded_dims[2];
            auto f = padded_dims[1];
            auto b = padded_dims[0];

            wgs.global[0] = x * y;
            wgs.global[1] = ceil_div(f, fsv) * b;
            wgs.global[2] = 1;

            wgs.local[0] = x * y;
            wgs.local[1] = 1;
            wgs.local[2] = 1;

            auto max_wgs = params.get_device_info().max_work_group_size;

            size_t divisor = 2;
            while (wgs.local[0] > (max_wgs / fsv)) {
                if (wgs.global[0] % divisor == 0) {
                    wgs.local[0] = wgs.global[0] / divisor;
                }
                divisor += 1;
            }
            wgs.local[0] *= fsv;
            wgs.global[0] = wgs.local[0];
        }};
    }
};

class GroupNormalizationFsv16FusedImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::GroupNormalizationFsv16FusedImpl)

    Stage::Ptr calc_sqr_mean = make_stage<GroupNormalizationGeneratorCalcSQRMean>();

    GroupNormalizationFsv16FusedImpl() : PrimitiveImplOCL(GroupNormalizationFsv16Fused::get_type_info_static()) {}
    GroupNormalizationFsv16FusedImpl(const program_node& node, const RuntimeParams& params) : GroupNormalizationFsv16FusedImpl() {
        add_stage(calc_sqr_mean, params);
    }

    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GroupNormalizationFsv16FusedImpl>(this);
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GroupNormalizationFsv16Fused::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<group_normalization>());
    return std::make_unique<GroupNormalizationFsv16FusedImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::GroupNormalizationFsv16FusedImpl)
