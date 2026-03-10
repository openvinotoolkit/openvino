// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "segment_max.hpp"

#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/segment_max.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "segment_max_inst.h"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {

namespace {

enum KernelsTypes {
    REF = 0,
    OPT,
};

class SegmentMaxRef : public KernelGenerator {
public:
    SegmentMaxRef() : KernelGenerator("segment_max_ref") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto& desc = params.typed_desc<segment_max>();

        jit.add(make_jit_constant("FILL_MODE", static_cast<int>(desc->fill_mode)));

        if (desc->fill_mode == ov::op::FillMode::ZERO) {
            jit.add(make_jit_constant("EMPTY_SEGMENT_VALUE", "0"));
        } else {
            jit.add(make_jit_constant("EMPTY_SEGMENT_VALUE", "OUTPUT_VAL_MIN"));
        }

        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            const auto& out_l = params.get_output_layout(0);

            size_t output_size = out_l.is_dynamic() ? 1 : out_l.count();
            wgs.global = {output_size, 1, 1};
            wgs.local = ov::intel_gpu::get_optimal_lws(wgs.global, params.get_device_info());
        }};
    }
};

class SegmentMaxOpt : public KernelGenerator {
public:
    SegmentMaxOpt() : KernelGenerator("segment_max_opt") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto& desc = params.typed_desc<segment_max>();

        jit.add(make_jit_constant("FILL_MODE", static_cast<int>(desc->fill_mode)));

        if (desc->fill_mode == ov::op::FillMode::ZERO) {
            jit.add(make_jit_constant("EMPTY_SEGMENT_VALUE", "0"));
        } else {
            jit.add(make_jit_constant("EMPTY_SEGMENT_VALUE", "OUTPUT_VAL_MIN"));
        }

        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            const auto& out_l = params.get_output_layout(0);

            if (out_l.is_dynamic()) {
                wgs.global = {1, 1, 1};
            } else {
                size_t inner_dim = static_cast<size_t>(out_l.feature()) * out_l.spatial(2) * out_l.spatial(1) * out_l.spatial(0);
                size_t num_segments = static_cast<size_t>(out_l.batch());
                wgs.global = {inner_dim, num_segments, 1};
            }
            wgs.local = ov::intel_gpu::get_optimal_lws(wgs.global, params.get_device_info());
        }};
    }
};

class SegmentMaxImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::SegmentMaxImpl)

    Stage::Ptr ref_stage = make_stage<SegmentMaxRef>();
    Stage::Ptr opt_stage = make_stage<SegmentMaxOpt>();

    SegmentMaxImpl() : PrimitiveImplOCL(SegmentMax::get_type_info_static()) {}
    SegmentMaxImpl(const program_node& node, const RuntimeParams& params) : SegmentMaxImpl() {
        add_stage(ref_stage, params);
        add_stage(opt_stage, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<SegmentMaxImpl>(this);
    }

    std::vector<size_t> get_stages_execution_order(const cldnn::primitive_inst& instance) const override {
        // Always prefer the optimized kernel (binary search) since segment_ids
        // is required to be sorted (non-decreasing) by the SegmentMax-16 spec.
        return {KernelsTypes::OPT};
    }
};

}  // namespace

std::unique_ptr<primitive_impl> SegmentMax::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<segment_max>());
    return std::make_unique<SegmentMaxImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::segment_max)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::SegmentMaxImpl)
