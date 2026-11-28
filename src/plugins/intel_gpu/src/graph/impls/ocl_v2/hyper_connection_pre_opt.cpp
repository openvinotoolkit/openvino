// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hyper_connection_pre_opt.hpp"

#include "intel_gpu/primitives/hyper_connection_pre.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {

constexpr size_t hc_pre_lws = 256;
constexpr size_t hc_pre_ksplit = 4;  // K chunks per logit in the mixes stage (more work-groups at decode)

size_t hc_pre_rows(const RuntimeParams& params) {
    const auto& shape = params.get_input_layout(0).get_shape();  // [..., n, e]
    size_t rows = 1;
    for (size_t i = 0; i + 2 < shape.size(); i++) {
        rows *= shape[i];
    }
    return rows;
}

class HyperConnectionPreGeneratorBase : public KernelGenerator {
public:
    explicit HyperConnectionPreGeneratorBase(std::string_view suffix) : KernelGenerator("hyper_connection_pre_opt", suffix) {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<hyper_connection_pre>();
        const auto& s = params.get_input_layout(0).get_partial_shape();
        jit.make("HC_N", desc->n_streams);
        jit.make("HC_E", s[s.size() - 1].get_length());
        jit.make("HC_MIX", (2 + desc->n_streams) * desc->n_streams);
        jit.make("HC_ITERS", desc->sinkhorn_iterations);
        jit.make("HC_EPS", desc->eps);
        jit.make("HC_RMS_EPS", desc->rms_eps);
        jit.make("HC_LWS", hc_pre_lws);
        jit.make("HC_KSPLIT", hc_pre_ksplit);
        return jit;
    }

    static void add_rows_scalar(const RuntimeParams& params, KernelData& kd) {
        kd.params.scalars.clear();
        scalar_desc s;
        s.t = scalar_desc::Types::INT32;
        s.v.s32 = static_cast<int32_t>(hc_pre_rows(params));
        kd.params.scalars.push_back(s);
    }
};

class HyperConnectionPreMixesGenerator : public HyperConnectionPreGeneratorBase {
public:
    HyperConnectionPreMixesGenerator() : HyperConnectionPreGeneratorBase("mixes") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = HyperConnectionPreGeneratorBase::get_jit_constants(params);
        jit.make("HC_PRE_MIXES", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
            assert(!params.is_dynamic());
            const auto n = params.typed_desc<hyper_connection_pre>()->n_streams;
            auto& wgs = kd.params.workGroups;
            wgs.global = {hc_pre_lws, static_cast<size_t>((2 + n) * n) * hc_pre_ksplit, hc_pre_rows(params)};
            wgs.local = {hc_pre_lws, 1, 1};
            add_rows_scalar(params, kd);
        }};
    }
};

class HyperConnectionPreMixSumGenerator : public HyperConnectionPreGeneratorBase {
public:
    HyperConnectionPreMixSumGenerator() : HyperConnectionPreGeneratorBase("mixsum") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = HyperConnectionPreGeneratorBase::get_jit_constants(params);
        jit.make("HC_PRE_MIXSUM", 1);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 2});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* /*rt_params*/) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            wgs.global = {hc_pre_rows(params) * hc_pre_lws, 1, 1};
            wgs.local = {hc_pre_lws, 1, 1};
            add_rows_scalar(params, kd);
        }};
    }
};

class HyperConnectionPreOptImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::HyperConnectionPreOptImpl)

    Stage::Ptr mixes = make_stage<HyperConnectionPreMixesGenerator>();
    Stage::Ptr mixsum = make_stage<HyperConnectionPreMixSumGenerator>();

    HyperConnectionPreOptImpl() : PrimitiveImplOCL(HyperConnectionPreOpt::get_type_info_static()) {}
    HyperConnectionPreOptImpl(const program_node& node, const RuntimeParams& params) : HyperConnectionPreOptImpl() {
        add_stage(mixes, params);
        add_stage(mixsum, params);
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<HyperConnectionPreOptImpl>(this);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        const auto n = params.typed_desc<hyper_connection_pre>()->n_streams;
        const auto& s = params.get_input_layout(0).get_partial_shape();
        size_t rows = 1;
        for (size_t i = 0; i + 2 < s.size(); i++) {
            rows *= s[i].is_static() ? static_cast<size_t>(s[i].get_length()) : size_t{1};
        }
        return {BufferDescriptor{std::max<size_t>(rows, 1) * static_cast<size_t>((2 + n) * n) * 2 * hc_pre_ksplit, ov::element::f32}};
    }
};

}  // namespace

std::unique_ptr<primitive_impl> HyperConnectionPreOpt::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<hyper_connection_pre>());
    return std::make_unique<HyperConnectionPreOptImpl>(node, params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::hyper_connection_pre)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::HyperConnectionPreOptImpl)
