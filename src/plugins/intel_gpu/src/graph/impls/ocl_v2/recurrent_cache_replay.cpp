// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "recurrent_cache_replay.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::ocl {
namespace {
enum class ReplayStage { gather, compute };
size_t subgroup(gpu_arch arch) {
    return one_of(arch, {gpu_arch::gen9, gpu_arch::gen11, gpu_arch::xe_lp, gpu_arch::xe_hp, gpu_arch::xe_hpg}) ? 8 : 16;
}
class ReplayGenerator : public KernelGenerator {
    bool m_conv;
    ReplayStage m_stage;

  public:
    ReplayGenerator(bool conv, ReplayStage stage)
        : KernelGenerator(conv ? "recurrent_conv_cache_replay" : "recurrent_gdn_cache_replay",
                          std::to_string(static_cast<int>(stage))),
          m_conv(conv), m_stage(stage) {}

  protected:
    JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto constants = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<recurrent_cache_replay>();
        constants.make("REPLAY_STAGE", static_cast<int>(m_stage));
        constants.make("HISTORY", desc->history);
        if (!m_conv) {
            const auto q = params.get_input_layout(0).get_partial_shape();
            const auto k = params.get_input_layout(1).get_partial_shape();
            const auto v = params.get_input_layout(2).get_partial_shape();
            constants.make("Q_HEAD_NUM", q[2].get_length());
            constants.make("K_HEAD_NUM", k[2].get_length());
            constants.make("V_HEAD_NUM", v[2].get_length());
            constants.make("K_HEAD_DIM", q[3].get_length());
            constants.make("V_HEAD_DIM", v[3].get_length());
            constants.make("GATE_DIM", params.get_input_layout(4).get_partial_shape()[3].get_length());
            constants.make("SUBGROUP_SIZE", subgroup(params.get_device_info().arch));
            constants.make("V_BLOCK_SIZE", 4);
            constants.make("GROUPED_HEADS", desc->grouped_heads ? 1 : 0);
            constants.make("FUSE_QK_L2NORM", desc->fuse_qk_l2norm ? 1 : 0);
            constants.make("Q_L2_NORM_EPS", desc->q_l2_norm_eps);
            constants.make("K_L2_NORM_EPS", desc->k_l2_norm_eps);
            constants.make("SCALE_FACTOR", 1.0f / std::sqrt(float(q[3].get_length())));
        }
        return constants;
    }
    Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        if (params.is_dynamic())
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        // The bank arguments point at the updated-bank outputs. For a distinct
        // binding execute() initializes the output with the original bank.
        uint32_t bank_output = 1;
        for (uint32_t i = 0; i < params.input_layouts.size(); ++i) {
            const bool bank = m_conv ? i == 0 : i == 3 || i == 6 || i == 7 || i == 8;
            if (bank)
                args.push_back({ArgumentDescriptor::Types::OUTPUT, bank_output++});
            else
                args.push_back({ArgumentDescriptor::Types::INPUT, i});
        }
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        for (uint32_t i = 0; i < 5; ++i)
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        return args;
    }
    DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{
            [conv = m_conv, stage = m_stage](const RuntimeParams& params, KernelData& kernel, ImplRuntimeParams*) {
                const auto desc = params.typed_desc<recurrent_cache_replay>();
                const auto data = params.get_input_layout(conv ? 2 : 0).get_partial_shape();
                const auto B = size_t(data[conv ? 1 : 0].get_length());
                const auto T = size_t(data[conv ? 2 : 1].get_length());
                const auto N = params.get_input_layout(conv ? 1 : 9).count() / 2;
                const auto J = params.get_input_layout(conv ? 5 : 11).count();
                size_t features = params.get_input_layout(conv ? 0 : 3).get_partial_shape()[3].get_length();
                OPENVINO_ASSERT(N >= B && params.get_input_layout(conv ? 4 : 10).count() == B + 1,
                                "Replay needs one mapping and count for every active sequence");
                const auto slots = params.get_input_layout(conv ? 0 : 3).get_partial_shape()[2].get_length();
                OPENVINO_ASSERT(slots > 0 && B > 0 && T > 0, "Replay requires nonempty banks and active tokens");
                if (!conv) {
                    for (size_t port : {1, 2, 4, 5}) {
                        const auto shape = params.get_input_layout(port).get_partial_shape();
                        OPENVINO_ASSERT(shape[0] == data[0] && shape[1] == data[1],
                                        "Replay token and sequence dimensions differ");
                    }
                    for (size_t port : {6, 7, 8})
                        OPENVINO_ASSERT(params.get_input_layout(port).get_partial_shape()[2] == slots,
                                        "Replay bank capacities differ");
                }
                size_t pending;
                if (conv) {
                    pending = features / size_t(data[3].get_length()) - desc->history;
                } else {
                    const auto k = params.get_input_layout(1).get_partial_shape();
                    pending = params.get_input_layout(6).get_partial_shape()[3].get_length() /
                              (k[2].get_length() * k[3].get_length());
                    for (size_t port : {6, 7, 8})
                        features += params.get_input_layout(port).get_partial_shape()[3].get_length();
                }
                const auto local =
                    conv || stage == ReplayStage::gather ? size_t(256) : subgroup(params.get_device_info().arch);
                size_t groups;
                if (stage == ReplayStage::gather) {
                    groups = (N * features + local - 1) / local;
                } else if (conv) {
                    const auto C = size_t(data[3].get_length());
                    const auto output_groups = (B * T * C + local - 1) / local;
                    const auto history_groups = (B * size_t(desc->history) * C + local - 1) / local;
                    groups = output_groups + history_groups + (N - B + J) * ((features + local - 1) / local);
                } else {
                    const auto v = params.get_input_layout(2).get_partial_shape();
                    groups = B * v[2].get_length() * ((v[3].get_length() + 3) / 4) +
                             (N - B + J) * ((features + local * 4 - 1) / (local * 4));
                }
                kernel.params.workGroups.global = {std::max<size_t>(groups, 1) * local, 1, 1};
                kernel.params.workGroups.local = {local, 1, 1};
                kernel.params.scalars.clear();
                for (auto value : {B, T, N, J, pending}) {
                    scalar_desc scalar;
                    scalar.t = scalar_desc::Types::INT32;
                    scalar.v.s32 = static_cast<int32_t>(value);
                    kernel.params.scalars.push_back(scalar);
                }
            }};
    }
};
class CacheReplayImpl : public PrimitiveImplOCL {
  public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::CacheReplayImpl)
    Stage::Ptr conv_gather = make_stage<ReplayGenerator>(true, ReplayStage::gather);
    Stage::Ptr conv_compute = make_stage<ReplayGenerator>(true, ReplayStage::compute);
    Stage::Ptr gdn_gather = make_stage<ReplayGenerator>(false, ReplayStage::gather);
    Stage::Ptr gdn_compute = make_stage<ReplayGenerator>(false, ReplayStage::compute);
    CacheReplayImpl() : PrimitiveImplOCL(RecurrentCacheReplay::get_type_info_static()) {}
    CacheReplayImpl(const program_node&, const RuntimeParams& params) : CacheReplayImpl() {
        const bool conv = params.typed_desc<recurrent_cache_replay>()->history > 0;
        add_stage(conv ? conv_gather : gdn_gather, params);
        add_stage(conv ? conv_compute : gdn_compute, params);
    }
    std::unique_ptr<primitive_impl> clone() const override { return make_deep_copy<CacheReplayImpl>(this); }
    std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        size_t count = 1;
        if (!params.is_dynamic()) {
            const bool conv = params.typed_desc<recurrent_cache_replay>()->history > 0;
            size_t features = params.get_input_layout(conv ? 0 : 3).get_partial_shape()[3].get_length();
            if (!conv)
                for (size_t port : {6, 7, 8})
                    features += params.get_input_layout(port).get_partial_shape()[3].get_length();
            count = (params.get_input_layout(conv ? 1 : 9).count() / 2) * features;
        }
        return {BufferDescriptor(std::max<size_t>(count, 1), ov::element::f32, false, false)};
    }
    cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& inst) const override {
        auto args = PrimitiveImplOCL::get_arguments(inst);
        const auto reset = inst.get_impl_params()->typed_desc<recurrent_cache_replay>()->history > 0 ? 5 : 11;
        // Empty reset lists have no allocation. J=0 prevents any read of this placeholder.
        if (!args.inputs[reset] && inst.get_impl_params()->get_input_layout(reset).count() == 0)
            args.inputs[reset] = args.intermediates[0];
        return args;
    }
    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& inst) override {
        kernel_dump_info.clear_entries();
        update_rt_params(inst);
        const bool conv = inst.get_impl_params()->typed_desc<recurrent_cache_replay>()->history > 0;
        auto& gather = conv ? conv_gather : gdn_gather;
        auto& compute = conv ? conv_compute : gdn_compute;
        std::vector<event::ptr> dependencies = events;
        // All persisted outputs use the corresponding input layout, including its
        // offset and pitches. Preserve inactive rows and partial destination tails.
        auto& stream = inst.get_network().get_stream();
        size_t bank_index = 0;
        for (size_t input : {3, 6, 7, 8}) {
            const auto src = inst.dep_memory_ptr(conv ? 0 : input);
            const auto dst = inst.output_memory_ptr(++bank_index);
            OPENVINO_ASSERT(src->get_layout() == dst->get_layout(), "Replay bank layout changed at dispatch");
            GPU_DEBUG_TRACE_DETAIL << "Replay bank " << bank_index
                                   << " direct=" << inst.get_network().get_engine().is_the_same_buffer(*src, *dst)
                                   << std::endl;
            if (!inst.get_network().get_engine().is_the_same_buffer(*src, *dst)) {
                stream.wait_for_events(events);
                if (auto copy = dst->copy_from(stream, *src, false))
                    dependencies.push_back(copy);
            }
            if (conv)
                break;
        }
        auto run = [&](Stage::Ptr& stage) {
            auto& kernel = stage->kd;
            if (kernel.need_dispatch_data_update) {
                kernel.update_dispatch_data_func(*inst.get_impl_params(), kernel, m_rt_params.get());
                kernel.need_dispatch_data_update = false;
            }
            auto event = execute_stage(dependencies, inst, stage);
            dependencies = {event};
            return event;
        };
        // This dependency is the global original-read boundary for all banks.
        run(gather);
        return run(compute);
    }
};
} // namespace
std::unique_ptr<primitive_impl> RecurrentCacheReplay::create_impl(const program_node& node,
                                                                  const RuntimeParams& params) const {
    return std::make_unique<CacheReplayImpl>(node, params);
}
} // namespace ov::intel_gpu::ocl
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::recurrent_cache_replay)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::CacheReplayImpl)
