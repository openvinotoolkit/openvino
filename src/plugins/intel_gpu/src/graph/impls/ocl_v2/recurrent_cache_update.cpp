// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "recurrent_cache_update.hpp"

#include "intel_gpu/primitives/recurrent_cache_update.hpp"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"
namespace ov::intel_gpu::ocl {
namespace {
size_t subgroup(gpu_arch arch) {
    return one_of(arch, {gpu_arch::gen9, gpu_arch::gen11, gpu_arch::xe_lp, gpu_arch::xe_hp, gpu_arch::xe_hpg}) ? 8 : 16;
}
class UpdateGenerator : public KernelGenerator {
    int m_stage;

public:
    UpdateGenerator(bool conv, int stage)
        : KernelGenerator(conv ? "recurrent_conv_cache_update" : "recurrent_gated_delta_net_update", std::to_string(stage)),
          m_stage(stage) {}

protected:
    JitConstants get_jit_constants(const RuntimeParams& p) const override {
        auto j = KernelGenerator::get_jit_constants(p);
        auto d = p.typed_desc<recurrent_cache_update>();
        bool conv = d->history > 0;
        j.make("INPLACE_STAGE", m_stage);
        j.make("HISTORY", d->history);
        if (!conv) {
            auto q = p.get_input_layout(0).get_partial_shape(), v = p.get_input_layout(2).get_partial_shape();
            j.make("K_HEAD_NUM", q[2].get_length());
            j.make("V_HEAD_NUM", v[2].get_length());
            j.make("K_HEAD_DIM", q[3].get_length());
            j.make("V_HEAD_DIM", v[3].get_length());
            j.make("SUBGROUP_SIZE", subgroup(p.get_device_info().arch));
            j.make("V_BLOCK_SIZE", 4);
            j.make("FUSE_QK_L2NORM", d->fuse_qk_l2norm ? 1 : 0);
            j.make("Q_L2_NORM_EPS", d->q_l2_norm_eps);
            j.make("K_L2_NORM_EPS", d->k_l2_norm_eps);
            j.make("SCALE_FACTOR", 1.0f / std::sqrt(float(q[3].get_length())));
            j.make("SNAPSHOT_COUNT", d->snapshot_count);
        }
        return j;
    }
    Arguments get_arguments_desc(const RuntimeParams& p) const override {
        Arguments a;
        if (p.is_dynamic())
            a.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        for (uint32_t i = 0; i < p.input_layouts.size(); ++i)
            a.push_back({ArgumentDescriptor::Types::INPUT, i});
        for (uint32_t i = 0; i < 2; ++i)
            a.push_back({ArgumentDescriptor::Types::OUTPUT, i});
        bool gdn = p.typed_desc<recurrent_cache_update>()->history == 0;
        if (m_stage == 2 && gdn)
            for (uint32_t i = 0; i < 7; ++i)
                a.push_back({ArgumentDescriptor::Types::SCALAR, i});
        a.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        a.push_back({ArgumentDescriptor::Types::SCALAR, m_stage == 2 && gdn ? 7u : 0u});
        return a;
    }
    DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[stage = m_stage](const RuntimeParams& p, KernelData& kd, ImplRuntimeParams*) {
            auto d = p.typed_desc<recurrent_cache_update>();
            bool conv = d->history > 0;
            const auto cache = p.get_input_layout(conv ? 0 : 3);
            auto data = p.get_input_layout(conv ? 6 : 0).get_partial_shape();
            auto& wg = kd.params.workGroups;
            if (stage == 2 && !conv) {
                auto v = p.get_input_layout(2).get_partial_shape();
                auto sg = subgroup(p.get_device_info().arch);
                wg.global = {size_t(data[0].get_length() + cache.get_partial_shape()[2].get_length()),
                             size_t(v[2].get_length()),
                             size_t((v[3].get_length() + 3) / 4) * sg};
                wg.local = {1, 1, sg};
            } else {
                size_t count = stage == 0 ? (size_t(p.get_input_layout(conv ? 1 : 6).count()) + p.get_input_layout(conv ? 2 : 7).count()) *
                                                size_t(cache.get_partial_shape()[3].get_length())
                                          : std::max(p.output_layouts[0].count(), p.output_layouts[1].count());
                wg.global = {((count + 255) / 256) * 256, 1, 1};
                wg.local = {256, 1, 1};
            }
            kd.params.scalars.clear();
            if (stage == 2 && !conv) {
                std::vector<int32_t> s{int32_t(data[1].get_length())};
                for (size_t i = 0; i < 3; ++i) {
                    auto pads = p.get_input_layout(i).data_padding._lower_size;
                    s.push_back(pads.size() > 2 ? int32_t(pads[2]) : 0);
                }
                for (size_t i = 0; i < 3; ++i) {
                    auto pitch = p.get_input_layout(i).get_pitches();
                    s.push_back(int32_t(pitch[1]));
                }
                for (auto v : s) {
                    scalar_desc x;
                    x.t = scalar_desc::Types::INT32;
                    x.v.s32 = v;
                    kd.params.scalars.push_back(x);
                }
            }
            scalar_desc x;
            x.t = scalar_desc::Types::INT32;
            x.v.s32 = 0;
            kd.params.scalars.push_back(x);
        }};
    }
};
class CacheUpdateImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::CacheUpdateImpl)
    // The gather stage is a global read-before-write boundary. The compute
    // stage owns reset/reorder/writeback too, avoiding a third launch and allowing
    // recurrence owners to skip cache rows that their snapshots overwrite.
    Stage::Ptr conv_gather = make_stage<UpdateGenerator>(true, 0);
    Stage::Ptr conv_compute = make_stage<UpdateGenerator>(true, 2);
    Stage::Ptr gdn_gather = make_stage<UpdateGenerator>(false, 0);
    Stage::Ptr gdn_compute = make_stage<UpdateGenerator>(false, 2);
    CacheUpdateImpl() : PrimitiveImplOCL(RecurrentCacheUpdate::get_type_info_static()) {}
    CacheUpdateImpl(const program_node&, const RuntimeParams& p) : CacheUpdateImpl() {
        bool c = p.typed_desc<recurrent_cache_update>()->history > 0;
        add_stage(c ? conv_gather : gdn_gather, p);
        add_stage(c ? conv_compute : gdn_compute, p);
        OPENVINO_ASSERT(_order.size() == 2, "Cache Update requires gather and compute stages");
    }
    std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<CacheUpdateImpl>(this);
    }
    bool supports_replay() const override {
        return false;
    }  // Stage choice depends on caller memory aliasing.
    std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& p) const override {
        bool c = p.typed_desc<recurrent_cache_update>()->history > 0;
        size_t n = 1;
        if (!p.is_dynamic()) {
            auto cache = p.get_input_layout(c ? 0 : 3);
            n = (p.get_input_layout(c ? 1 : 6).count() + p.get_input_layout(c ? 2 : 7).count()) * size_t(cache.get_partial_shape()[3].get_length());
        }
        return {BufferDescriptor(std::max<size_t>(n, 1), ov::element::f32, false, false)};
    }
    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& inst) override {
        kernel_dump_info.clear_entries();
        update_rt_params(inst);
        bool conv = inst.get_impl_params()->typed_desc<recurrent_cache_update>()->history > 0;
        auto& engine = inst.get_network().get_engine();
        bool inplace = engine.is_the_same_buffer(inst.input_memory(conv ? 0 : 3), inst.output_memory(1));
        auto& gather = conv ? conv_gather : gdn_gather;
        auto& compute = conv ? conv_compute : gdn_compute;
        std::vector<event::ptr> deps = events;
        auto run = [&](Stage::Ptr& stage) {
            auto& kd = stage->kd;
            if (kd.need_dispatch_data_update) {
                kd.update_dispatch_data_func(*inst.get_impl_params(), kd, m_rt_params.get());
                kd.need_dispatch_data_update = false;
            }
            // A binding can alternate between alias and distinct without a shape
            // change. Update the scalar/arguments only when this mode changes.
            const int32_t mode = inplace ? 1 : 0;
            kd.need_args_update |= kd.params.scalars.back().v.s32 != mode;
            kd.params.scalars.back().v.s32 = mode;
            auto e = execute_stage(deps, inst, stage);
            deps = {e};
            return e;
        };
        if (inplace) {
            run(gather);
        }
        return run(compute);
    }
};
}  // namespace
std::unique_ptr<primitive_impl> RecurrentCacheUpdate::create_impl(const program_node& n, const RuntimeParams& p) const {
    return std::make_unique<CacheUpdateImpl>(n, p);
}
}  // namespace ov::intel_gpu::ocl
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::recurrent_cache_update)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::CacheUpdateImpl)
