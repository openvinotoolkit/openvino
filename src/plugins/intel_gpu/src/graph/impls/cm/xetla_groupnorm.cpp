// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xetla_groupnorm.hpp"

#include "primitive_cm_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::cm {
namespace {
    using PostOp = GroupnormImplementationManager::PostOp;

class XetlaGroupnormGenerator : public KernelGenerator {
public:
    XetlaGroupnormGenerator() : KernelGenerator("xetla_groupnorm") {}
    GroupnormImplementationManager::NormKnobs norm_knobs;
    GroupnormImplementationManager::GroupnormDesc gn_desc;

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + " -Qxcm_jit_option=-DPASTokenReduction "
                                                            " -mllvm --vc-disable-indvars-opt=true "
                                                            " /Qxcm_jit_option=-enableBCR /Qxcm_doubleGRF "
                                                            " -DXETLA_CODE_BASE=__CM__ ";
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit_constants = KernelGenerator::get_jit_constants(params);
        jit_constants.add({make_jit_constant("KERNEL_NAME", get_entry_point(params)),
                           make_jit_constant("SIZE_N", gn_desc.n),
                           make_jit_constant("SIZE_W", gn_desc.w),
                           make_jit_constant("SIZE_C", gn_desc.c),
                           make_jit_constant("GROUP_COUNT", gn_desc.group_count),
                           make_jit_constant("GROUP_SIZE", gn_desc.group_size),
                           make_jit_constant("SRC_DT", "fp16"),
                           make_jit_constant("WEI_DT", "fp16"),
                           make_jit_constant("OUT_DT", "fp16"),
                           make_jit_constant("ACC_DT", "float"),
                           make_jit_constant("WG_TILE_N", norm_knobs.wg_tile_n),
                           make_jit_constant("WG_TILE_W", norm_knobs.wg_tile_w),
                           make_jit_constant("WG_TILE_C", norm_knobs.wg_tile_c),
                           make_jit_constant("SG_TILE_N", norm_knobs.sg_tile_n),
                           make_jit_constant("SG_TILE_W", norm_knobs.sg_tile_w),
                           make_jit_constant("SG_TILE_C", norm_knobs.sg_tile_c),
                           make_jit_constant("POST_OP", static_cast<int>(gn_desc.post_op))});

        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});           // src
        args.push_back({ArgumentDescriptor::Types::INPUT, 3});           // sumx
        args.push_back({ArgumentDescriptor::Types::INPUT, 4});           // sumxsq
        args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // beta
        args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // gamma
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});                    // dst
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[kernel_knobs = norm_knobs, gn_desc = gn_desc](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            auto local_range_n = (kernel_knobs.wg_tile_n + kernel_knobs.sg_tile_n - 1) / kernel_knobs.sg_tile_n;
            auto local_range_w = (kernel_knobs.wg_tile_w + kernel_knobs.sg_tile_w - 1) / kernel_knobs.sg_tile_w;
            auto local_range_c = (kernel_knobs.wg_tile_c + kernel_knobs.sg_tile_c - 1) / kernel_knobs.sg_tile_c;

            auto global_range_n = (gn_desc.n + kernel_knobs.wg_tile_n - 1) / kernel_knobs.wg_tile_n;
            auto global_range_w = (gn_desc.w + kernel_knobs.wg_tile_w - 1) / kernel_knobs.wg_tile_w;
            auto global_range_c = (gn_desc.c + kernel_knobs.wg_tile_c - 1) / kernel_knobs.wg_tile_c;

            // multiply local & global slicing
            wgs.global = {global_range_n * local_range_n, global_range_w * local_range_w, global_range_c * local_range_c};
            wgs.local = {local_range_n, local_range_w, local_range_c};
        }};
    }
};

class GroupnormImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::GroupnormImpl)
    Stage::Ptr groupnorm = make_stage<XetlaGroupnormGenerator>();

    GroupnormImpl() : PrimitiveImplCM(GroupnormImplementationManager::get_type_info_static()) {}
    GroupnormImpl(const program_node& node, const RuntimeParams& params) : GroupnormImpl() {
        // Pass KernelKnobs to generator
        auto gn_desc = GroupnormImplementationManager::GroupnormDesc::from_node(node).value();
        auto key = gn_desc.get_shape_key();
        auto groupnorm_gen = dynamic_cast<XetlaGroupnormGenerator*>(groupnorm->codegen.get());
        groupnorm_gen->norm_knobs = GroupnormImplementationManager::NormMap.at(key);
        groupnorm_gen->gn_desc = gn_desc;
        add_stage(groupnorm, params);
    }
    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<GroupnormImpl>(this);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        return {};
    }

    [[nodiscard]] cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const override {
        cldnn::kernel_arguments_data args = PrimitiveImplCM::get_arguments(instance);
        return args;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> GroupnormImplementationManager::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<group_normalization>());
    return std::make_unique<GroupnormImpl>(node, params);
}

const std::unordered_map<std::string, GroupnormImplementationManager::NormKnobs> GroupnormImplementationManager::NormMap = {
    // VAE Decoder
    {"1x4096x512", NormKnobs{1, 512, 64, 1, 64, 16}},
    {"1x16384x512", NormKnobs{1, 512, 128, 1, 128, 16}},
    {"1x65536x512", NormKnobs{1, 512, 64, 1, 128, 16}},
    {"1x65536x256", NormKnobs{1, 512, 128, 1, 128, 16}},
    {"1x262144x128", NormKnobs{1, 512, 32, 1, 64, 16}}
};
}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::GroupnormImpl)
