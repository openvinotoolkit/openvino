// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xetla_convolution.hpp"

#include "primitive_cm_base.hpp"
#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::cm {
namespace {
    using PostOp = ConvolutionImplementationManager::PostOp;
    using GroupnormPostOp = ConvolutionImplementationManager::GroupnormPostOp;

class XetlaConvolutionGenerator : public KernelGenerator {
public:
    XetlaConvolutionGenerator() : KernelGenerator("xetla_convolution") {}
    ConvolutionImplementationManager::KernelKnobs kernel_knobs;
    ConvolutionImplementationManager::ConvDesc conv_desc;

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
                           make_jit_constant("POST_OP", uint32_t(conv_desc.post_op)),

                           make_jit_constant("SIZE_N", conv_desc.n),
                           make_jit_constant("SIZE_H", conv_desc.ih),
                           make_jit_constant("SIZE_W", conv_desc.iw),
                           make_jit_constant("SIZE_C", conv_desc.c),
                           make_jit_constant("SIZE_P", conv_desc.oh),
                           make_jit_constant("SIZE_Q", conv_desc.ow),
                           make_jit_constant("SIZE_K", conv_desc.k),

                           make_jit_constant("SRC_DT", "fp16"),
                           make_jit_constant("WEI_DT", "fp16"),
                           make_jit_constant("OUT_DT", "fp16"),
                           make_jit_constant("ACC_DT", "float"),
                           make_jit_constant("FILTER_SIZE", conv_desc.kernel_size),
                           make_jit_constant("PADDING", conv_desc.padding),
                           make_jit_constant("STRIDE", conv_desc.stride),
                           make_jit_constant("WG_TILE_N", kernel_knobs.wg_tile_n),
                           make_jit_constant("WG_TILE_P", kernel_knobs.wg_tile_p),
                           make_jit_constant("WG_TILE_Q", kernel_knobs.wg_tile_q),
                           make_jit_constant("WG_TILE_K", kernel_knobs.wg_tile_k),
                           make_jit_constant("SG_TILE_N", kernel_knobs.sg_tile_n),
                           make_jit_constant("SG_TILE_P", kernel_knobs.sg_tile_p),
                           make_jit_constant("SG_TILE_Q", kernel_knobs.sg_tile_q),
                           make_jit_constant("SG_TILE_K", kernel_knobs.sg_tile_k),
                           make_jit_constant("GLOBAL_SLICING", kernel_knobs.global_slicing),
                           make_jit_constant("LOCAL_SLICING", kernel_knobs.local_slicing),
                           make_jit_constant("PREFETCH_DISTANCE", kernel_knobs.prefetch_distance),
                           make_jit_constant("ACCUM_STEP", kernel_knobs.accum_step)});

        if (conv_desc.has_fused_groupnorm())
            jit_constants.add({make_jit_constant("GROUPNORM_GROUP_SIZE", conv_desc.group_size.value_or(0)),
                               make_jit_constant("GROUPNORM_GROUP_NUM", conv_desc.group_count.value_or(0)),
                               make_jit_constant("STAT_DT", "float")});

        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.push_back({ArgumentDescriptor::Types::WEIGHTS, 0});
        if (conv_desc.has_fused_groupnorm())
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});  // output
        else
            args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        switch (conv_desc.post_op) {
        case PostOp::Bias:
            args.push_back({ArgumentDescriptor::Types::BIAS, 0});
            break;
        case PostOp::BiasSum:
            args.push_back({ArgumentDescriptor::Types::BIAS, 0});
            args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, 0});
            break;
        case PostOp::BiasSumMulSum:
            args.push_back({ArgumentDescriptor::Types::BIAS, 0});
            args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, 0});
            args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, 1});
            args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, 2});
            break;
        case PostOp::Sum:
            args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, 0});
            break;
        case PostOp::SumMulSum:
            args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, 0});
            args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, 1});
            args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, 2});
            break;
        case PostOp::GnReduce:
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 5});
            break;
        case PostOp::BiasGnReduce:
            args.push_back({ArgumentDescriptor::Types::BIAS, 0});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 5});
            break;
        default:
            assert(0 && "Unknown PostOp value");
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{
            [kernel_knobs = kernel_knobs, conv_desc = conv_desc](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
                assert(!params.is_dynamic());
                auto& wgs = kd.params.workGroups;
                auto local_range_x = (kernel_knobs.wg_tile_k + kernel_knobs.sg_tile_k - 1) / kernel_knobs.sg_tile_k;
                auto local_range_y = (kernel_knobs.wg_tile_n + kernel_knobs.sg_tile_n - 1) / kernel_knobs.sg_tile_n;
                local_range_y *= (kernel_knobs.wg_tile_p + kernel_knobs.sg_tile_p - 1) / kernel_knobs.sg_tile_p;
                local_range_y *= (kernel_knobs.wg_tile_q + kernel_knobs.sg_tile_q - 1) / kernel_knobs.sg_tile_q;
                auto global_range_x = (conv_desc.k + kernel_knobs.wg_tile_k - 1) / kernel_knobs.wg_tile_k;
                auto global_range_y = (conv_desc.n + kernel_knobs.wg_tile_n - 1) / kernel_knobs.wg_tile_n;
                global_range_y *= (conv_desc.oh + kernel_knobs.wg_tile_p - 1) / kernel_knobs.wg_tile_p;
                global_range_y *= (conv_desc.ow + kernel_knobs.wg_tile_q - 1) / kernel_knobs.wg_tile_q;

                // multiply local & global slicing
                wgs.global = {global_range_x * local_range_x, global_range_y * local_range_y, kernel_knobs.global_slicing * kernel_knobs.local_slicing};
                wgs.local = {local_range_x, local_range_y, kernel_knobs.local_slicing};
            }};
    }
};

class XetlaGroupnormGenerator : public KernelGenerator {
public:
    XetlaGroupnormGenerator() : KernelGenerator("xetla_groupnorm") {}
    ConvolutionImplementationManager::NormKnobs norm_knobs;
    ConvolutionImplementationManager::ConvDesc conv_desc;

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
                           make_jit_constant("SIZE_N", conv_desc.n),
                           make_jit_constant("SIZE_W", conv_desc.ow * conv_desc.oh),
                           make_jit_constant("SIZE_C", conv_desc.c),
                           make_jit_constant("GROUP_COUNT", conv_desc.group_count.value_or(0)),
                           make_jit_constant("GROUP_SIZE", conv_desc.group_size.value_or(0)),
                           make_jit_constant("SRC_DT", "fp16"),
                           make_jit_constant("WEI_DT", "fp16"),
                           make_jit_constant("OUT_DT", "fp16"),
                           make_jit_constant("ACC_DT", "float"),
                           make_jit_constant("WG_TILE_N", norm_knobs.wg_tile_n),
                           make_jit_constant("WG_TILE_W", norm_knobs.wg_tile_w),
                           make_jit_constant("WG_TILE_C", norm_knobs.wg_tile_c),
                           make_jit_constant("SG_TILE_N", norm_knobs.sg_tile_n),
                           make_jit_constant("SG_TILE_W", norm_knobs.sg_tile_w),
                           make_jit_constant("SG_TILE_C", norm_knobs.sg_tile_c)/**/,
                           make_jit_constant("POST_OP",
                                static_cast<int>(conv_desc.gn_post_op.value_or(GroupnormPostOp::None)))});

        return jit_constants;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 3});           // src
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 4});           // sumx
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 5});           // sumxsq
        args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, 1});  // beta
        args.push_back({ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, 0});  // gamma
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});                    // dst
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[kernel_knobs = norm_knobs, conv_desc = conv_desc](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            assert(!params.is_dynamic());
            auto& wgs = kd.params.workGroups;
            auto local_range_n = (kernel_knobs.wg_tile_n + kernel_knobs.sg_tile_n - 1) / kernel_knobs.sg_tile_n;
            auto local_range_w = (kernel_knobs.wg_tile_w + kernel_knobs.sg_tile_w - 1) / kernel_knobs.sg_tile_w;
            auto local_range_c = (kernel_knobs.wg_tile_c + kernel_knobs.sg_tile_c - 1) / kernel_knobs.sg_tile_c;

            auto global_range_n = (conv_desc.n + kernel_knobs.wg_tile_n - 1) / kernel_knobs.wg_tile_n;
            auto global_range_w = (conv_desc.oh * conv_desc.ow + kernel_knobs.wg_tile_w - 1) / kernel_knobs.wg_tile_w;
            auto global_range_c = (conv_desc.k + kernel_knobs.wg_tile_c - 1) / kernel_knobs.wg_tile_c;

            // multiply local & global slicing
            wgs.global = {global_range_n * local_range_n, global_range_w * local_range_w, global_range_c * local_range_c};
            wgs.local = {local_range_n, local_range_w, local_range_c};
        }};
    }
};

class ConvolutionImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::ConvolutionImpl)
    Stage::Ptr conv = make_stage<XetlaConvolutionGenerator>();
    Stage::Ptr groupnorm = make_stage<XetlaGroupnormGenerator>();

    ConvolutionImpl() : PrimitiveImplCM(ConvolutionImplementationManager::get_type_info_static()) {}
    ConvolutionImpl(const program_node& node, const RuntimeParams& params) : ConvolutionImpl() {
        // Pass KernelKnobs to generator
        auto conv_desc = ConvolutionImplementationManager::ConvDesc::from_node(node).value();
        auto key = conv_desc.get_shape_key();
        auto conv_gen = dynamic_cast<XetlaConvolutionGenerator*>(conv->codegen.get());
        conv_gen->kernel_knobs = ConvolutionImplementationManager::ConvMap.at(key);
        conv_gen->conv_desc = conv_desc;
        add_stage(conv, params);
        if (conv_desc.has_fused_groupnorm()) {
            auto groupnorm_gen = dynamic_cast<XetlaGroupnormGenerator*>(groupnorm->codegen.get());
            groupnorm_gen->norm_knobs = ConvolutionImplementationManager::NormMap.at(key);
            groupnorm_gen->conv_desc = conv_desc;
            add_stage(groupnorm, params);
        }
    }
    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<ConvolutionImpl>(this);
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        // This is not correct but rather approximation to integrate fast
        auto desc = ConvolutionImplementationManager::ConvDesc::from_rt_params(params).value();
        auto size_scratchpad = desc.n * desc.oh * desc.ow * desc.k;
        auto size_acc = size_scratchpad;
        auto size_cnt = size_scratchpad;

        auto buffers = std::vector<BufferDescriptor>{
            BufferDescriptor{size_scratchpad, ov::element::f32},
            BufferDescriptor{size_acc, ov::element::f32},
            BufferDescriptor{size_cnt, ov::element::u32}
        };
        if (desc.has_fused_groupnorm()) {
            auto size_sumx = desc.n * desc.group_count.value();
            buffers.push_back(BufferDescriptor{size_scratchpad, ov::element::f16});
            buffers.push_back(BufferDescriptor{size_sumx, ov::element::f32});
            buffers.push_back(BufferDescriptor{size_sumx, ov::element::f32});
        }
        return buffers;
    }

    [[nodiscard]] cldnn::kernel_arguments_data get_arguments(const cldnn::primitive_inst& instance) const override {
        cldnn::kernel_arguments_data args = PrimitiveImplCM::get_arguments(instance);
        const auto& conv_instance = reinterpret_cast<const typed_primitive_inst<convolution>&>(instance);
        args.weights = conv_instance.weights_memory();
        args.bias = conv_instance.bias_term() ? conv_instance.bias_memory() : nullptr;
        return args;
    }
};

}  // namespace

std::unique_ptr<primitive_impl> ConvolutionImplementationManager::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<convolution>());
    return std::make_unique<ConvolutionImpl>(node, params);
}

const std::unordered_map<std::string, ConvolutionImplementationManager::KernelKnobs> ConvolutionImplementationManager::ConvMap = {
    // N x H x W x C x K x KERNEL_SIZE x STRIDE x DILATION x PADDING
    {"1x14x14x256x512x3x1x1x1", KernelKnobs{1, 7 * 2, 8 * 2, 32 * 4, 1, 7, 8, 32, 2, 1, 3, 32}},  // C2 (post op variants: [bias+eltwise_add])
    {"1x7x7x256x512x3x1x1x1", KernelKnobs{1, 1 * 7, 8 * 1, 32 * 2, 1, 1, 8, 32, 1, 1, 3, 32}},    // C2 (post op variants: [bias+eltwise_add])
    {"1x7x7x512x512x3x1x1x1", KernelKnobs{1, 7 * 1, 8 * 1, 32 * 8, 1, 7, 8, 32, 4, 1, 3, 32}},    // C2 (post op variants: [bias])
    {"1x64x128x64x128x4x2x1x1", KernelKnobs{1, 4 * 4, 8 * 2, 32 * 4, 1, 4, 8, 32, 1, 1, 3, 32}},  // C4 (post op variants: [none])
    {"1x32x64x128x256x4x2x1x1", KernelKnobs{1, 4 * 4, 8 * 2, 32 * 2, 1, 4, 8, 32, 1, 1, 3, 32}},  // C4 (post op variants: [none])
    {"1x16x32x256x256x3x1x1x1", KernelKnobs{1, 8 * 2, 8 * 2, 32 * 2, 1, 8, 8, 32, 1, 1, 3, 32}},  // C4 (post op variants: [none])
    // J (post op variants: [bias], [eltwise_add], [bias+eltwise_add+eltwise_mul+eltwise_add])
    {"1x10x18x256x256x3x1x1x1", KernelKnobs{1, 5 * 2, 8 * 2, 32 * 2, 1, 5, 8, 32, 1, 1, 3, 32}},
    {"1x10x18x256x512x3x2x1x1", KernelKnobs{1, 5 * 2, 8 * 2, 32 * 2, 1, 5, 8, 32, 1, 1, 3, 32}},  // J (post op variants: [bias])
    // J (post op variants: [bias], [eltwise_add], [bias+eltwise_add+eltwise_mul+eltwise_add])
    {"1x5x9x512x512x3x1x1x1", KernelKnobs{1, 1 * 5, 8 * 2, 32 * 2, 1, 1, 8, 32, 1, 1, 3, 32}},
    // VAE Decoder
    {"1x64x64x4x512x3x1x1x1", KernelKnobs{1, 16, 16, 64, 1, 4, 8, 32, 1, 1, 3, 32}},
    {"1x64x64x512x512x3x1x1x1", KernelKnobs{1, 32, 16, 128, 1, 8, 8, 32, 1, 1, 3, 32}},
    {"1x128x128x512x512x3x1x1x1", KernelKnobs{1, 32, 16, 128, 1, 8, 8, 32, 1, 1, 3, 32}},
    {"1x256x256x512x512x3x1x1x1", KernelKnobs{1, 32, 16, 128, 1, 8, 8, 32, 1, 1, 3, 32}},
    {"1x256x256x256x256x3x1x1x1", KernelKnobs{1, 32, 16, 128, 1, 8, 8, 32, 1, 1, 3, 32}},
    {"1x512x512x128x128x3x1x1x1", KernelKnobs{1, 64, 16, 64, 1, 8, 8, 32, 1, 1, 3, 32}},
    {"1x512x512x256x128x1x0x1x1", KernelKnobs{1, 16, 64, 64, 1, 4, 16, 32, 1, 1, 3, 32}},
    // {"1x512x512x128x128x1x1x1x1", KernelKnobs{1, 16, 16, 128, 1, 8, 4, 32, 1, 1, 3, 32}},
    {"1x512x512x256x128x3x1x1x1", KernelKnobs{1, 32, 32, 64, 1, 8, 8, 32, 1, 1, 3, 32}}
};

const std::unordered_map<std::string, ConvolutionImplementationManager::NormKnobs> ConvolutionImplementationManager::NormMap = {
    {"1x14x14x256x512x3x1x1x1", NormKnobs{1, 7 * 2 * 8 * 2, 32 * 4, 1, 7 * 8, 32}},  // C2 (post op variants: [bias+eltwise_add])
    {"1x7x7x256x512x3x1x1x1", NormKnobs{1, 1 * 7 * 8 * 1, 32 * 2, 1, 1 * 8, 32}},    // C2 (post op variants: [bias+eltwise_add])
    {"1x7x7x512x512x3x1x1x1", NormKnobs{1, 7 * 1 * 8 * 1, 32 * 8, 1, 7 * 8, 32}},    // C2 (post op variants: [bias])
    {"1x64x128x64x128x4x2x1x1", NormKnobs{1, 4 * 4 * 8 * 2, 32 * 4, 1, 4 * 8, 32}},  // C4 (post op variants: [none])
    {"1x32x64x128x256x4x2x1x1", NormKnobs{1, 4 * 4 * 8 * 2, 32 * 2, 1, 4 * 8, 32}},  // C4 (post op variants: [none])
    {"1x16x32x256x256x3x1x1x1", NormKnobs{1, 8 * 2 * 8 * 2, 32 * 2, 1, 8 * 8, 32}},  // C4 (post op variants: [none])
    // J (post op variants: [bias], [eltwise_add], [bias+eltwise_add+eltwise_mul+eltwise_add])
    {"1x10x18x256x256x3x1x1x1", NormKnobs{1, 5 * 2 * 8 * 2, 32 * 2, 1, 5 * 8, 32}},
    {"1x10x18x256x512x3x2x1x1", NormKnobs{1, 5 * 2 * 8 * 2, 32 * 2, 1, 5 * 8, 32}},  // J (post op variants: [bias])
    // J (post op variants: [bias], [eltwise_add], [bias+eltwise_add+eltwise_mul+eltwise_add])
    {"1x5x9x512x512x3x1x1x1", NormKnobs{1, 1 * 5 * 8 * 2, 32 * 2, 1, 1 * 8, 32}},
    // VAE Decoder
    {"1x64x64x4x512x3x1x1x1", NormKnobs{1, 512, 64, 1, 64, 16}},
    {"1x64x64x512x512x3x1x1x1", NormKnobs{1, 512, 64, 1, 64, 16}},
    {"1x128x128x512x512x3x1x1x1", NormKnobs{1, 512, 128, 1, 128, 16}},
    {"1x256x256x512x512x3x1x1x1", NormKnobs{1, 512, 64, 1, 128, 16}},
    {"1x256x256x256x256x3x1x1x1", NormKnobs{1, 512, 128, 1, 128, 16}},
    {"1x512x512x128x128x3x1x1x1", NormKnobs{1, 512, 32, 1, 64, 16}},
    {"1x512x512x256x128x1x0x1x1", NormKnobs{1, 512, 32, 1, 64, 16}},
    // {"1x512x512x128x128x1x1x1x1", NormKnobs{1, 16 * 16, 128, 1, 8 * 4, 32}},
    {"1x512x512x256x128x3x1x1x1", NormKnobs{1, 512, 32, 1, 64, 16}}
};
}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::ConvolutionImpl)
