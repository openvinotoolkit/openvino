// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <optional>
#include <sstream>
#include <unordered_map>

#include "convolution_inst.h"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::cm {

struct ConvolutionImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::conv")
    explicit ConvolutionImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::cm, shape_type, std::move(vf)) {}

    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<convolution>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        in_fmts[0] = format::byxf;
        in_fmts[1] = format::yxio;
        for (size_t idx = 2; idx < node.get_dependencies().size(); idx++) {
            in_fmts[idx] = format::byxf;
        }
        for (size_t idx = 0; idx < node.get_outputs_count(); idx++) {
            out_fmts[idx] = format::byxf;
        }

        return {in_fmts, out_fmts};
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<convolution>());

        auto& engine = node.get_program().get_engine();
        const auto& config = node.get_program().get_config();
        // const auto& info = engine.get_device_info();

        if (!check_cm_jit_support(engine, config) || /*info.arch != gpu_arch::xe2 ||*/ !config.get_use_cm()) {
            return false;
        }

        auto desc = ConvDesc::from_node(node);
        if (!desc.has_value())
            return false;

        auto key = desc.value().get_shape_key();

        if (ConvMap.find(key) == ConvMap.end())
            return false;

        if (desc.value().has_fused_groupnorm())
            if (NormMap.find(key) == NormMap.end())
                return false;

        return true;
    }

    enum class PostOp : uint32_t{
        None = 0,
        Bias,
        BiasSum,
        BiasSumMulSum,
        Sum,
        SumMulSum,
        GnReduce,
        BiasGnReduce,
    };

    struct ConvDesc {
        size_t n, ih, iw, c, k, kernel_size, stride, dilation, padding;
        size_t ow, oh;
        std::optional<size_t> group_count, group_size;
        PostOp post_op;

        bool process_fused_ops(const std::vector<cldnn::fused_primitive_desc> &fused_ops, bool bias) {
            // should check fused ops shapes, dtypes,...
            if (fused_ops.size() == 0) {
                if (!bias)
                    post_op = PostOp::None;
                else
                    post_op = PostOp::Bias;
            } else if (fused_ops.size() == 1) {
                if (fused_ops[0].is_type<group_normalization>()) {
                    auto groupnorm0 = std::static_pointer_cast<const group_normalization>(fused_ops[0].desc);
                    group_count = groupnorm0->num_groups;
                    group_size = k / group_count.value();
                    if (!bias)
                        post_op = PostOp::GnReduce;
                    else
                        post_op = PostOp::BiasGnReduce;
                } else if (fused_ops[0].is_type<eltwise>()) {
                    auto eltwise0 = std::static_pointer_cast<const eltwise>(fused_ops[0].desc);
                    if (eltwise0->mode != eltwise_mode::sum)
                        return false;
                    if (!bias)
                        post_op = PostOp::Sum;
                    else
                        post_op = PostOp::BiasSum;
                } else {
                    return false;
                }
            } else if (fused_ops_are_one_of<eltwise>(fused_ops) && fused_ops.size() == 3) {
                auto eltwise0 = std::static_pointer_cast<const eltwise>(fused_ops[0].desc);
                auto eltwise1 = std::static_pointer_cast<const eltwise>(fused_ops[1].desc);
                auto eltwise2 = std::static_pointer_cast<const eltwise>(fused_ops[2].desc);
                if (eltwise0->mode != eltwise_mode::sum || eltwise1->mode != eltwise_mode::prod || eltwise2->mode != eltwise_mode::sum)
                    return false;
                if (!bias)
                    post_op = PostOp::SumMulSum;
                else
                    post_op = PostOp::BiasSumMulSum;
            } else {
                return false;
            }
            return true;
        }

        static std::optional<ConvDesc> from_node(const program_node& node) {
            ConvDesc desc;
            const auto& conv_node = node.as<convolution>();
            const auto conv_prim = conv_node.get_primitive();

            auto in_layouts = node.get_input_layouts();
            auto out_layouts = node.get_output_layouts();
            desc.n = in_layouts[0].get_shape()[0];
            desc.ih = in_layouts[0].get_shape()[2];
            desc.iw = in_layouts[0].get_shape()[3];
            desc.c = in_layouts[0].get_shape()[1];
            desc.k = in_layouts[1].get_shape()[0];
            desc.oh = out_layouts[0].get_shape()[2];
            desc.ow = out_layouts[0].get_shape()[3];

            // should check kernel h/w
            desc.kernel_size = in_layouts[1].get_shape()[2];
            // should check all strides
            desc.stride = conv_prim->stride[0];
            desc.dilation = conv_prim->dilation[0];
            // should check all paddings
            desc.padding = conv_prim->padding_begin[0];

            if (!desc.process_fused_ops(conv_node.get_fused_primitives(), conv_node.bias_term()))
                return std::nullopt;
            return desc;
        }

        static std::optional<ConvDesc> from_rt_params(const RuntimeParams& params) {
            ConvDesc desc;

            const auto conv_prim = reinterpret_cast<const convolution*>(params.desc.get());
            auto in_layouts = params.input_layouts;
            auto out_layouts = params.output_layouts;
            desc.n = in_layouts[0].get_shape()[0];
            desc.ih = in_layouts[0].get_shape()[2];
            desc.iw = in_layouts[0].get_shape()[3];
            desc.c = in_layouts[0].get_shape()[1];
            desc.k = in_layouts[1].get_shape()[0];
            desc.oh = out_layouts[0].get_shape()[2];
            desc.ow = out_layouts[0].get_shape()[3];

            // should check kernel h/w
            desc.kernel_size = in_layouts[1].get_shape()[2];
            // should check all strides
            desc.stride = conv_prim->stride[0];
            desc.dilation = conv_prim->dilation[0];
            // should check all paddings
            desc.padding = conv_prim->padding_begin[0];

            const auto& fused_ops = params.fused_desc;
            if (!desc.process_fused_ops(fused_ops, conv_prim->bias.is_valid()))
                return std::nullopt;
            return desc;
        }

        // N x H x W x C x K x KERNEL_SIZE x STRIDE x DILATION x PADDING
        std::string get_shape_key() const {
            std::stringstream stream;
            stream << n;
            for (auto& e : {ih, iw, c, k, kernel_size, stride, dilation, padding}) {
                stream << "x" << e;
            }
            return stream.str();
        }

        bool has_fused_groupnorm() const {
            return post_op == PostOp::GnReduce || post_op == PostOp::BiasGnReduce;
        }
    };

    struct KernelKnobs {
        size_t wg_tile_n, wg_tile_p, wg_tile_q, wg_tile_k;
        size_t sg_tile_n, sg_tile_p, sg_tile_q, sg_tile_k;
        size_t global_slicing, local_slicing, prefetch_distance, accum_step;
    };

    struct NormKnobs {
        size_t wg_tile_n, wg_tile_w, wg_tile_c;
        size_t sg_tile_n, sg_tile_w, sg_tile_c;
    };

    static const std::unordered_map<std::string, KernelKnobs> ConvMap;
    static const std::unordered_map<std::string, NormKnobs> NormMap;
};

}  // namespace ov::intel_gpu::cm
