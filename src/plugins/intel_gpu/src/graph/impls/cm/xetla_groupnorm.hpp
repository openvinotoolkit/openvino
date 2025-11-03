// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <optional>
#include <sstream>
#include <unordered_map>

#include "group_normalization_inst.h"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::cm {

struct GroupnormImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::groupnorm")
    explicit GroupnormImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::cm, shape_type, std::move(vf)) {}

    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<group_normalization>());
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        in_fmts[0] = format::byxf;
        for (size_t idx = 1; idx < node.get_dependencies().size(); idx++) {
            in_fmts[idx] = format::bfyx;
        }
        for (size_t idx = 0; idx < node.get_outputs_count(); idx++) {
            out_fmts[idx] = format::byxf;
        }

        return {in_fmts, out_fmts};
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<group_normalization>());

        auto& engine = node.get_program().get_engine();
        const auto& config = node.get_program().get_config();
        // const auto& info = engine.get_device_info();

        if (!check_cm_jit_support(engine, config) || /*info.arch != gpu_arch::xe2 ||*/ !config.get_use_cm()) {
            return false;
        }
        auto desc = GroupnormDesc::from_node(node);
        if (!desc.has_value())
            return false;

        auto key = desc.value().get_shape_key();

        if (NormMap.find(key) == NormMap.end())
            return false;

        return true;
    }
    enum class PostOp : uint32_t{
        None = 0,
        SiLU = 1
    };
    struct NormKnobs {
        size_t wg_tile_n, wg_tile_w, wg_tile_c;
        size_t sg_tile_n, sg_tile_w, sg_tile_c;
    };
    struct GroupnormDesc {
        size_t n, w, c;
        size_t group_count, group_size;
        PostOp post_op;

        bool process_fused_ops(const std::vector<cldnn::fused_primitive_desc> &fused_ops) {
            if (fused_ops.size() == 0) {
                post_op = PostOp::None;
            } else if (fused_ops.size() == 1 && fused_ops[0].is_type<activation>()) {
                auto activation0 = std::static_pointer_cast<const activation>(fused_ops[0].desc);
                auto activation_function = activation0->activation_function;
                if (activation_function == activation_func::swish) {
                    post_op = PostOp::SiLU;
                } else {
                    return false;
                }
            } else {
                return false;
            }
            return true;
        }

        static std::optional<GroupnormDesc> from_node(const program_node& node) {
            GroupnormDesc desc;
            const auto& gn_node = node.as<group_normalization>();
            const auto gn_prim = gn_node.get_primitive();
            desc.group_count = gn_prim->num_groups;

            auto in_layouts = node.get_input_layouts();
            if (in_layouts.size() != 5)
                return std::nullopt; // Does not include sumx and sumxsq
            auto data_shape = in_layouts[0].get_shape();
            desc.n = data_shape[0];
            desc.c = data_shape[1];
            desc.w = 1;
            for (size_t i = 2; i < data_shape.size(); i++) {
                desc.w *= data_shape[i];
            }
            desc.group_size = desc.c / desc.group_count;

            if (!desc.process_fused_ops(gn_node.get_fused_primitives()))
                return std::nullopt;
            return desc;
        }

        static std::optional<GroupnormDesc> from_rt_params(const RuntimeParams& params) {
            GroupnormDesc desc;

            const auto gn_prim = reinterpret_cast<const group_normalization*>(params.desc.get());
            desc.group_count = gn_prim->num_groups;

            auto in_layouts = params.input_layouts;
            if (in_layouts.size() != 5)
                return std::nullopt; // Does not include sumx and sumxsq
            auto data_shape = in_layouts[0].get_shape();
            desc.n = data_shape[0];
            desc.c = data_shape[1];
            desc.w = 1;
            for (size_t i = 2; i < data_shape.size(); i++) {
                desc.w *= data_shape[i];
            }
            desc.group_size = desc.c / desc.group_count;

            if (!desc.process_fused_ops(params.fused_desc))
                return std::nullopt;
            return desc;
        }

        // N x W x C
        std::string get_shape_key() const {
            std::stringstream stream;
            stream << n;
            for (auto& e : {w, c}) {
                stream << "x" << e;
            }
            return stream.str();
        }
    };

    static const std::unordered_map<std::string, NormKnobs> NormMap;
};
}  // namespace ov::intel_gpu::cm
