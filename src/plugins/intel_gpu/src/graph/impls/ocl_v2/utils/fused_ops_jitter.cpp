// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_ops_jitter.hpp"

#include "activation_inst.h"
#include "common_utils/dispatch_utils.hpp"
#include "eltwise_inst.h"
#include "intel_gpu/graph/fused_primitive_desc.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "jitter.hpp"
#include "openvino/core/type/element_type.hpp"
#include "quantize_inst.h"

using namespace cldnn;
namespace ov::intel_gpu::ocl {
namespace {

inline JitTerm zero(ov::element::Type_t type = ov::element::Type_t::i32) {
    if (type == ov::element::Type_t::f16) {
        return JitTerm{"0.0h"};
    }
    if (type == ov::element::Type_t::f32) {
        return JitTerm{"0.0f"};
    }
    return JitTerm{"0"};
}
inline JitTerm one(ov::element::Type_t type = ov::element::Type_t::i32) {
    if (type == ov::element::Type_t::f16) {
        return JitTerm{"1.0h"};
    }
    if (type == ov::element::Type_t::f32) {
        return JitTerm{"1.0f"};
    }
    return JitTerm{"1"};
}

inline const JitTerm indent{"\\\n\t"};

template <typename... Args>
JitTerm make_statement(Args&&... args) {
    return concat(indent, std::forward<Args>(args)..., ";");
}

JitConstants make_fused_ops_decls_jit_constants(const RuntimeParams& params, const std::vector<FusedOpsConfiguration>& conf) {
    JitConstants jit = {};

    if (conf.empty()) {
        return jit;
    }

    std::string input_decls;
    std::string input_args;

    const auto fused_ops_descs = params.fused_desc;

    for (size_t i = 0; i < fused_ops_descs.size(); i++) {
        auto fused_dep_codegen = FusedOpsCodeGenerator(fused_ops_descs[i], params, i);

        jit.add(fused_dep_codegen.make_fused_tensor_jit_constants(conf[0]));
        jit.add(fused_dep_codegen.make_input_decls_jit_constants(conf[0]));
        if (!fused_ops_descs[i].deps.empty()) {
            std::string optional_comma = (!input_decls.empty() ? "," : "");
            input_decls += optional_comma + "\\\n\tFUSED_OP" + to_code_string(i) + "_DECLS";
            input_args += optional_comma + "\\\n\tFUSED_OP" + to_code_string(i) + "_ARGS";
        }
    }

    jit.make("FUSED_OPS_DECLS", input_decls);
    jit.make("FUSED_OPS_ARGS", input_args);
    jit.make("HAS_FUSED_OPS", true);
    jit.make("HAS_FUSED_OPS_DECLS", !input_decls.empty());

    return jit;
}
}  // namespace

JitConstants make_fused_ops_jit_constants(const RuntimeParams& params, const std::vector<FusedOpsConfiguration>& conf) {
    JitConstants jit = {};
    OPENVINO_ASSERT(params.output_layouts.size() == 1, "[GPU] Multi-output primitives are not supported yet by fused ops codegen");

    if (conf.empty()) {
        return jit;
    }

    const auto& fused_ops_descs = params.fused_desc;
    if (std::all_of(fused_ops_descs.cbegin(), fused_ops_descs.cend(), [](const fused_primitive_desc& desc) {
            return desc.is_type<reorder>();
        })) {
        return jit;
    }

    try {
        for (const auto& c : conf) {
            std::string fused_ops;
            std::string fused_ops_preload;
            std::string fused_ops_calc;
            JitTerm in_name{c.input_var_name};
            JitTerm out_name;
            ov::element::Type in_type = c.input_dt;
            bool can_all_use_preload = true;

            for (size_t i = 0; i < fused_ops_descs.size(); i++) {
                // Reorder is not processed by jitter
                if (fused_ops_descs[i].is_type<reorder>()) {
                    continue;
                }

                auto fused_dep_codegen = FusedOpsCodeGenerator(fused_ops_descs[i], params, i);
                jit.add(fused_dep_codegen.make_load_jit_constants(c, params.output_layouts[0]));
                jit.add(fused_dep_codegen.make_op_jit_constants(c, in_name, in_type, out_name));

                bool can_use_preload = fused_dep_codegen.can_preload_data(c);
                can_all_use_preload &= can_use_preload;
                bool can_preload_eltwise = true;
                if (fused_ops_descs[i].is_type<eltwise>() && c.load_type == FusedOpsConfiguration::LoadType::FEATURE_SHUFFLE) {
                    can_preload_eltwise = false;
                }
                fused_ops += concat(indent, "FUSED_OP", i, "_LOAD", c.suffix).str();
                fused_ops += concat(indent, "FUSED_OP", i, "_ACTION", c.suffix).str();
                if (can_use_preload && can_preload_eltwise) {
                    fused_ops_preload += concat(indent, "FUSED_OP", i, "_LOAD", c.suffix).str();
                }
                if (c.allow_for_partial_preload && (!can_use_preload || !can_preload_eltwise)) {
                    fused_ops_calc += concat(indent, "FUSED_OP", i, "_LOAD", c.suffix).str();
                }
                fused_ops_calc += concat(indent, "FUSED_OP", i, "_ACTION", c.suffix).str();
            }

            jit.make("FUSED_OPS" + c.suffix, fused_ops);
            jit.make("FUSED_OPS_PRELOAD" + c.suffix, fused_ops_preload);
            jit.make("FUSED_OPS_CALC" + c.suffix, fused_ops_calc);
            jit.make("FUSED_OPS_RESULT" + c.suffix, out_name);

            bool can_any_use_preload = !fused_ops_preload.empty();
            jit.make("FUSED_OPS_CAN_USE_PRELOAD" + c.suffix, can_all_use_preload || (c.allow_for_partial_preload && can_any_use_preload));
        }

        jit.add(make_fused_ops_decls_jit_constants(params, conf));
    } catch (std::exception& ex) {
        throw std::runtime_error("[GPU] Fused op code generation for node " + params.desc->id + " failed with error: " + ex.what());
    }

    return jit;
}

bool FusedOpsCodeGenerator::can_preload_data(const FusedOpsConfiguration& conf) const {
    if (conf.loop_axes.empty()) {
        return true;
    }

    bool can_preload = true;
    // Check that tensor offset doesn't have dependency from the loop dimensions
    for (const auto& d : conf.loop_axes) {
        for (const auto& in_d : desc.inputs) {
            if (in_d.m_type != FusedInputType::EXTERNAL) {
                continue;
            }
            auto idx = IndexDesc{conf.bfzyx_idx_order, params.get_input_layout(in_d.m_idx)};
            switch (d) {
            case ChannelName::BATCH:
                can_preload &= idx.b == "0";
                break;
            case ChannelName::FEATURE:
                can_preload &= idx.f == "0";
                break;
            case ChannelName::W:
                can_preload &= idx.w == "0";
                break;
            case ChannelName::Z:
                can_preload &= idx.z == "0";
                break;
            case ChannelName::Y:
                can_preload &= idx.y == "0";
                break;
            case ChannelName::X:
                can_preload &= idx.x == "0";
                break;
            default:
                return false;
            }
        }
    }

    return can_preload;
}

JitTerm FusedOpsCodeGenerator::get_op_type() const {
    if (desc.is_type<eltwise>()) {
        return JitTerm{"eltwise"};
    } else if (desc.is_type<quantize>()) {
        return JitTerm{"quantize"};
    } else if (desc.is_type<activation>()) {
        return JitTerm{"activation"};
    }
    return {};
}

JitConstants FusedOpsCodeGenerator::make_fused_tensor_jit_constants(const FusedOpsConfiguration& /*conf*/) const {
    JitConstants jit{};
    for (const auto& in_d : desc.inputs) {
        if (in_d.m_type != FusedInputType::EXTERNAL) {
            continue;
        }

        const auto in_idx = in_d.m_idx;
        std::string name = get_input_tensor_name(in_idx).str();
        jit.add(make_layout_jit_constants(name, params.get_input_layout(in_idx), params.in_port_to_shape_info_offset.at(in_idx)));
    }
    // Use shape_ids from output tensor as won't support fused ops which changes out shape for now
    jit.add(make_layout_jit_constants(get_output_tensor_name().str(), desc.output_layout, params.out_port_to_shape_info_offset.at(0)));
    return jit;
}

JitConstants FusedOpsCodeGenerator::make_input_decls_jit_constants(const FusedOpsConfiguration& /*conf*/) const {
    JitConstants jit = {};

    std::string input_decls;
    std::string input_args;
    for (const auto& in_d : desc.inputs) {
        if (in_d.m_type != FusedInputType::EXTERNAL) {
            continue;
        }
        auto ptr_name = get_input_ptr_name(in_d.m_idx);
        if (!input_decls.empty()) {
            input_decls += ",";
        }

        if (!input_args.empty()) {
            input_args += ",";
        }

        input_decls += concat(indent, make_const_global_ptr(params.get_input_layout(in_d.m_idx).data_type, ptr_name)).str();
        input_args += concat(indent, ptr_name).str();
    }

    jit.make("FUSED_OP" + to_code_string(op_idx) + "_DECLS", input_decls);
    jit.make("FUSED_OP" + to_code_string(op_idx) + "_ARGS", input_args);
    return jit;
}

JitConstants FusedOpsCodeGenerator::make_load_jit_constants(const FusedOpsConfiguration& conf, const cldnn::layout& prim_output) const {
    JitConstants jit = {};

    auto idx = conf.bfzyx_idx_order;
    auto fused_op_config = conf;

    std::string load_decls;
    static thread_local int i = 0;
    // TODO: check if there is a use case for index reuse or it can be removed
    bool reuse_index = false;
    bool safe_load = conf.boundary_check == FusedOpsConfiguration::BoundaryCheck::ENABLED;
    JitTerm reused_idx = concat("reused_idx_", i++);
    if (reuse_index) {
        load_decls += make_statement(reused_idx.assign(get_idx(0, IndexDesc{idx, params.get_input_layout(0)}, safe_load))).str();
    }
    // TODO: add some generic way to support shuffled feature, lets say possibility to add separate config for each fused op
    if (desc.is_type<eltwise>() && conf.load_type == FusedOpsConfiguration::LoadType::FEATURE_SHUFFLE) {
        std::string sub_group_local_id_str = "get_sub_group_local_id()";
        size_t found_sub = conf.bfzyx_idx_order[1].rfind(sub_group_local_id_str);
        if (found_sub != std::string::npos)
            fused_op_config.bfzyx_idx_order[1].replace(found_sub, sub_group_local_id_str.length(), fused_op_config.shuffle_var_name);
    }

    for (auto op_input_id : get_required_inputs()) {
        auto in_type = make_type(concat(get_input_tensor_name(op_input_id), "_TYPE"), conf.vec_size);
        load_decls +=
            make_statement(
                declare_var(in_type, get_input_var_name(op_input_id), get_jit_load(fused_op_config, op_input_id, prim_output, reuse_index, reused_idx)))
                .str();
    }

    jit.make("FUSED_OP" + to_code_string(op_idx) + "_LOAD" + conf.suffix, load_decls);

    return jit;
}

JitConstants FusedOpsCodeGenerator::make_op_jit_constants(const FusedOpsConfiguration& conf,
                                                          const JitTerm& in_var,
                                                          ov::element::Type_t in_type,
                                                          JitTerm& out_var) const {
    JitConstants jit = {};

    std::string op_decls;
    auto vec_size = conf.vec_size;
    JitTerm shuffle_var{conf.shuffle_var_name};
    bool is_shuffled = false;
    bool floor_integer_div = false;

    const auto& dep_data = desc.inputs;
    bool is_original_input = dep_data[0].m_type == FusedInputType::ORIGINAL;
    size_t in_idx = dep_data[0].m_idx;

    std::vector<JitTerm> input_vars;

    out_var = get_output_var_name(in_var, op_idx);
    const auto& out_type = desc.output_layout.data_type;

    if (conf.load_type == FusedOpsConfiguration::LoadType::FEATURE_SHUFFLE && desc.is_type<quantize>()) {
        is_shuffled = true;
    }

    std::vector<JitTerm> in_vars_converted;
    for (const auto& in_d : dep_data) {
        auto in_name = get_input_var_name(in_d.m_idx, is_shuffled, shuffle_var);
        if (params.input_layouts[in_d.m_idx].data_type != out_type) {
            in_name = convert_to_output_type(in_name, vec_size);
        }
        in_vars_converted.emplace_back(in_name);
    }

    if (desc.is_type<eltwise>()) {
        auto p = desc.get_typed_fuse_params<EltwiseFuseParams>();

        if (p->_desc->mode == eltwise_mode::div) {
            if (p->_desc->m_pythondiv) {
                floor_integer_div = true;
            }
        }
    }

    auto get_acc_t = [&]() -> ov::element::Type {
        std::vector<ov::element::Type> input_types = {desc.output_layout.data_type};
        for (const auto& dep : dep_data) {
            input_types.emplace_back(params.input_layouts[dep.m_idx].data_type);
        }

        std::vector<ov::element::Type> types_prioritized = {};
        if (floor_integer_div) {
            if (std::all_of(input_types.begin(), input_types.end(), [=](const ov::element::Type& t) -> bool {
                    return (t != ov::element::f32 && t != ov::element::f16);
                })) {
                types_prioritized =
                    {ov::element::i64, ov::element::i32, ov::element::u32, ov::element::i16, ov::element::u16, ov::element::i8, ov::element::u8};
                for (auto& type : types_prioritized) {
                    if (std::any_of(input_types.begin(), input_types.end(), [=](const ov::element::Type& t) -> bool {
                            return (t == type);
                        })) {
                        return type;
                    }
                }
            }
        }

        floor_integer_div = false;
        types_prioritized.clear();
        types_prioritized = {ov::element::f32, ov::element::f16};
        for (auto& type : types_prioritized) {
            if (std::any_of(input_types.begin(), input_types.end(), [=](const ov::element::Type& t) -> bool {
                    return t == type;
                })) {
                return type;
            }
        }

        return ov::element::f32;
    };

    auto get_input = [&](size_t index) -> JitTerm {
        const auto dep = dep_data[index];
        auto input_name = (dep.m_type == FusedInputType::ORIGINAL)   ? in_var
                          : (dep.m_type == FusedInputType::INTERNAL) ? get_output_var_name(in_var, dep.m_idx)
                                                                     : get_input_var_name(dep.m_idx, is_shuffled, shuffle_var);
        auto input_type = (dep.m_type == FusedInputType::ORIGINAL) ? in_type : params.get_input_layout(dep.m_idx).data_type;
        auto acc_t = get_acc_t();

        if (input_type != acc_t) {
            return convert_to_type(input_name, acc_t, vec_size);
        }
        return input_name;
    };

    input_vars.reserve(dep_data.size());
    for (size_t i = 0; i < dep_data.size(); i++) {
        input_vars.emplace_back(get_input(i));
    }

    if (desc.is_type<eltwise>()) {
        auto p = desc.get_typed_fuse_params<EltwiseFuseParams>();
        JitTerm op;
        switch (p->_desc->mode) {
        case eltwise_mode::sum:
            op = input_vars[0] + input_vars[1];
            break;
        case eltwise_mode::prod:
            op = input_vars[0] * input_vars[1];
            break;
        case eltwise_mode::sub:
            op = input_vars[0] - input_vars[1];
            break;
        case eltwise_mode::div:
            op = input_vars[0] / input_vars[1];
            break;
        default:
            OPENVINO_THROW("[GPU] Eltwise mode is not supported in fused ops codegen");
        }

        auto tmp_var = concat(out_var, "_tmp");
        auto acc_t_type = make_type(get_acc_t(), vec_size);
        op_decls += make_statement(declare_var(acc_t_type, tmp_var, op)).str();
        if (floor_integer_div) {
            auto tmp_var_rem = concat(tmp_var, "_rem");
            op_decls += make_statement(declare_var(acc_t_type, tmp_var_rem, input_vars[0] % input_vars[1])).str();
            auto in0_is_neg = input_vars[0].lt(zero(dep_data[0].m_element_type));
            auto in1_is_neg = input_vars[1].lt(zero(dep_data[1].m_element_type));
            auto expr = ternary(logical_and(tmp_var_rem.ne(zero(get_acc_t())), in0_is_neg.ne(in1_is_neg)), one(get_acc_t()), zero(get_acc_t()));
            op_decls += make_statement(tmp_var -= expr).str();
        }
        op_decls += concat(indent, get_output_type(vec_size), " ", out_var, " = ", convert_to_output_type(tmp_var, vec_size), ";").str();
    } else if (desc.is_type<quantize>()) {
        auto p = desc.get_typed_fuse_params<QuantizeFuseParams>();

        JitTerm in_converted = is_original_input ? in_var : get_output_var_name(in_var, in_idx);
        ov::element::Type input_type = is_original_input ? in_type : dep_data[0].m_element_type;
        ov::element::Type tmp_et = ov::element::f32;
        JitTerm tmp_type = make_type(tmp_et, vec_size);
        JitTerm tmp_var = concat(out_var, "_tmp");

        if (input_type != tmp_et) {
            in_converted = convert_to_type(in_converted, tmp_et, vec_size);
        }

        const auto out_scale_idx = dep_data[p->out_scale_idx].m_idx;
        const auto out_shift_idx = dep_data[p->out_shift_idx].m_idx;
        const auto in_scale_idx = dep_data[p->in_scale_idx].m_idx;
        const auto in_shift_idx = dep_data[p->in_shift_idx].m_idx;
        const auto in_range_lo_idx = dep_data[p->in_range_lo_idx].m_idx;
        const auto in_range_hi_idx = dep_data[p->in_range_hi_idx].m_idx;

        auto post_scale = p->_per_tensor_output_scale ? broadcast(JitTerm{p->_out_scale}, tmp_et, vec_size)
                                                      : convert_to_type(get_input_var_name(out_scale_idx, is_shuffled, shuffle_var), tmp_et, vec_size);
        auto post_shift = p->_per_tensor_output_shift ? broadcast(JitTerm{p->_out_shift}, tmp_et, vec_size)
                                                      : convert_to_type(get_input_var_name(out_shift_idx, is_shuffled, shuffle_var), tmp_et, vec_size);
        auto pre_scale = p->_per_tensor_input_scale ? broadcast(JitTerm{p->_in_scale}, tmp_et, vec_size)
                                                    : convert_to_type(get_input_var_name(in_scale_idx, is_shuffled, shuffle_var), tmp_et, vec_size);
        auto pre_shift = p->_per_tensor_input_shift ? broadcast(JitTerm{p->_in_shift}, tmp_et, vec_size)
                                                    : convert_to_type(get_input_var_name(in_shift_idx, is_shuffled, shuffle_var), tmp_et, vec_size);

        if (p->_per_tensor_output_range && p->_out_lo < p->_out_hi) {
            // Input scale
            op_decls += make_statement(declare_var(tmp_type, tmp_var, in_converted * pre_scale)).str();

            // Input shift
            if (p->_need_pre_shift) {
                op_decls += make_statement(tmp_var.assign(tmp_var + pre_shift)).str();
            }

            // Round operation isn't needed if output type is int8/uint8 and scale coefficient in all output channels is equal to 1.0
            bool output_type_is_int8 = desc.output_layout.data_type == ov::element::u8 || desc.output_layout.data_type == ov::element::i8;
            if (((p->_need_post_scale || p->_need_post_shift) && output_type_is_int8) || !output_type_is_int8) {
                op_decls += make_statement(tmp_var.assign(round(tmp_var))).str();
            }

            // Output scale
            if (p->_need_post_scale) {
                op_decls += make_statement(tmp_var.assign(tmp_var * post_scale)).str();
            }

            // Output shift
            if (p->_need_post_shift) {
                op_decls += make_statement(tmp_var.assign(tmp_var + post_shift)).str();
            }

            // Output range
            auto out_lo = broadcast(JitTerm{p->_out_lo}, tmp_et, vec_size);
            auto out_hi = broadcast(JitTerm{p->_out_hi}, tmp_et, vec_size);

            // Output clamp
            if (p->_need_clamp) {
                if (p->_need_min_clamp && p->_need_max_clamp) {
                    op_decls += make_statement(tmp_var.assign(clamp(tmp_var, out_lo, out_hi))).str();
                } else if (p->_need_min_clamp) {
                    op_decls += make_statement(tmp_var.assign(max(tmp_var, out_lo))).str();
                } else {
                    op_decls += make_statement(tmp_var.assign(min(tmp_var, out_hi))).str();
                }
            }

            // Output conversion with rounding and saturation
            op_decls += make_statement(get_output_type(vec_size), " ", out_var.assign(convert_to_output_type_sat(tmp_var, vec_size))).str();
        } else {
            // Input range
            auto in_lo = p->_per_tensor_input_range ? broadcast(JitTerm{p->_in_lo}, tmp_et, vec_size)
                                                    : convert_to_type(get_input_var_name(in_range_lo_idx, is_shuffled, shuffle_var), tmp_et, vec_size);
            auto in_hi = p->_per_tensor_input_range ? broadcast(JitTerm{p->_in_hi}, tmp_et, vec_size)
                                                    : convert_to_type(get_input_var_name(in_range_hi_idx, is_shuffled, shuffle_var), tmp_et, vec_size);

            // Input clamp
            if (p->_need_clamp) {
                if (p->_need_min_clamp && p->_need_max_clamp) {
                    op_decls += make_statement(declare_var(tmp_type, tmp_var, clamp(in_converted, in_lo, in_hi))).str();
                } else if (p->_need_min_clamp) {
                    op_decls += make_statement(declare_var(tmp_type, tmp_var, max(in_converted, in_lo))).str();
                } else {
                    op_decls += make_statement(declare_var(tmp_type, tmp_var, min(in_converted, in_hi))).str();
                }
            } else {
                op_decls += make_statement(declare_var(tmp_type, tmp_var, in_converted)).str();
            }

            // Input scale
            op_decls += make_statement(tmp_var.assign(tmp_var * pre_scale)).str();

            // Input shift
            if (p->_need_pre_shift) {
                op_decls += make_statement(tmp_var.assign(tmp_var + pre_scale)).str();
            }

            // Round operation isn't needed if output type is int8/uint8 and scale coefficient in all output channels is equal to 1.0
            bool output_type_is_int8 = desc.output_layout.data_type == ov::element::u8 || desc.output_layout.data_type == ov::element::i8;
            if (((p->_need_post_scale || p->_need_post_shift) && output_type_is_int8) || !output_type_is_int8) {
                op_decls += make_statement(tmp_var.assign(round(tmp_var))).str();
            }

            // Output scale
            if (p->_need_post_scale) {
                op_decls += make_statement(tmp_var.assign(tmp_var * post_scale)).str();
            }

            // Output shift
            if (p->_need_post_shift) {
                op_decls += make_statement(tmp_var.assign(tmp_var + post_shift)).str();
            }

            // Output conversion with rounding and saturation
            op_decls += make_statement(declare_var(get_output_type(vec_size), out_var, convert_to_output_type_sat(tmp_var, vec_size))).str();
        }
    } else if (desc.is_type<activation>()) {
        auto p = desc.get_typed_fuse_params<ActivationFuseParams>();
        const auto& activation_f = p->_desc->activation_function;
        const auto& activation_p = p->_desc->additional_params;
        JitTerm new_in_var = is_original_input ? in_var : get_output_var_name(in_var, in_idx);
        op_decls += make_statement(declare_var(get_output_type(vec_size), out_var, convert_to_output_type(new_in_var, vec_size))).str();
        if (activation_f != cldnn::activation_func::none) {
            auto suffix = "_FUSED_OP" + to_code_string(op_idx) + conf.suffix;
            JitTerm nl_m{activation_p.a};
            JitTerm nl_n{activation_p.b};

            if (activation_f == cldnn::activation_func::none) {
                if (out_type == ov::element::i8) {
                    nl_m = JitTerm{std::max<float>(activation_p.a, std::numeric_limits<int8_t>::min())};
                    nl_n = JitTerm{std::min<float>(activation_p.b, std::numeric_limits<int8_t>::max())};
                } else if (out_type == ov::element::u8) {
                    nl_m = JitTerm{std::max(activation_p.a, 0.0f)};
                    nl_n = JitTerm{std::min<float>(activation_p.b, std::numeric_limits<uint8_t>::max())};
                }
            }

            if (dep_data.size() == 2) {
                if (dep_data[1].m_element_type != out_type) {
                    nl_m = convert_to_output_type(get_input_var_name(0), vec_size);
                } else {
                    nl_m = get_input_var_name(0);
                }
            } else {
                nl_m = broadcast(nl_m, out_type, vec_size);
            }

            nl_n = broadcast(nl_n, out_type, vec_size);

            // Disable type casts in activation, since current jit generator for activation don't respect vector size of parameters.
            // So conversion is explicitly done in params declaration
            jit.add(make_activation_jit_constants(suffix, activation_f, out_type, out_type));
            JitTerm activation = concat("ACTIVATION_FUNC", suffix);
            op_decls += make_statement(out_var.assign(activation(out_var, nl_m, nl_n))).str();
        }
    }

    jit.make("FUSED_OP" + to_code_string(op_idx) + "_ACTION" + conf.suffix, op_decls);

    return jit;
}

JitTerm FusedOpsCodeGenerator::get_input_tensor_name(size_t input_id) const {
    return concat("FUSED_OP_", op_idx, "_INPUT", input_id);
}

JitTerm FusedOpsCodeGenerator::get_output_tensor_name() const {
    return concat("FUSED_OP_", op_idx, "_OUTPUT");
}

JitTerm FusedOpsCodeGenerator::get_idx(size_t input_id, const IndexDesc& idx, bool should_be_safe) const {
    const auto rank = params.get_input_layout(input_id).get_rank();

    JitTerm index_func;
    if (should_be_safe) {
        index_func = concat(get_input_tensor_name(input_id), "_GET_INDEX_SAFE");
    } else {
        index_func = concat(get_input_tensor_name(input_id), "_GET_INDEX");
    }

    if (rank <= 4) {
        return index_func(idx.b, idx.f, idx.y, idx.x);
    }
    if (rank == 5) {
        return index_func(idx.b, idx.f, idx.z, idx.y, idx.x);
    }
    if (rank == 6) {
        return index_func(idx.b, idx.f, idx.w, idx.z, idx.y, idx.x);
    }
    if (rank == 7) {
        return index_func(idx.b, idx.f, idx.u, idx.w, idx.z, idx.y, idx.x);
    }
    if (rank == 8) {
        return index_func(idx.b, idx.f, idx.v, idx.u, idx.w, idx.z, idx.y, idx.x);
    }

    OPENVINO_THROW("[GPU] Unsupported tensor rank: ", rank, " in FusedOpsCodeGenerator");
}

JitTerm FusedOpsCodeGenerator::get_jit_load(const FusedOpsConfiguration& conf,
                                            size_t input_id,
                                            const cldnn::layout& prim_output,
                                            bool reuse_index,
                                            const JitTerm& reused_idx) const {
    const auto& input_tensor = params.get_input_layout(input_id);
    size_t vec_size = 1;
    auto input_dt = input_tensor.data_type;
    JitTerm in_ptr = get_input_ptr_name(input_id);
    JitTerm in_var = get_input_var_name(input_id);

    const auto in_f = extract_channel(ChannelName::FEATURE, input_tensor);
    const auto out_f = extract_channel(ChannelName::FEATURE, prim_output);

    bool valid_broadcast_case = input_tensor.count() == out_f || input_tensor.count() == 1;

    // Eltwise fused op can't have full tensor argument when requested vec_size > 1, since it might require
    // splitting load into several parts and some kind of index recalculation which is not supported
    format orig_output_format = conf.is_post_reorder_fused() ? conf.orig_output_layout : prim_output.format;

    if (desc.is_type<eltwise>() && !valid_broadcast_case && input_tensor.format != orig_output_format && conf.vec_size > 1) {
        throw std::runtime_error("[GPU] Mixed layouts of input tensors are not supported in fused eltwise:"
                                 "\nfused_input: " +
                                 input_tensor.to_string() + "\noutput: " + prim_output.to_string());
    }

    if (conf.vec_axis != ChannelName::UNKNOWN && extract_channel(conf.vec_axis, input_tensor) != 1) {
        vec_size = conf.vec_size;
    }

    auto idx = conf.bfzyx_idx_order;
    if (vec_size == 0 || vec_size > 8) {
        throw std::invalid_argument("[GPU] Invalid vector size in jit definitions: " + to_code_string(vec_size));
    }

    bool safe_load = conf.boundary_check == FusedOpsConfiguration::BoundaryCheck::ENABLED;

    // Fsv16 Eltwise whcih requires f axis broadcast such as input[1,1,z,1,1], output[b,f,z,y,x] need to use LT unligned read.
    // In this case, intel_sub_group_block_read() introduces increasing index in feature block.
    bool f_axis_broadcast = (in_f != out_f) && (in_f == 1);
    // Change JitLoad to ignore LT_ALIGNED_READ LoadType if this input tensor has a planar format(SimpleLayout)
    if (desc.is_type<eltwise>() && conf.load_type == FusedOpsConfiguration::LoadType::LT_ALIGNED_READ &&
        ((format::is_simple_data_format(input_tensor.format) && input_tensor.format != orig_output_format) || f_axis_broadcast) &&
        (!format::is_simple_data_format(input_tensor.format) && (input_tensor.get_partial_shape() == prim_output.get_partial_shape() || f_axis_broadcast)) &&
        input_tensor.count() != 1) {
        std::string sub_group_local_id_str = "get_sub_group_local_id";
        size_t found_sub = conf.bfzyx_idx_order[1].rfind(sub_group_local_id_str);
        OPENVINO_ASSERT(found_sub == std::string::npos, "[GPU] LT_ALIGNED_READ LoadType is used with get_sub_group_local_id.");

        auto new_idx_order = conf.bfzyx_idx_order;
        new_idx_order[1] = "(" + conf.bfzyx_idx_order[1] + " + " + sub_group_local_id_str + "()" + ")";
        if (vec_size > 1) {
            auto vec_axis_idx = conf.get_dim_index_from_order(conf.vec_axis);
            OPENVINO_ASSERT(vec_axis_idx != -1, "[GPU] Incorrect vec_axis value ", static_cast<int>(conf.vec_axis), " for bfzyx_idx_order order");
            new_idx_order[vec_axis_idx] = "((" + conf.bfzyx_idx_order[vec_axis_idx] + ") + loop_var)";
        }
        JitTerm new_index_func_call{get_idx(input_id, IndexDesc{new_idx_order, input_tensor}, safe_load)};

        if (vec_size > 1) {
            JitTerm loop_var{"loop_var"};
            JitTerm loop_var_type{"uint"};
            return for_loop(declare_var(loop_var_type, loop_var, zero()), loop_var.lt(JitTerm{vec_size}), loop_var++)
                .body(make_statement(in_var[loop_var].assign(in_ptr[new_index_func_call])));
        }

        return in_ptr[new_index_func_call];
    }

    JitTerm index_func_call_vec{reuse_index ? reused_idx : get_idx(input_id, IndexDesc{idx, input_tensor}, safe_load)};
    JitTerm index_func_call{reuse_index ? reused_idx : get_idx(input_id, IndexDesc{idx, input_tensor}, safe_load)};
    if (conf.index_type == FusedOpsConfiguration::IndexType::LINEAR_OFFSET) {
        JitTerm offset{conf.bfzyx_idx_order[0]};
        if (safe_load) {
            offset = offset % JitTerm{to_code_string(input_tensor.count())};
        }

        if (vec_size > 1) {
            return global_ptr_cast(input_dt, vec_size, in_ptr + offset)[zero()];
        }
        return in_ptr[offset];
    }
    // TODO: Need to add smarter vectors handling:
    // 1. Boundary checks for safe load
    // 2. If in given configuration data can't be loaded by a simple UNIT_BLOCK_READx call or load from casted ptr,
    //    we can gather the data to vector
    if (conf.load_type == FusedOpsConfiguration::LoadType::LT_ALIGNED_READ) {
        bool multiple_elements = false;
        // For dynamic shape input tensor, check any one of static dimension has more than one element.
        if (input_tensor.is_dynamic()) {
            for (const auto& dim : input_tensor.get_partial_shape()) {
                if (dim.is_static() && dim.get_length() > 1) {
                    multiple_elements = true;
                    break;
                }
            }
        }

        if (input_tensor.count() > 1 || multiple_elements) {
            // Currently we assume that in such scenario we can safely load sub_group_size elements from the pointer
            return make_block_read(input_dt, vec_size, in_ptr + index_func_call);
        }

        // Input has only one element, so broadcast it for the whole vector size
        return broadcast(in_ptr[index_func_call], input_dt, vec_size);
    }

    if (vec_size > 1) {
        return global_ptr_cast(input_dt, vec_size, in_ptr + index_func_call_vec)[zero()];
    }

    return in_ptr[index_func_call];
}

JitTerm FusedOpsCodeGenerator::get_input_ptr_name(size_t input_id) const {
    return concat(get_op_type(), op_idx, "_input", input_id);
}

JitTerm FusedOpsCodeGenerator::get_input_var_name(size_t input_id, bool is_shuffled, const JitTerm& shuffle_var) const {
    if (is_shuffled) {
        JitTerm shuffle_f{"_sub_group_shuffle"};
        return shuffle_f(concat(get_op_type(), op_idx, "_data", input_id), shuffle_var);
    }
    return concat(get_op_type(), op_idx, "_data", input_id);
}

JitTerm FusedOpsCodeGenerator::get_output_var_name(const JitTerm& input_var, size_t op_id) const {
    auto copy = input_var.str();
    std::replace_if(
        copy.begin(),
        copy.end(),
        [](char& c) {
            return c == '[' || c == ']' || c == ' ' || c == '.';
        },
        '_');

    return concat(copy, "_out_", op_id);
}

JitTerm FusedOpsCodeGenerator::get_output_type(size_t vec_size) const {
    return make_type(desc.output_layout.data_type, vec_size);
}

JitTerm FusedOpsCodeGenerator::convert_to_output_type(const JitTerm& var, size_t vec_size) const {
    return convert_to_type(var, desc.output_layout.data_type, vec_size);
}

JitTerm FusedOpsCodeGenerator::convert_to_output_type_sat(const JitTerm& var, size_t vec_size) const {
    if (desc.output_layout.data_type == ov::element::f32 || desc.output_layout.data_type == ov::element::f16) {
        return convert_to_type(var, desc.output_layout.data_type, vec_size);
    }

    return concat("convert_", get_output_type(vec_size), "_sat_rte")(var);
}

std::vector<size_t> FusedOpsCodeGenerator::get_required_inputs() const {
    if (desc.is_type<quantize>()) {
        if (auto p = std::dynamic_pointer_cast<QuantizeFuseParams>(desc.f_param)) {
            std::vector<size_t> res = {};

            if (!p->_need_out_range && p->_need_clamp) {
                res.push_back(desc.inputs[p->in_range_lo_idx].m_idx);
                res.push_back(desc.inputs[p->in_range_hi_idx].m_idx);
            }
            if (!p->_per_tensor_input_scale) {
                res.push_back(desc.inputs[p->in_scale_idx].m_idx);
            }
            if (p->_need_pre_shift && !p->_per_tensor_input_shift) {
                res.push_back(desc.inputs[p->in_shift_idx].m_idx);
            }
            if (p->_need_post_scale && !p->_per_tensor_output_scale) {
                res.push_back(desc.inputs[p->out_scale_idx].m_idx);
            }
            if (p->_need_post_shift && !p->_per_tensor_output_shift) {
                res.push_back(desc.inputs[p->out_shift_idx].m_idx);
            }

            return res;
        }
    }

    std::vector<size_t> res;
    for (const auto& in_d : desc.inputs) {
        if (in_d.m_type == FusedInputType::EXTERNAL) {
            res.push_back(in_d.m_idx);
        }
    }
    return res;
}

JitConstants make_activation_jit_constants(const std::string& suffix,
                                           activation_func activation_function,
                                           ov::element::Type_t calc_dt,
                                           ov::element::Type_t out_dt) {
    const JitTerm activation_f = concat("ACTIVATION", suffix);
    const JitTerm activation_impl_f = concat("ACTIVATION_FUNC", suffix);
    const JitTerm cat_f{"CAT"};
    JitConstants jit = {};
    JitTerm type_suffix{out_dt == ov::element::f32 ? "f" : "h"};
    assert(activation_function != activation_func::none);

    jit.add(make_type_jit_constants(activation_impl_f.str(), calc_dt));
    if (out_dt != calc_dt) {
        jit.add(make_type_jit_constants(activation_impl_f.str() + "_OUT", out_dt));
    }

    const JitTerm one = concat("1.0", type_suffix);
    const JitTerm zero = concat("0.0", type_suffix);
    const JitTerm input{"input"};

    JitTerm macro_def = activation_impl_f(input, "m"_jit, "n"_jit);

    jit.add(make_jit_constant("ACTIVATION_PARAMS" + suffix, "NL_M" + suffix + ", NL_N" + suffix));

    switch (activation_function) {
    case activation_func::logistic:
        jit.add(make_jit_constant(macro_def, one / (one + exp(neg(input)))));
        break;
    case activation_func::hyperbolic_tan:
        jit.add(make_jit_constant(macro_def, tanh(input)));
        break;
    case activation_func::relu:
        jit.add(make_jit_constant(macro_def, max(zero, input)));
        break;
    case activation_func::relu_negative_slope: {
        const JitTerm slope = convert_to_type("m"_jit, calc_dt);
        jit.add(make_jit_constant(macro_def, ternary(isinf(slope), ternary(input.ge(zero), input, neg(slope)), max(input, zero) + (slope * min(input, zero)))));
        break;
    }
    case activation_func::elu: {
        const JitTerm alpha = convert_to_type("m"_jit, calc_dt);
        jit.add(make_jit_constant(macro_def, max(input, zero) + (alpha * (exp(min(input, zero)) - one))));
        break;
    }
    case activation_func::clamp: {
        const JitTerm m = convert_to_type("m"_jit, calc_dt);
        const JitTerm n = convert_to_type("n"_jit, calc_dt);
        jit.add(make_jit_constant(macro_def, max(m, min(n, input))));
        break;
    }
    case activation_func::softrelu:
        jit.add(make_jit_constant(macro_def, log(one + exp(input))));
        break;
    case activation_func::abs:
        if (out_dt == ov::element::f32 || out_dt == ov::element::f16) {
            jit.add(make_jit_constant(macro_def, fabs(input)));
        } else {
            jit.add(make_jit_constant(macro_def, abs(input)));
        }
        break;
    case activation_func::linear: {
        const JitTerm m = convert_to_type("m"_jit, calc_dt);
        const JitTerm n = convert_to_type("n"_jit, calc_dt);
        jit.add(make_jit_constant(macro_def, m * input + n));
        break;
    }
    case activation_func::square:
        jit.add(make_jit_constant(macro_def, input * input));
        break;
    case activation_func::sqrt:
        jit.add(make_jit_constant(macro_def, sqrt(input)));
        break;
    case activation_func::sin:
        jit.add(make_jit_constant(macro_def, sin(input)));
        break;
    case activation_func::asin:
        jit.add(make_jit_constant(macro_def, asin(input)));
        break;
    case activation_func::sinh:
        jit.add(make_jit_constant(macro_def, sinh(input)));
        break;
    case activation_func::asinh:
        jit.add(make_jit_constant(macro_def, asinh(input)));
        break;
    case activation_func::cos:
        jit.add(make_jit_constant(macro_def, cos(input)));
        break;
    case activation_func::acos:
        jit.add(make_jit_constant(macro_def, acos(input)));
        break;
    case activation_func::cosh:
        jit.add(make_jit_constant(macro_def, cosh(input)));
        break;
    case activation_func::acosh:
        jit.add(make_jit_constant(macro_def, acosh(input)));
        break;
    case activation_func::log:
        jit.add(make_jit_constant(macro_def, log(input)));
        break;
    case activation_func::log2:
        jit.add(make_jit_constant(macro_def, log2(input)));
        break;
    case activation_func::exp:
        jit.add(make_jit_constant(macro_def, exp(input)));
        break;
    case activation_func::pow: {
        const JitTerm m = convert_to_type("m"_jit, calc_dt);
        jit.add(make_jit_constant(macro_def, pow(input, m)));
        break;
    }
    case activation_func::tan:
        jit.add(make_jit_constant(macro_def, tan(input)));
        break;
    case activation_func::atan:
        jit.add(make_jit_constant(macro_def, atan(input)));
        break;
    case activation_func::atanh:
        jit.add(make_jit_constant(macro_def, atanh(input)));
        break;
    case activation_func::floor:
        if (out_dt == ov::element::f32 || out_dt == ov::element::f16) {
            jit.add(make_jit_constant(macro_def, floor(input)));
        } else {
            jit.add(make_jit_constant(macro_def, input));
        }
        break;
    case activation_func::ceil:
        if (out_dt == ov::element::f32 || out_dt == ov::element::f16) {
            jit.add(make_jit_constant(macro_def, ceil(input)));
        } else {
            jit.add(make_jit_constant(macro_def, input));
        }
        break;
    case activation_func::negative:
        jit.add(make_jit_constant(macro_def, neg(input)));
        break;
    case activation_func::erf:
        jit.add(make_jit_constant(macro_def, erf(input)));
        break;
    case activation_func::hard_sigmoid: {
        const JitTerm alpha = convert_to_type("m"_jit, calc_dt);
        const JitTerm beta = convert_to_type("n"_jit, calc_dt);
        jit.add(make_jit_constant(macro_def, max(zero, min(one, alpha * input + beta))));
        break;
    }
    case activation_func::hsigmoid: {
        const JitTerm three = concat("3.0", type_suffix);
        const JitTerm six = concat("6.0", type_suffix);
        jit.add(make_jit_constant(macro_def, min(max(zero, input + three), six) / six));
        break;
    }
    case activation_func::sign:
        jit.add(make_jit_constant(macro_def, ternary(input.gt(zero), one, ternary(input.eq(zero), zero, neg(one)))));
        break;
    case activation_func::reciprocal:
        jit.add(make_jit_constant(macro_def, one / input));
        break;
    case activation_func::selu: {
        const JitTerm alpha = convert_to_type("m"_jit, calc_dt);
        const JitTerm gamma = convert_to_type("n"_jit, calc_dt);
        jit.add(make_jit_constant(macro_def, ternary(input.le(zero), gamma * (alpha * exp(input) - alpha), gamma * input)));
        break;
    }
    case activation_func::softplus: {
        jit.add(make_jit_constant(macro_def, log(exp(input) + one)));
        break;
    }
    case activation_func::softsign: {
        jit.add(make_jit_constant(macro_def, input / (one + abs(input))));
        break;
    }
    case activation_func::swish: {
        const JitTerm beta = convert_to_type("m"_jit, calc_dt);
        jit.add(make_jit_constant(macro_def, input / (one + exp(neg(beta * input)))));
        break;
    }
    case activation_func::hswish: {
        const JitTerm three = concat("3.0", type_suffix);
        const JitTerm six = concat("6.0", type_suffix);
        jit.add(make_jit_constant(macro_def, input * min(max(zero, input + three), six) / six));
        break;
    }
    case activation_func::mish: {
        auto bound = calc_dt == ov::element::f32 ? "9.9f"_jit : "4.75h"_jit;
        const JitTerm two = concat("2.0", type_suffix);
        const JitTerm n((exp(input) + two) * exp(input));
        const JitTerm common_mish_formula((input * n) / (n + two));

        jit.add(make_jit_constant(macro_def, ternary(input.ge(bound), input, common_mish_formula)));
        break;
    }
    case activation_func::gelu: {
        const JitTerm half = concat("0.5", type_suffix);
        const JitTerm mult = concat("0.7071067811865475", type_suffix);  // (1 / sqrt(2))
        jit.add(make_jit_constant(macro_def, half * input * (one + erf((input * mult)))));
        break;
    }
    case activation_func::gelu_tanh: {
        const JitTerm half = concat("0.5", type_suffix);
        const JitTerm mult = concat("0.044715", type_suffix);
        const JitTerm sqrt_2_over_pi = concat("0.79788458347320556640625", type_suffix);
        jit.add(make_jit_constant(macro_def, half * input * (one + tanh(sqrt_2_over_pi * input * (one + mult * input * input)))));
        break;
    }
    case activation_func::negation:
        jit.add(make_jit_constant(macro_def, ternary(input.eq(zero), one, zero)));  // the workaround for OpenCL's vector type result (!input)
        break;
    case activation_func::round_half_to_even:
        jit.add(make_jit_constant(macro_def, rint(input)));
        break;
    case activation_func::round_half_away_from_zero:
        jit.add(make_jit_constant(macro_def, round(input)));
        break;
    case activation_func::none:
    default:
        jit.add(make_jit_constant(macro_def, input));
        break;
    }

    jit.add(make_jit_constant(activation_f(input, "m"_jit, "n"_jit), activation_impl_f(input, "m"_jit, "n"_jit)));

    return jit;
}

}  // namespace ov::intel_gpu::ocl
