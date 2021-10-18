// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "primitive_inst.h"
#include "cldnn/runtime/error_handler.hpp"
#include "cldnn/runtime/memory.hpp"
#include "to_string_utils.h"
#include "register.hpp"
#include "utils.hpp"

#include "quantize_inst.h"
#include "reorder_inst.h"

#include "reorder/reorder_weights_kernel_selector.h"
#include "reorder/reorder_kernel_base.h"

#include <vector>
#include <list>
#include <utility>

#include <oneapi/dnnl/dnnl.hpp>

namespace cldnn {
namespace onednn {

enum class onednn_post_op_type : uint32_t {
    eltwise_act,
    eltwise_clip,
    eltwise_linear,
    eltwise_round,
    binary_mul,
    binary_add,
    binary_max,
    binary_min,
    scale,
    sum,
    optimized,
    optimized_eltwise,
    optimized_sum
};

struct onednn_post_op_desc {
    onednn_post_op_type op_type;
    size_t mem_offset;
    size_t mem_dep;
};

// This map contains information about onednn post-ops types, memory buffer offsets and dependencies
//     key is cldnn::primitive_id,
//     value is an instance of struct onednn_post_op_desc containing info about post-ops related to the node defined by key:
//         op_type - onednn_post_op_type (enum),
//         mem_offset - index of memory buffer for current post-operation,
//         mem_dep - memory dependency for working with fused node
static std::unordered_map<cldnn::primitive_id, std::vector<onednn_post_op_desc>> onednn_fusing_map;

template <class PType, class DescType, class PrimDescType = dnnl::primitive_desc, class PrimType = dnnl::primitive>
struct typed_primitive_onednn_impl : public typed_primitive_impl<PType> {
    const typed_program_node<PType>& _outer;
    std::shared_ptr<DescType> _desc;
    std::shared_ptr<dnnl::primitive_attr> _attrs;
    PrimDescType _pd;
    PrimType _prim;
    std::unordered_map<uint32_t, std::unordered_map<int, dnnl::memory>> _args;

    typed_primitive_onednn_impl(const typed_program_node<PType>& arg,
                                std::shared_ptr<DescType> desc,
                                std::shared_ptr<dnnl::primitive_attr> attrs,
                                const PrimDescType& pd,
                                kernel_selector::WeightsReorderParams weights_reorder = {})
        : typed_primitive_impl<PType>(weights_reorder, pd.impl_info_str()),
          _outer(arg),
          _desc(desc),
          _attrs(attrs),
          _pd(pd),
          _prim(pd) { }

    bool is_cpu() const override { return false; }

protected:
    virtual bool optimized_out(typed_primitive_inst<PType>&) const { return false; }

    static bool has_out_scales(const std::shared_ptr<dnnl::primitive_attr>& attr) {
        int mask;
        std::vector<float> scales;
        attr->get_output_scales(mask, scales);
        const auto drfv = reinterpret_cast<const int32_t&>(DNNL_RUNTIME_F32_VAL);
        return !scales.empty() && (reinterpret_cast<const int32_t&>(scales[0]) == drfv);
    }

    static bool has_zero_points(int arg, const std::shared_ptr<dnnl::primitive_attr>& attr) {
        int mask;
        std::vector<int32_t> zp;
        attr->get_zero_points(arg, mask, zp);
        const auto drsv = reinterpret_cast<const int32_t&>(DNNL_RUNTIME_S32_VAL);
        return !zp.empty() && (reinterpret_cast<const int32_t&>(zp[0]) == drsv);
    }

    static dnnl::post_ops try_optimize_post_ops(const typed_program_node<PType>& arg, dnnl::post_ops& p_ops,
                                                const std::shared_ptr<dnnl::primitive_attr>& attr,
                                                bool& optimization_is_completed) {
        // Get current node id for creating of optimization map
        auto node_id = arg.id();

        // Create new dnnl::post_ops object which will be filled inside the optimization process
        dnnl::post_ops optimized_p_ops;

        // Add new post-op into optimized_p_ops structure
        auto add_post_op = [&](onednn_post_op_type type, const dnnl::post_ops& cur_p_ops, dnnl::post_ops& new_p_ops, int idx) {
            switch (type) {
                case onednn_post_op_type::eltwise_act:
                case onednn_post_op_type::eltwise_clip:
                case onednn_post_op_type::eltwise_linear:
                case onednn_post_op_type::eltwise_round:
                {
                    dnnl::algorithm alg;
                    float scale, alpha, beta;
                    cur_p_ops.get_params_eltwise(idx, scale, alg, alpha, beta);
                    new_p_ops.append_eltwise(scale, alg, alpha, beta);
                    break;
                }

                case onednn_post_op_type::binary_add:
                case onednn_post_op_type::binary_mul:
                case onednn_post_op_type::binary_max:
                case onednn_post_op_type::binary_min:
                {
                    dnnl::algorithm alg;
                    dnnl::memory::desc desc;
                    cur_p_ops.get_params_binary(idx, alg, desc);
                    new_p_ops.append_binary(alg, desc);
                    break;
                }

                case onednn_post_op_type::scale:
                {
                    break;
                }

                case onednn_post_op_type::sum:
                {
                    float scale;
                    dnnl::memory::data_type data_type;
                    cur_p_ops.get_params_sum(idx, scale, data_type);
                    new_p_ops.append_sum(scale, data_type);
                    break;
                }

                case onednn_post_op_type::optimized:
                case onednn_post_op_type::optimized_sum:
                case onednn_post_op_type::optimized_eltwise:
                {
                    // Current operation already has been optimized => don't need extra actions
                    break;
                }

                default:
                    throw std::runtime_error("Unsupported onednn post-operation type");
            }
        };

        auto& cur_post_ops = onednn_fusing_map[node_id];

        size_t cur_post_op_idx = 1;
        size_t prev_post_op_idx = 0;
        bool optimization_done = false;

        // Check and update post-op map if we already optimized something
        for (size_t post_op_idx = 0; post_op_idx < cur_post_ops.size(); post_op_idx++) {
            if (cur_post_ops[post_op_idx].op_type == onednn_post_op_type::optimized_sum)
                cur_post_ops[post_op_idx].op_type = onednn_post_op_type::sum;
            else if (cur_post_ops[post_op_idx].op_type == onednn_post_op_type::optimized_eltwise)
                cur_post_ops[post_op_idx].op_type = onednn_post_op_type::eltwise_linear;
            else if (cur_post_ops[post_op_idx].op_type == onednn_post_op_type::optimized)
                cur_post_ops.erase(cur_post_ops.begin() + post_op_idx);
        }

        // Get post-ops size for current node
        auto post_ops_size = cur_post_ops.size();

        // Try to combine pairs of arithmetic post-ops (adds and muls) into one operation inside this cycle
        while (!optimization_done) {
            auto cur_type = cur_post_ops[cur_post_op_idx].op_type;
            auto prev_type = cur_post_ops[prev_post_op_idx].op_type;

            // Ignore optimized operations for "previous" operation in our operation pair
            while ((prev_type == onednn_post_op_type::optimized || prev_type == onednn_post_op_type::optimized_sum ||
                   prev_type == onednn_post_op_type::optimized_eltwise) && cur_post_op_idx < post_ops_size - 1) {
                prev_post_op_idx++;
                cur_post_op_idx++;
                prev_type = cur_post_ops[prev_post_op_idx].op_type;
                cur_type = cur_post_ops[cur_post_op_idx].op_type;
            }

            // Ignore optimized operations for "current" operation in our operation pair
            while ((cur_type == onednn_post_op_type::optimized || cur_type == onednn_post_op_type::optimized_sum ||
                   cur_type == onednn_post_op_type::optimized_eltwise) && cur_post_op_idx < post_ops_size - 1) {
                cur_post_op_idx++;
                cur_type = cur_post_ops[cur_post_op_idx].op_type;
            }

            auto cur_idx = static_cast<int>(has_out_scales(attr) ? (cur_post_op_idx >= 1 ? cur_post_op_idx - 1 : 0) : cur_post_op_idx);
            auto prev_idx = static_cast<int>(has_out_scales(attr) ? (prev_post_op_idx >= 1 ? prev_post_op_idx - 1 : 0) : prev_post_op_idx);
            auto cur_type_is_optimized = cur_type == onednn_post_op_type::optimized ||
                                         cur_type == onednn_post_op_type::optimized_sum ||
                                         cur_type == onednn_post_op_type::optimized_eltwise;
            auto prev_type_is_optimized = prev_type == onednn_post_op_type::optimized ||
                                          prev_type == onednn_post_op_type::optimized_sum ||
                                          prev_type == onednn_post_op_type::optimized_eltwise;

            // If this is the last pair and it's optimized - add the last post-op and go out from the cycle
            if (cur_post_op_idx == post_ops_size - 1 && (cur_type_is_optimized || prev_type_is_optimized)) {
                if (!prev_type_is_optimized) {
                    add_post_op(prev_type, p_ops, optimized_p_ops, prev_idx);
                }
                if (!cur_type_is_optimized) {
                    add_post_op(cur_type, p_ops, optimized_p_ops, cur_idx);
                }
                break;
            }

            auto equal_ops = cur_type == prev_type;
            auto cur_type_is_binary_add_or_mul = cur_type == onednn_post_op_type::binary_add || cur_type == onednn_post_op_type::binary_mul;
            auto prev_type_is_binary_add_or_mul = prev_type == onednn_post_op_type::binary_add || prev_type == onednn_post_op_type::binary_mul;

            // Post-ops combinations which can be simplified
            auto eltw_and_eltw = equal_ops && cur_type == onednn_post_op_type::eltwise_linear;
            auto bin_and_eltw = cur_type_is_binary_add_or_mul && prev_type == onednn_post_op_type::eltwise_linear;
            auto eltw_and_bin = cur_type == onednn_post_op_type::eltwise_linear && prev_type_is_binary_add_or_mul;
            auto eltw_and_sum = cur_type == onednn_post_op_type::eltwise_linear && prev_type == onednn_post_op_type::sum;
            auto eltw_and_scale = cur_type == onednn_post_op_type::eltwise_linear && prev_type == onednn_post_op_type::scale;

            auto can_try_optimize = eltw_and_eltw ||
                                    bin_and_eltw ||
                                    eltw_and_bin ||
                                    eltw_and_sum ||
                                    eltw_and_scale;

            bool cur_ops_pair_is_optimized = false;

            if (can_try_optimize) {
                if (eltw_and_eltw) {
                    dnnl::algorithm alg;
                    float cur_scale, prev_scale, cur_alpha, prev_alpha, cur_beta, prev_beta;

                    p_ops.get_params_eltwise(prev_idx, prev_scale, alg, prev_alpha, prev_beta);
                    p_ops.get_params_eltwise(cur_idx, cur_scale, alg, cur_alpha, cur_beta);

                    // Eltwise + eltwise pair can be optimized only if cur_alpha is equal to 1.0f
                    if (cur_alpha == 1.0f && prev_scale == cur_scale) {
                        dnnl::post_ops eltw_p_op;
                        eltw_p_op.append_eltwise(cur_scale, alg, prev_alpha, cur_beta + prev_beta);

                        // Combine 2 eltwises into one
                        add_post_op(cur_type, eltw_p_op, optimized_p_ops, 0);

                        // Marked current and previous eltwise operations as 'optimized' (they will be ignored on the next iteration of cycle)
                        cur_post_ops[cur_post_op_idx].op_type = onednn_post_op_type::optimized;
                        cur_post_ops[prev_post_op_idx].op_type = onednn_post_op_type::optimized_eltwise;

                        // Set the flag if extra optimizations checking is needed
                        if (cur_post_op_idx < post_ops_size - 1) {
                            if (cur_post_ops[cur_post_op_idx + 1].op_type == onednn_post_op_type::eltwise_linear ||
                                cur_post_ops[cur_post_op_idx + 1].op_type == onednn_post_op_type::binary_add ||
                                cur_post_ops[cur_post_op_idx + 1].op_type == onednn_post_op_type::binary_mul ||
                                cur_post_ops[cur_post_op_idx + 1].op_type == onednn_post_op_type::optimized_eltwise) {
                                optimization_is_completed = true;
                            }
                        }
                        cur_ops_pair_is_optimized = true;
                    }
                } else if (bin_and_eltw) {
                    dnnl::algorithm alg;
                    dnnl::memory::desc desc;
                    float scale, alpha, beta;

                    cldnn::program_node& cur_node = arg.get_dependency(cur_post_ops[cur_post_op_idx].mem_dep);

                    p_ops.get_params_binary(cur_idx, alg, desc);
                    p_ops.get_params_eltwise(prev_idx, scale, alg, alpha, beta);

                    // Eltwise operations can use runtime non-constant data buffers, so check that memory buffers consist of constant data only
                    auto bin_ops_can_be_optimized = cur_node.is_type<data>() && cur_node.is_constant() &&
                                                    cur_node.get_users().size() == 1 && desc.data_type() == dnnl_f32;

                    auto bin_add_and_eltw = alpha == 1.0f && scale == 1.0f && cur_type == onednn_post_op_type::binary_add && bin_ops_can_be_optimized;
                    auto bin_mul_and_eltw = beta == 0.f && scale == 1.0f && cur_type == onednn_post_op_type::binary_mul && bin_ops_can_be_optimized;

                    if (bin_add_and_eltw || bin_mul_and_eltw) {
                        memory::ptr cur_bin_mem_ptr = cur_node.as<data>().get_attached_memory_ptr();
                        auto& stream = cur_bin_mem_ptr->get_engine()->get_program_stream();
                        mem_lock<float, mem_lock_type::write> bin_and_eltw_lock(cur_bin_mem_ptr, stream);

                        size_t cur_bin_mem_size = cur_node.get_output_layout().count();

                        // Update all binary coefficients
                        if (bin_add_and_eltw) {
                            for (size_t data_idx = 0; data_idx < cur_bin_mem_size; data_idx++) {
                                bin_and_eltw_lock[data_idx] += beta;
                            }
                        } else {
                            for (size_t data_idx = 0; data_idx < cur_bin_mem_size; data_idx++) {
                                bin_and_eltw_lock[data_idx] *= alpha;
                            }
                        }

                        // Marked previous eltwise operation as 'optimized' (it will be ignored on the next iteration of cycle)
                        cur_post_ops[prev_post_op_idx].op_type = onednn_post_op_type::optimized;

                        cur_ops_pair_is_optimized = true;
                    }
                } else if (eltw_and_bin) {
                    dnnl::algorithm alg;
                    dnnl::memory::desc desc;
                    float scale, alpha, beta;

                    cldnn::program_node& prev_node = arg.get_dependency(cur_post_ops[prev_post_op_idx].mem_dep);

                    p_ops.get_params_eltwise(cur_idx, scale, alg, alpha, beta);
                    p_ops.get_params_binary(prev_idx, alg, desc);

                    // Eltwise operations can use runtime non-constant data buffers, so check that memory buffers consist of constant data only
                    auto bin_ops_can_be_optimized = prev_node.is_type<data>() && prev_node.is_constant() &&
                                                    prev_node.get_users().size() == 1 && desc.data_type() == dnnl_f32;

                    auto eltw_and_bin_add = alpha == 1.0f && scale == 1.0f && prev_type == onednn_post_op_type::binary_add && bin_ops_can_be_optimized;
                    auto eltw_and_bin_mul = beta == 0.f && scale == 1.0f && prev_type == onednn_post_op_type::binary_mul && bin_ops_can_be_optimized;

                    if (eltw_and_bin_add || eltw_and_bin_mul) {
                        memory::ptr prev_bin_mem_ptr = prev_node.as<data>().get_attached_memory_ptr();
                        auto& stream = prev_bin_mem_ptr->get_engine()->get_program_stream();
                        mem_lock<float, mem_lock_type::write> eltw_and_bin_lock(prev_bin_mem_ptr, stream);

                        size_t prev_bin_mem_size = prev_node.get_output_layout().count();

                        // Update all binary coefficients
                        if (eltw_and_bin_add) {
                            for (size_t data_idx = 0; data_idx < prev_bin_mem_size; data_idx++) {
                                eltw_and_bin_lock[data_idx] += beta;
                            }
                        } else {
                            for (size_t data_idx = 0; data_idx < prev_bin_mem_size; data_idx++) {
                                eltw_and_bin_lock[data_idx] *= alpha;
                            }
                        }

                        // Marked current eltwise operation as 'optimized' (it will be ignored on the next iteration of cycle)
                        cur_post_ops[cur_post_op_idx].op_type = onednn_post_op_type::optimized;

                        cur_ops_pair_is_optimized = true;
                    }
                } else if (eltw_and_sum) {
                    dnnl::algorithm alg;
                    float cur_scale, prev_scale, alpha, beta;
                    dnnl::memory::data_type data_type;

                    cldnn::program_node& prev_node = arg.get_dependency(cur_post_ops[prev_post_op_idx].mem_dep);

                    p_ops.get_params_eltwise(cur_idx, cur_scale, alg, alpha, beta);
                    p_ops.get_params_sum(prev_idx, prev_scale, data_type);

                    // Eltwise operations can use runtime non-constant data buffers, so check that memory buffers consist of constant data only
                    auto eltw_ops_can_be_optimized = prev_node.is_type<data>() && prev_node.is_constant() &&
                                                     prev_node.get_users().size() == 1;

                    // Eltwise can be inserted into the scale field of previous sum if cur_beta is equal to 0.f
                    if (beta == 0.f && cur_scale == 1.0f && eltw_ops_can_be_optimized) {
                        dnnl::post_ops sum_p_op;
                        sum_p_op.append_sum(alpha * prev_scale, data_type);

                        // Insert cur eltwise into sum
                        add_post_op(prev_type, sum_p_op, optimized_p_ops, 0);

                        memory::ptr prev_eltw_mem_ptr = prev_node.as<data>().get_attached_memory_ptr();
                        auto& stream = prev_eltw_mem_ptr->get_engine()->get_program_stream();
                        mem_lock<float, mem_lock_type::write> eltw_and_sum_lock(prev_eltw_mem_ptr, stream);

                        size_t prev_eltw_mem_size = prev_node.get_output_layout().count();

                        // Also multiply sum on alpha for getting valid results
                        for (size_t data_idx = 0; data_idx < prev_eltw_mem_size; data_idx++) {
                            eltw_and_sum_lock[data_idx] *= alpha;
                        }

                        // Marked current and previous operations as 'optimized' (they will be ignored on the next iteration of cycle)
                        cur_post_ops[cur_post_op_idx].op_type = onednn_post_op_type::optimized;
                        cur_post_ops[prev_post_op_idx].op_type = onednn_post_op_type::optimized_sum;

                        // Set the flag if extra optimizations checking is needed
                        if (cur_post_op_idx < post_ops_size - 1) {
                            if (cur_post_ops[cur_post_op_idx + 1].op_type == onednn_post_op_type::eltwise_linear ||
                                cur_post_ops[cur_post_op_idx + 1].op_type == onednn_post_op_type::optimized_eltwise) {
                                optimization_is_completed = true;
                            }
                        }
                        cur_ops_pair_is_optimized = true;
                    }
                } else if (eltw_and_scale) {
                    dnnl::algorithm alg;
                    float cur_scale, alpha, beta;

                    cldnn::program_node& prev_node = arg.get_dependency(cur_post_ops[prev_post_op_idx].mem_dep);

                    p_ops.get_params_eltwise(cur_idx, cur_scale, alg, alpha, beta);

                    // Eltwise can be inserted into output_scale if cur_beta is equal to 0.f and cur_scale is equal to 1.0f
                    if (beta == 0.f && cur_scale == 1.0f && prev_node.get_output_layout().data_type == data_types::f32) {
                        memory::ptr prev_scale_mem_ptr = prev_node.as<data>().get_attached_memory_ptr();
                        auto& stream = prev_scale_mem_ptr->get_engine()->get_program_stream();
                        mem_lock<float, mem_lock_type::write> eltw_and_scale_lock(prev_scale_mem_ptr, stream);

                        size_t prev_scale_mem_size = prev_node.get_output_layout().count();

                        // Update all scale coefficients
                        for (size_t data_idx = 0; data_idx < prev_scale_mem_size; data_idx++) {
                            eltw_and_scale_lock[data_idx] *= alpha;
                        }

                        // Marked current eltwise operation as 'optimized' (it will be ignored on the next iteration of cycle)
                        cur_post_ops[cur_post_op_idx].op_type = onednn_post_op_type::optimized;

                        cur_ops_pair_is_optimized = true;
                    }
                }
            }

            // If no optimizations have been applied then copy post-op info into the new optimized_p_ops structure
            if (!(has_out_scales(attr) && prev_post_op_idx == 0) && !cur_ops_pair_is_optimized) {
                add_post_op(prev_type, p_ops, optimized_p_ops, prev_idx);
            }

            if (cur_post_op_idx == post_ops_size - 1 && !cur_ops_pair_is_optimized) {
                add_post_op(cur_type, p_ops, optimized_p_ops, cur_idx);
                optimization_done = true;
            } else if (cur_post_ops[cur_post_op_idx].op_type != onednn_post_op_type::optimized) {
                cur_post_op_idx++;
                prev_post_op_idx++;
            }
        }

        optimization_is_completed = !optimization_is_completed;

        return optimized_p_ops;
    }

    void configure_post_ops_arguments(typed_primitive_inst<PType>& instance, std::unordered_map<int, dnnl::memory>& args) const {
        // Get current node id for creating of optimization map
        auto node_id = instance.id();
        auto& engine = instance.get_network().get_engine();
        auto dnnl_engine = engine.get_onednn_engine();

        // Get current post-ops info
        dnnl::post_ops post_ops = _attrs->get_post_ops();

        // Create onednn memory buffers for post-ops
        auto& cur_post_ops = onednn_fusing_map[node_id];
        auto post_ops_size = cur_post_ops.size();
        for (size_t post_op_idx = 0, num_of_optimized_post_ops = 0; post_op_idx < post_ops_size; post_op_idx++) {
            auto post_op_type = cur_post_ops[post_op_idx].op_type;
            auto memory_offset = cur_post_ops[post_op_idx].mem_offset;
            auto onednn_post_op_idx = has_out_scales(_attrs) && post_op_idx > 0 ? post_op_idx - 1 : post_op_idx;
            onednn_post_op_idx -= num_of_optimized_post_ops;

            switch (post_op_type) {
                case onednn_post_op_type::eltwise_act:
                case onednn_post_op_type::eltwise_clip:
                case onednn_post_op_type::eltwise_linear:
                case onednn_post_op_type::eltwise_round:
                {
                    // onednn elwise doesn't need any data from memory buffers
                    break;
                }

                case onednn_post_op_type::binary_add:
                case onednn_post_op_type::binary_mul:
                case onednn_post_op_type::binary_max:
                case onednn_post_op_type::binary_min:
                {
                    auto binary_op_mem = instance.fused_memory(memory_offset);
                    dnnl::algorithm alg;
                    dnnl::memory::desc desc;
                    post_ops.get_params_binary(static_cast<int>(onednn_post_op_idx), alg, desc);
                    args.insert({DNNL_ARG_ATTR_MULTIPLE_POST_OP(static_cast<int>(onednn_post_op_idx)) | DNNL_ARG_SRC_1,
                                 binary_op_mem->get_onednn_memory(desc)});
                    break;
                }

                case onednn_post_op_type::scale:
                {
                    auto scale_op_mem = instance.fused_memory(memory_offset);
                    dnnl::memory::desc desc = onednn::layout_to_memory_desc(scale_op_mem->get_layout(), dnnl::memory::format_tag::a, true);
                    args.insert({DNNL_ARG_ATTR_OUTPUT_SCALES, scale_op_mem->get_onednn_memory(desc)});
                    break;
                }

                case onednn_post_op_type::sum:
                case onednn_post_op_type::optimized_sum:
                case onednn_post_op_type::optimized_eltwise:
                {
                    break;
                }

                case onednn_post_op_type::optimized:
                {
                    // Optimized post-op, count it to respect onednn_post_op_idx in the next operations
                    num_of_optimized_post_ops++;
                    break;
                }

                default:
                    throw std::runtime_error("Unsupported onednn post-operation type");
            }
        }
    }

    virtual std::unordered_map<int, dnnl::memory> get_arguments(typed_primitive_inst<PType>& instance) const {
        std::unordered_map<int, dnnl::memory> args;
        auto& engine = instance.get_network().get_engine();
        auto dnnl_engine = engine.get_onednn_engine();

        {
            auto& input = instance.input_memory(0);
            args.insert({DNNL_ARG_SRC, input.get_onednn_memory(_pd.dnnl::primitive_desc_base::src_desc(0))});
        }

        {
            auto& output = instance.output_memory();
            args.insert({DNNL_ARG_DST, output.get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(0))});
        }

        configure_post_ops_arguments(instance, args);

        return args;
    }

    void init_kernels() override { }

    static std::shared_ptr<dnnl::primitive_attr> get_primitive_attributes(const typed_program_node<PType>& arg) {
        const std::vector<fused_primitive_desc>& cldnn_post_ops = arg.get_fused_primitives();
        auto attrs = std::make_shared<dnnl::primitive_attr>();
        dnnl::post_ops post_ops;
        size_t memory_offset = 0;

        // Create onednn post-ops list related to the current node
        std::vector<onednn_post_op_desc> fused_ops;

        // Added this for debug purposes only
        size_t empty_mem = 0xff;

        // Add information about post-operation into the list, update indices
        auto update_onednn_post_op_list = [&](onednn_post_op_type type, size_t m_dep) {
            onednn_post_op_desc cur_op_desc = { type, memory_offset, m_dep };
            fused_ops.push_back(cur_op_desc);

            auto has_memory_buffers = type == onednn_post_op_type::binary_add ||
                                      type == onednn_post_op_type::binary_mul ||
                                      type == onednn_post_op_type::binary_max ||
                                      type == onednn_post_op_type::binary_min ||
                                      type == onednn_post_op_type::scale ||
                                      type == onednn_post_op_type::sum;
            if (has_memory_buffers)
                memory_offset++;
        };

        for (size_t idx = 0; idx < cldnn_post_ops.size(); idx++) {
            auto node = cldnn_post_ops[idx].node;

            if (node->is_type<activation>()) {
                auto fused_desc = node->as<activation>().get_primitive();
                dnnl::algorithm alg = onednn::convert_activation_func(fused_desc->activation_function);
                post_ops.append_eltwise(1.0f, alg, fused_desc->additional_params.a, fused_desc->additional_params.b);
                update_onednn_post_op_list(onednn_post_op_type::eltwise_act, empty_mem);
            } else if (node->is_type<eltwise>()) {
                auto& e_node = node->as<eltwise>();
                auto dep_idx = cldnn_post_ops[idx].dep_start_idx;
                auto in = arg.get_dependency(dep_idx).get_output_layout();

                if (e_node.get_primitive()->mode == eltwise_mode::sum) {
                    if (e_node.get_primitive()->needs_onednn_sum_post_op(in)) {
                        post_ops.append_sum(1.0f, onednn::convert_data_type(in.data_type));
                        update_onednn_post_op_list(onednn_post_op_type::sum, dep_idx);
                    } else {
                        dnnl::memory::desc in_desc = onednn::layout_to_memory_desc(in, dnnl::memory::format_tag::ab, true);
                        post_ops.append_binary(dnnl::algorithm::binary_add, in_desc);
                        update_onednn_post_op_list(onednn_post_op_type::binary_add, dep_idx);
                    }
                } else {
                    if (in.size.spatial[0] > 1 || in.size.spatial[1] > 1 || in.size.batch[0] > 1)
                        throw std::runtime_error("Unsupported eltwise mode for fused onednn op");
                    if (idx == 0 && !has_out_scales(attrs)) {
                        int mask = in.count() > 1 ? 2 : 0;
                        attrs->set_output_scales(mask, {DNNL_RUNTIME_F32_VAL});
                        update_onednn_post_op_list(onednn_post_op_type::scale, dep_idx);
                    } else {
                        dnnl::memory::desc in_desc = onednn::layout_to_memory_desc(in, dnnl::memory::format_tag::ab, true);
                        post_ops.append_binary(dnnl::algorithm::binary_mul, in_desc);
                        update_onednn_post_op_list(onednn_post_op_type::binary_mul, dep_idx);
                    }
                }
            } else if (node->is_type<quantize>()) {
                auto& q_node = node->as<quantize>();
                auto dep_idx = cldnn_post_ops[idx].dep_start_idx;

                if (q_node.get_per_tensor_output_range() && q_node.get_output_lo_val() < q_node.get_output_hi_val()) {
                    // 1. pre-scale & pre-shift
                    {
                        if (q_node.get_per_tensor_input_scale() && q_node.get_per_tensor_input_shift()) {
                            post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_input_scale_val(), q_node.get_input_shift_val());
                            update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                        } else {
                            if (q_node.get_per_tensor_input_scale()) {
                                post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_input_scale_val(), 0.0f);
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto in_scale = arg.get_dependency(dep_idx++).get_output_layout();
                                if (idx == 0 && !has_out_scales(attrs) && in_scale.data_type == data_types::f32 &&
                                    arg.type() == convolution::type_id() &&
                                    !data_type_traits::is_floating_point(arg.get_dependency(0).get_output_layout().data_type)) {
                                    int mask = in_scale.count() > 1 ? 2 : 0;
                                    attrs->set_output_scales(mask, {DNNL_RUNTIME_F32_VAL});
                                    update_onednn_post_op_list(onednn_post_op_type::scale, dep_idx - 1);
                                } else {
                                    dnnl::memory::desc in_scale_desc = onednn::layout_to_memory_desc(in_scale, dnnl::memory::format_tag::ab, true);
                                    post_ops.append_binary(dnnl::algorithm::binary_mul, in_scale_desc);
                                    update_onednn_post_op_list(onednn_post_op_type::binary_mul, dep_idx - 1);
                                }
                            }

                            if (q_node.get_need_pre_shift()) {
                                if (q_node.get_per_tensor_input_shift()) {
                                    post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, 1.0f, q_node.get_input_shift_val());
                                    update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                                } else {
                                    auto in_shift = arg.get_dependency(dep_idx++).get_output_layout();
                                    dnnl::memory::desc in_shift_desc = onednn::layout_to_memory_desc(in_shift, dnnl::memory::format_tag::ab, true);
                                    post_ops.append_binary(dnnl::algorithm::binary_add, in_shift_desc);
                                    update_onednn_post_op_list(onednn_post_op_type::binary_add, dep_idx - 1);
                                }
                            }
                        }
                    }

                    // 2. round
                    auto out_dt = cldnn_post_ops[idx].output_layout.data_type;
                    bool output_type_is_int8 = out_dt == data_types::u8 || out_dt == data_types::i8;
                    if (!output_type_is_int8) {
                        post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_round, 0.0f, 0.0f);
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_round, empty_mem);
                    }

                    // 3. post-scale & post-shift
                    if (q_node.get_need_post_scale() && q_node.get_need_post_shift() &&
                        q_node.get_per_tensor_output_scale() && q_node.get_per_tensor_output_shift()) {
                        post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_output_scale_val(), q_node.get_output_shift_val());
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                    } else {
                        if (q_node.get_need_post_scale()) {
                            if (q_node.get_per_tensor_output_scale()) {
                                post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_output_scale_val(), 0.0f);
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto out_scale = arg.get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc out_scale_desc = onednn::layout_to_memory_desc(out_scale, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_mul, out_scale_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_mul, dep_idx - 1);
                            }
                        }

                        if (q_node.get_need_post_shift()) {
                            if (q_node.get_per_tensor_output_shift()) {
                                post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, 1.0f, q_node.get_output_shift_val());
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto out_shift = arg.get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc out_shift_desc = onednn::layout_to_memory_desc(out_shift, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_add, out_shift_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_add, dep_idx - 1);
                            }
                        }
                    }

                    // 4. clamp
                    if (q_node.get_need_clamp()) {
                        float out_lo = q_node.get_need_min_clamp() ? q_node.get_output_lo_val() : data_type_traits::min<float>(out_dt);
                        float out_hi = q_node.get_need_max_clamp() ? q_node.get_output_hi_val() : data_type_traits::max<float>(out_dt);
                        post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_clip, out_lo, out_hi);
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_clip, empty_mem);
                    }
                } else {
                    // 1. clamp
                    if (q_node.get_need_clamp()) {
                        auto in_lo = arg.get_dependency(dep_idx++).get_output_layout();
                        auto in_hi = arg.get_dependency(dep_idx++).get_output_layout();
                        dnnl::algorithm clamp_max = dnnl::algorithm::binary_max;
                        dnnl::algorithm clamp_min = dnnl::algorithm::binary_min;
                        dnnl::memory::desc in_lo_desc = onednn::layout_to_memory_desc(in_lo, dnnl::memory::format_tag::ab, true);
                        dnnl::memory::desc in_hi_desc = onednn::layout_to_memory_desc(in_hi, dnnl::memory::format_tag::ab, true);

                        post_ops.append_binary(clamp_max, in_lo_desc);
                        update_onednn_post_op_list(onednn_post_op_type::binary_max, dep_idx - 2);
                        post_ops.append_binary(clamp_min, in_hi_desc);
                        update_onednn_post_op_list(onednn_post_op_type::binary_min, dep_idx - 1);
                    }

                    // 2. pre-scale & pre-shift
                    {
                        if (q_node.get_per_tensor_input_scale() && q_node.get_per_tensor_input_shift()) {
                            post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_input_scale_val(), q_node.get_input_shift_val());
                            update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                        } else {
                            if (q_node.get_per_tensor_input_scale()) {
                                post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_input_scale_val(), 0.0f);
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto in_scale = arg.get_dependency(dep_idx++).get_output_layout();
                                if (idx == 0 && !q_node.get_need_clamp() && !has_out_scales(attrs) && in_scale.data_type == data_types::f32 &&
                                    arg.type() == convolution::type_id() &&
                                    !data_type_traits::is_floating_point(arg.get_dependency(0).get_output_layout().data_type)) {
                                    int mask = in_scale.count() > 1 ? 2 : 0;
                                    attrs->set_output_scales(mask, {DNNL_RUNTIME_F32_VAL});
                                    update_onednn_post_op_list(onednn_post_op_type::scale, dep_idx - 1);
                                } else {
                                    dnnl::memory::desc in_scale_desc = onednn::layout_to_memory_desc(in_scale, dnnl::memory::format_tag::ab, true);
                                    post_ops.append_binary(dnnl::algorithm::binary_mul, in_scale_desc);
                                    update_onednn_post_op_list(onednn_post_op_type::binary_mul, dep_idx - 1);
                                }
                            }

                            if (q_node.get_need_pre_shift()) {
                                if (q_node.get_per_tensor_input_shift()) {
                                    post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, 1.0f, q_node.get_input_shift_val());
                                    update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                                } else {
                                    auto in_shift = arg.get_dependency(dep_idx++).get_output_layout();
                                    dnnl::memory::desc in_shift_desc = onednn::layout_to_memory_desc(in_shift, dnnl::memory::format_tag::ab, true);
                                    post_ops.append_binary(dnnl::algorithm::binary_add, in_shift_desc);
                                    update_onednn_post_op_list(onednn_post_op_type::binary_add, dep_idx - 1);
                                }
                            }
                        }
                    }

                    // 3. round
                    {
                        post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_round, 0.0f, 0.0f);
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_round, empty_mem);
                    }

                    // 4. post-scale & post-shift
                    if (q_node.get_need_post_scale() && q_node.get_need_post_shift() &&
                        q_node.get_per_tensor_output_scale() && q_node.get_per_tensor_output_shift()) {
                        post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_output_scale_val(), q_node.get_output_shift_val());
                        update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                    } else {
                        if (q_node.get_need_post_scale()) {
                            if (q_node.get_per_tensor_output_scale()) {
                                post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, q_node.get_output_scale_val(), 0.0f);
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto out_scale = arg.get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc out_scale_desc = onednn::layout_to_memory_desc(out_scale, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_mul, out_scale_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_mul, dep_idx - 1);
                            }
                        }

                        if (q_node.get_need_post_shift()) {
                            if (q_node.get_per_tensor_output_shift()) {
                                post_ops.append_eltwise(1.0f, dnnl::algorithm::eltwise_linear, 1.0f, q_node.get_output_shift_val());
                                update_onednn_post_op_list(onednn_post_op_type::eltwise_linear, empty_mem);
                            } else {
                                auto out_shift = arg.get_dependency(dep_idx++).get_output_layout();
                                dnnl::memory::desc out_shift_desc = onednn::layout_to_memory_desc(out_shift, dnnl::memory::format_tag::ab, true);
                                post_ops.append_binary(dnnl::algorithm::binary_add, out_shift_desc);
                                update_onednn_post_op_list(onednn_post_op_type::binary_add, dep_idx - 1);
                            }
                        }
                    }
                }
            } else if (node->is_type<reorder>()) {
                continue;
            } else {
                throw std::runtime_error("Unsupported fused op of " + node->get_primitive()->type_string() + " type for oneDNN primitive");
            }
        }

        if (cldnn_post_ops.size() && arg.get_fused_activations_funcs().size())
            throw std::runtime_error("Unsupported mix of fused ops and activations");

        for (size_t i = 0; i < arg.get_fused_activations_funcs().size(); i++) {
            auto activation_type = arg.get_fused_activations_funcs()[i];
            auto params = arg.get_fused_activations_params()[i];
            dnnl::algorithm alg = onednn::convert_activation_func(activation_type);
            post_ops.append_eltwise(1.0f, alg, params.a, params.b);
            update_onednn_post_op_list(onednn_post_op_type::eltwise_act, empty_mem);
        }

        // Update total onednn post-ops info
        auto it = onednn_fusing_map.find(arg.id());
        if (it != onednn_fusing_map.end()) {
            it->second = std::move(fused_ops);
        } else {
            onednn_fusing_map.emplace(arg.id(), std::move(fused_ops));
        }

        // Trying to optimize more than 1 post-ops
        auto post_ops_size = onednn_fusing_map[arg.id()].size();

        if (post_ops_size > 1) {
            dnnl::post_ops optimized_post_ops = post_ops;
            bool optimization_is_finished = false;

            // Trying to combine multiplications and additions which are placed one after another.
            // We do it in the cycle because "eltw + eltw" cases can be simplified again in some cases.
            do {
                optimized_post_ops = try_optimize_post_ops(arg, optimized_post_ops, attrs, optimization_is_finished);
            } while (!optimization_is_finished);

            attrs->set_post_ops(optimized_post_ops);
        } else {
            // Set post-ops without any optimizations
            attrs->set_post_ops(post_ops);
        }

        return attrs;
    }

    event::ptr aggregate_events(const std::vector<event::ptr>& events, stream& stream, bool group = false, bool is_output = false) const {
        if (events.size() == 1 && !is_output)
            return events[0];

        if (group && !is_output)
            return stream.group_events(events);

        return stream.enqueue_marker(events, is_output);
    }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override {
        uint32_t net_id = instance.get_network().get_id();
        _args[net_id] = get_arguments(instance);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& /* events */,
                            typed_primitive_inst<PType>& instance) override {
        auto& network = instance.get_network();
        auto& engine = network.get_engine();
        auto& stream = network.get_stream();
        auto profiling = engine.configuration().enable_profiling;
        auto net_id = network.get_id();
        event::ptr event;

        if (profiling) {
            stream.finish();
            event = stream.create_user_event(false);
        }

        _prim.execute(stream.get_onednn_stream(), _args[net_id]);

        if (profiling) {
            stream.finish();
            event->set();
        } else {
            // Create and set user event as complete
            event = stream.create_user_event(true);
        }

        if (!event) {
            std::string error_msg = "Event was not created properly for " + instance.id();
            throw std::runtime_error(error_msg);
        }

        return event;
    }
};

}  // namespace onednn
}  // namespace cldnn
