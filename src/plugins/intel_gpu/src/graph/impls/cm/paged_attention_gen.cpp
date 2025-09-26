// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention_gen.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <utility>

#include "../ocl_v2/sdpa/paged_attention_common.hpp"
#include "../ocl_v2/sdpa/sdpa_base.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "openvino/core/partial_shape.hpp"
#include "paged_attention_inst.h"
#include "primitive_cm_base.hpp"
#include "primitive_inst.h"

#ifdef CM_PA_ENABLE
namespace ov::intel_gpu::cm {

using namespace ov;
using namespace ov::intel_gpu::ocl;
using namespace cldnn;
namespace {

// constexpr ov::element::Type softmax_accumulator_type = ov::element::f32;
// constexpr size_t paged_attention_block_size = 16;
// constexpr size_t seq_len_partition_size = 256;
// constexpr size_t subgroup_size = 16;
constexpr size_t WG_SIZE = 16;
constexpr size_t kv_split_data_size = 16;
constexpr size_t split_output_idx = 3;
// constexpr size_t lse_idx = 4;

}  // namespace

// struct PagedAttentionRuntimeParams : public ImplRuntimeParams {
//     PagedAttentionStage stage;
//     size_t num_of_partitions;
//     size_t partition_size;
//     size_t paged_attention_aligned_seq_len;
//     size_t sdpa_opt_max_seq_len;
//     // size_t sdpa_opt_seq_len_partition_size;
// };

// inline size_t get_target_seq_len_block_size() {
//     constexpr size_t block_size = 16;
//     return block_size;
// }

// inline size_t get_generate_stage_block_size(size_t head_size) {
//     auto preferred_block_size = {4, 2, 1};
//     for (const auto& block_size : preferred_block_size) {
//         if (head_size % (block_size * subgroup_size) == 0) {
//             return block_size;
//         }
//     }
//     return 1;
// }

// This function returns the kv_step and kv_split_len based on the architecture.
// return {kv_step, kv_split_len}
inline std::pair<size_t, size_t> get_kv_split_size(size_t arch) {
    if (arch == 1) {
        return {8, 32};  // For Xe1
    } else if (arch == 2) {
        return {16, 32};  // For Xe2
    }
    OPENVINO_ASSERT(false, "Unsupported architecture for KV split size");
    return {0, 0};  // Fallback case, should not be reached
}

inline size_t get_q_step(size_t arch, bool is_single_token = false) {
    if (arch == 1) {
        return is_single_token ? 1 : 8;  // For Xe1
    } else if (arch == 2) {
        return is_single_token ? 1 : 16;  // For Xe2
    }
    OPENVINO_ASSERT(false, "Unsupported architecture for Q step");
    return 0;  // Fallback case, should not be reached
}

inline size_t get_kv_len(const RuntimeParams& params, const PagedAttentionStage& stage) {
    if (stage == PagedAttentionStage::PREFILL) {
        auto key_shape = params.input_layouts[1].get_shape();
        const size_t kv_len = key_shape[key_shape.size() - 2];
        return kv_len;
    } else if (stage == PagedAttentionStage::GENERATE) {
        // TODO FIX: key_cache shape = [16, 128+4, 4, 2269]
        //  auto key_cache_shape = params.input_layouts[3].get_shape();
        //  const size_t kv_len = key_cache_shape[0] * key_cache_shape[key_cache_shape.size() - 2];
        auto key_shape = params.input_layouts[1].get_shape();
        const size_t kv_len = key_shape[key_shape.size() - 2];
        // size_t i = 0;
        // for (auto& l : params.input_layouts) {
        //     auto _shape = l.get_shape();
        //     std::cout << i++ << " shape: " << _shape.to_string() << std::endl;
        // }
        // std::cout << std::endl;
        return kv_len;
    }
    OPENVINO_ASSERT(false, "Unsupported PagedAttentionStage for get_kv_len");
    return 0;  // Fallback case, should not be reached
}

// inline size_t get_split_num(const RuntimeParams& params, const PagedAttentionStage& stage) {
//     const size_t kv_len = get_kv_len(params, stage);
//     auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
//     const size_t split_num = kv_len / get_kv_split_size(xe_arch).second;

//     return split_num;
// }

inline bool is_qwen3_vl_dynamic_layout(const cldnn::layout& layout) {
    // sdap 3D shape: [batch_size(?), seqlen(?), head_size] --> [batch_size(?), 1, seqlen(?), head_size]
    // sdpa 4D shape: [batch_size(?), head_num, seqlen(?), head_size]
    // Qwen3 vl dynamic 3D layout:
    //                [seq_len(?), head_num, head_size]  --> [seq_len(?), 1, head_num, head_size]
    // Qwen3 vl dynamic 4D layout:
    //                Is it? [batch_size(?), head_num, seqlen(?), head_size]
    auto shape = layout.get_partial_shape();
    if (shape.size() == 3) {
        return true;
    }
    if (shape[0].is_dynamic() && shape[1].is_static() && shape[2].is_static()) {
        return true;
    }
    return false;
}

JitConstants PagedAttentionGeneratorBase::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = KernelGenerator::get_jit_constants(params);
    jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
    // std::cout << "PagedAttentionGeneratorBase::get_jit_constants: " << get_entry_point(params) << std::endl;

    if (params.is_type<paged_attention>()) {
        auto desc = params.typed_desc<paged_attention>();

        const float scale_factor = 1.0 / std::sqrt(static_cast<double>(desc->k_head_size));
        jit.add(make_jit_constant("CMFLA_SCALE_FACTOR", scale_factor));
        jit.make("CMFLA_NUM_HEADS", desc->heads_num);
        jit.make("CMFLA_HEAD_SIZE", desc->k_head_size);
        jit.make("CMFLA_NUM_KV_HEADS", desc->kv_heads_num);
    } else {
        auto new_params = SDPABase::requires_shape_canonicalization(params) ? SDPABase::static_canonicalize_shapes(params) : params;
        auto desc = new_params.typed_desc<scaled_dot_product_attention>();
        auto extended_input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
        auto extended_input_k_transpose_order = extend_order_in_num_heads_dim(desc->input_k_transpose_order);
        auto extended_input_v_transpose_order = extend_order_in_num_heads_dim(desc->input_v_transpose_order);
        auto extended_output_transpose_order = extend_order_in_num_heads_dim(desc->output_transpose_order);

        size_t q_num_head = 1, k_head_size = 1, k_num_head = 1;
        auto is_qwen3_vl = is_qwen3_vl_dynamic_layout(params.get_input_layout(0));
        if (is_qwen3_vl) {
            q_num_head = get_batch_size(new_params.get_input_layout(0), extended_input_q_transpose_order);
            k_head_size = get_head_size(new_params.get_input_layout(1), extended_input_k_transpose_order);
            k_num_head = get_batch_size(new_params.get_input_layout(1), extended_input_k_transpose_order);
        } else {
            q_num_head = get_num_heads(new_params.get_input_layout(0), extended_input_q_transpose_order);
            k_head_size = get_head_size(new_params.get_input_layout(1), extended_input_k_transpose_order);
            k_num_head = get_num_heads(new_params.get_input_layout(1), extended_input_k_transpose_order);
        }

        auto not_need_output_transpose = desc->output_transpose_order == desc->input_q_transpose_order;
        if (not_need_output_transpose) {
            jit.make("CMFLA_OUTPUT_BHLS", 0);
        } else {
            jit.make("CMFLA_OUTPUT_BHLS", 1);
        }

        // std::cout << "PagedAttentionGeneratorBase::get_jit_constants: q_num_head = " << q_num_head
        //           << ", k_head_size = " << k_head_size << ", k_num_head = " << k_num_head << std::endl;
        // std::cout << "new_params.get_input_layout(0) = " << new_params.get_input_layout(0).to_string() << std::endl;
        // std::cout << "new_params.get_input_layout(1) = " << new_params.get_input_layout(1).to_string() << std::endl;

        const float scale_factor = 1.0 / std::sqrt(static_cast<double>(k_head_size));
        // jit.add(make_jit_constant("SCALE_FACTOR", scale_factor));
        jit.add(make_jit_constant("CMFLA_SCALE_FACTOR", scale_factor));
        jit.make("CMFLA_NUM_HEADS", q_num_head);
        jit.make("CMFLA_HEAD_SIZE", k_head_size);
        jit.make("CMFLA_NUM_KV_HEADS", k_num_head);

        // std::cout << "q_num_head: " << q_num_head << ", k_head_size: " << k_head_size << ", k_num_head: " << k_num_head
        //           << ", need_output_transpose = " << !not_need_output_transpose << ", is_qwen3_vl = " << is_qwen3_vl << std::endl;
    }
    // auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
    // jit.make("XE_ARCH", xe_arch);
    return jit;
}

Arguments PagedAttentionSDPAGeneratorMultiToken::get_arguments_desc(const kernel_impl_params& params) const {
    // const auto desc = params.typed_desc<paged_attention>();

    Arguments args;
    args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // query
    args.push_back({ArgumentDescriptor::Types::INPUT, 1});  // key
    args.push_back({ArgumentDescriptor::Types::INPUT, 2});  // value

    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // q_len
    // args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // k_after_padding
    // args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // v_after_padding

    return args;
}

JitConstants PagedAttentionSDPAGeneratorMultiToken::get_jit_constants(const kernel_impl_params& params) const {
    auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);

    if (params.is_type<paged_attention>()) {
        jit.make("CMFLA_IS_CAUSAL", 1);
    } else {
        auto is_qwen3_vl = is_qwen3_vl_dynamic_layout(params.get_input_layout(0));
        if (is_qwen3_vl) {
            jit.make("CMFLA_IS_CAUSAL", 0);
        } else {
            jit.make("CMFLA_IS_CAUSAL", 1);
        }
    }

    auto is_dynamic_padding = [](const layout& layout) {
        const auto& data_padding = layout.data_padding;

        auto dynamic_str = data_padding._dynamic_dims_mask.to_string();
        if (dynamic_str.find('1') != std::string::npos) {
            return true;  // return true if dynamic padding exists
        }
        return false;
    };
    // std::cout << "params.get_input_layout(0) = " << params.get_input_layout(0).to_string() << std::endl;
    // std::cout << "params.get_input_layout(1) = " << params.get_input_layout(1).to_string() << std::endl;
    // std::cout << "params.get_input_layout(2) = " << params.get_input_layout(2).to_string() << std::endl;
    // size_t query_dynamic_padding = is_dynamic_padding(params.get_input_layout(0));
    size_t key_dynamic_padding = is_dynamic_padding(params.get_input_layout(1));
    size_t value_dynamic_padding = is_dynamic_padding(params.get_input_layout(2));
    // std::cout << "query_dynamic_padding = " << query_dynamic_padding << " , value_dynamic_padding = " << value_dynamic_padding
    //           << " , key_dynamic_padding = " << key_dynamic_padding << std::endl;

    if (value_dynamic_padding && !key_dynamic_padding) {
        jit.make("CMFLA_V_FUSED", 1);
    } else {
        jit.make("CMFLA_V_FUSED", 0);
    }
    // for (auto& it : jit) {
    //     std::cout << "\tjit[" << it.name << "] = " << it.value << std::endl;
    // }
    // std::cout << std::endl;
    return jit;
}

DispatchDataFunc PagedAttentionSDPAGeneratorMultiToken::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;

        size_t heads_num = 1, batch = 1, q_len = 1;
        if (params.is_type<paged_attention>()) {
            auto desc = params.typed_desc<paged_attention>();
            // auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);
            heads_num = desc->heads_num;
            auto out_shape = params.output_layouts[0].get_shape();
            batch = out_shape.size() < 4 ? 1 : out_shape[0];
            q_len = out_shape[0];
        } else {
            auto& new_params = SDPABase::requires_shape_canonicalization(params) ? SDPABase::static_canonicalize_shapes(params) : params;
            auto desc = new_params.typed_desc<scaled_dot_product_attention>();
            auto extended_input_q_transpose_order = extend_order_in_num_heads_dim(desc->input_q_transpose_order);
            auto out_shape = new_params.output_layouts[0].get_partial_shape();

            auto is_qwen3_vl = is_qwen3_vl_dynamic_layout(params.get_input_layout(0));
            if (is_qwen3_vl) {
                batch = get_num_heads(new_params.get_input_layout(0), extended_input_q_transpose_order);
                q_len = get_seq_length(new_params.get_input_layout(0), extended_input_q_transpose_order);
                heads_num = get_batch_size(new_params.get_input_layout(0), extended_input_q_transpose_order);

            } else {
                batch = get_batch_size(new_params.get_input_layout(0), extended_input_q_transpose_order);
                q_len = get_seq_length(new_params.get_input_layout(0), extended_input_q_transpose_order);
                heads_num = get_num_heads(new_params.get_input_layout(0), extended_input_q_transpose_order);
            }

            // auto get_simple_pitch = [](const layout& layout) {
            //     size_t pitch = 1;
            //     auto dims_padding = layout.get_padded_dims();
            //     std::cout << "dims_padding: " << ov::Shape(dims_padding.begin(), dims_padding.end()).to_string() << std::endl;
            //     for (size_t i = dims_padding.size() - 1; i > 0; --i) {
            //         pitch = dims_padding[i];
            //         if (pitch > 1) {
            //             break;
            //         }
            //     }
            //     return pitch;
            // };

            // auto get_simple_padding = [](const layout& layout) {
            //     size_t offset = 0;
            //     const auto& data_padding = layout.data_padding;
            //     const auto& upper_pads = data_padding._upper_size;
            //     for (auto& it : upper_pads) {
            //         if (it > 0) {
            //             offset = it;
            //             break;
            //         }
            //     }
            //     return offset;
            // };

            // // [1, 8, 2778, 128] --> [1, 8, 2907, 128], k_after_padding = 129
            // size_t k_after_padding = 0, v_after_padding = 0;
            // k_after_padding = get_simple_padding(params.get_input_layout(1));
            // v_after_padding = get_simple_padding(params.get_input_layout(2));

            // size_t k_pitch = get_simple_pitch(params.get_input_layout(1));
            // size_t v_pitch = get_simple_pitch(params.get_input_layout(2));
            // std::cout << "k_after_padding = " << k_after_padding << " , v_after_padding = " << v_after_padding << std::endl;
            // std::cout << "k_pitch = " << k_pitch << " , v_pitch = " << v_pitch << std::endl;

            // auto print_order = [](const std::vector<int64_t>& order) {
            //     std::cout << "[";
            //     for (size_t i = 0; i < order.size(); ++i) {
            //         std::cout << order[i];
            //         if (i != order.size() - 1) {
            //             std::cout << ", ";
            //         }
            //     }
            //     std::cout << "]" << std::endl;
            // };

            // std::cout << "desc->input_q_transpose_order = ";
            // print_order(desc->input_q_transpose_order);
            // std::cout << "extended_input_q_transpose_order = ";
            // print_order(extended_input_q_transpose_order);
            // std::cout << "params.get_input_layout(0) = " << new_params.get_input_layout(0).to_string() << std::endl;

            // std::cout << "desc->input_k_transpose_order = ";
            // print_order(desc->input_k_transpose_order);
            // std::cout << "params.get_input_layout(1) = " << new_params.get_input_layout(1).to_string() << std::endl;

            // std::cout << "desc->input_v_transpose_order = ";
            // print_order(desc->input_v_transpose_order);
            // std::cout << "params.get_input_layout(2) = " << new_params.get_input_layout(2).to_string() << std::endl;

            // std::cout << "desc->output_transpose_order = ";
            // print_order(desc->output_transpose_order);
            // std::cout << "params.get_output_layout(0) = " << new_params.get_output_layout(0).to_string() << std::endl;
        }

        auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
        const size_t q_step = get_q_step(xe_arch, false);
        const size_t q_group_size = WG_SIZE * q_step;
        const size_t q_threads = align_to(q_len, q_group_size) / q_step;

        // std::cout << "GWS = [" << batch << ", " << heads_num << ", " << q_threads << "], LWS = [1, 1, " << WG_SIZE << "], q_len = " << q_len
        //           << ", q_step = " << q_step << ", q_group_size = " << q_group_size
        //           << ", is_qwen3_vl = " << is_qwen3_vl_dynamic_layout(params.get_input_layout(0)) << std::endl;

        wgs.global = {batch, heads_num, q_threads};
        wgs.local = {1, 1, WG_SIZE};

        std::vector<size_t> scaler_value = {q_len};
        // std::vector<size_t> scaler_value = {q_len, k_after_padding, v_after_padding};
        scalars.resize(scaler_value.size());
        for (size_t i = 0; i < scaler_value.size(); ++i) {
            scalars[i].t = ScalarDescriptor::Types::INT32;
            scalars[i].v.s32 = static_cast<int32_t>(scaler_value[i]);
        }
    }};
}

// JitConstants PagedAttentionGeneratorSingleToken::get_jit_constants(const kernel_impl_params& params) const {
//     auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);
//     jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));

//     auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
//     jit.make("Q_STEP", get_q_step(xe_arch, true));
//     auto kv_split_size = get_kv_split_size(xe_arch);
//     jit.make("KV_STEP", kv_split_size.first);
//     jit.make("KV_SPLIT_LEN", kv_split_size.second);

//     const size_t kv_len = get_kv_len(params, PagedAttentionStage::GENERATE);
//     jit.make("KV_LEN", kv_len);

//     return jit;
// }

// Arguments PagedAttentionGeneratorSingleToken::get_arguments_desc(const kernel_impl_params& params) const {
//     Arguments args;

//     const auto desc = params.typed_desc<paged_attention>();
//     // const auto has_scale_input = !desc->scale_val.has_value();
//     const auto has_scores_output = params.output_layouts.size() > 1;

//     OPENVINO_ASSERT(!has_scores_output, "[GPU][CM] PagedAttentionGeneratorSingleToken with scores output is not supported yet");

//     args.push_back({ArgumentDescriptor::Types::INPUT, 0});  // queries
//     args.push_back({ArgumentDescriptor::Types::INPUT, 3});  // keys cache
//     args.push_back({ArgumentDescriptor::Types::INPUT, 4});  // values cache

//     // TODO: HAS_ATTN_MASK_INPUT
//     args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, split_output_idx});      // split output
//     args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, split_output_idx + 1});  // lse output

//     args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // q_len==1
//     args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // kv_len

//     return args;
// }

// DispatchDataFunc PagedAttentionGeneratorSingleToken::get_dispatch_data_func() const {
//     return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
//         assert(!params.is_dynamic());
//         auto& wgs = kd.params.workGroups;
//         const auto desc = params.typed_desc<paged_attention>();

//         auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

//         const size_t batch = params.input_layouts[0].get_partial_shape()[0].get_length();
//         const size_t heads_num = desc->heads_num;
//         const size_t split_num = get_split_num(params, rtp->stage);
//         wgs.global = {batch, heads_num, split_num};
//         wgs.local = {1, 1, WG_SIZE};

//         // generate stage: q_len=1, kv_len=past_len + 1
//         auto& scalars = kd.params.scalars;
//         auto kv_len = rtp->paged_attention_aligned_seq_len;
//         std::vector<size_t> scaler_value = {1, kv_len};
//         scalars.resize(scaler_value.size());

//         // std::cout << "PagedAttentionGeneratorSingleToken::get_dispatch_data_func: "
//         //           << "batch: " << batch << ", heads_num: " << heads_num << ", split_num: " << split_num << ", kv_len: " << kv_len << std::endl;

//         for (size_t i = 0; i < scaler_value.size(); ++i) {
//             scalars[i].t = ScalarDescriptor::Types::INT32;
//             scalars[i].v.s32 = static_cast<int32_t>(scaler_value[i]);
//         }
//     }};
// }

// JitConstants PagedAttentionGeneratorSingleTokenFinalization::get_jit_constants(const kernel_impl_params& params) const {
//     auto jit = PagedAttentionGeneratorBase::get_jit_constants(params);

//     const auto desc = params.typed_desc<paged_attention>();
//     jit.make("KV_SPLIT_DATA_SIZE", kv_split_data_size);
//     auto xe_arch = params.get_device_info().arch < gpu_arch::xe2 ? 1 : 2;
//     jit.make("KV_SPLIT_LEN", get_kv_split_size(xe_arch).second);

//     // auto key_cache_shape = params.input_layouts[3].get_shape();
//     // const size_t kv_len = key_cache_shape[0] * key_cache_shape[key_cache_shape.size() - 2];
//     const size_t kv_len = get_kv_len(params, PagedAttentionStage::GENERATE);
//     jit.make("KV_LEN", kv_len);

//     return jit;
// }

// Arguments PagedAttentionGeneratorSingleTokenFinalization::get_arguments_desc(const kernel_impl_params& params) const {
//     Arguments args;

//     args.push_back({ArgumentDescriptor::Types::INPUT, 5});  // past_lens
//     args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

//     const auto has_scores_output = params.output_layouts.size() > 1;

//     OPENVINO_ASSERT(!has_scores_output, "[GPU][CM] PagedAttentionGeneratorSingleTokenFinalization with scores output is not supported yet");

//     args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, split_output_idx});      // split data
//     args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});                              // output
//     args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, split_output_idx + 1});  // values cache

//     return args;
// }

// DispatchDataFunc PagedAttentionGeneratorSingleTokenFinalization::get_dispatch_data_func() const {
//     return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
//         assert(!params.is_dynamic());
//         auto& wgs = kd.params.workGroups;
//         auto& scalars = kd.params.scalars;
//         scalars.resize(1);

//         const auto desc = params.typed_desc<paged_attention>();
//         // auto rtp = static_cast<PagedAttentionRuntimeParams*>(rt_params);

//         const size_t batch = params.input_layouts[0].get_partial_shape()[0].get_length();
//         const size_t heads_num = desc->heads_num;
//         const size_t head_size = desc->k_head_size;

//         wgs.global = {batch, heads_num, head_size / kv_split_data_size};
//         wgs.local = {1, 1, 1};
//     }};
// }

}  // namespace ov::intel_gpu::cm
#endif