// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <utility>
#include <vector>

#include "common_utils/dispatch_utils.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/fused_primitive_desc.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_gpu::ocl {

struct FusedOpsConfiguration {
    enum class LoadType { LT_UNALIGNED = 0, LT_ALIGNED_READ = 1, FEATURE_SHUFFLE = 2 };

    enum class BoundaryCheck { DISABLED = 0, ENABLED = 1 };

    enum class IndexType { TENSOR_COORD = 0, LINEAR_OFFSET = 1 };

    // Optional suffix that is added to each macro in the configuration.
    std::string suffix;
    // Indices to load additional data for a fused op.
    std::vector<std::string> bfzyx_idx_order;
    // Name of the input variable for the first fused op.
    std::string input_var_name;
    // Data type of the input
    ov::element::Type input_dt;
    // Data type vector size of the input
    size_t vec_size;
    // Represents a channel in the input tensor that is loaded to the input variable
    ChannelName vec_axis;
    // Sets used load type - aligned or unaligned. Aligned load requires specific extensions and adjusted indices.
    LoadType load_type;
    // Defines if safe index function should be used for offset calculation
    BoundaryCheck boundary_check;
    // Defines how to treat indices array
    IndexType index_type;
    // Defines outer loops channels where fused op is called.
    std::vector<ChannelName> loop_axes;
    // If allow_for_partial_preload is false, then it's required that all fused_ops can be preloaded.
    // If allow_for_partial_preload is true, then not preloaded fused_ops will be loaded in FUSED_OPS_CALC.
    bool allow_for_partial_preload;
    // Load index for shuffle fused op
    std::string shuffle_var_name;
    // Record original output layout before reorder is fused
    cldnn::format orig_output_layout;

    FusedOpsConfiguration(std::string suffix,
                          const std::vector<std::string>& bfzyx_idx_order,
                          std::string input_var_name,
                          ov::element::Type input_dt,
                          size_t vec_size = 1,
                          LoadType load_type = LoadType::LT_UNALIGNED,
                          BoundaryCheck boundary_check = BoundaryCheck::ENABLED,
                          IndexType index_type = IndexType::TENSOR_COORD,
                          ChannelName vec_axis = ChannelName::UNKNOWN,
                          const std::vector<ChannelName>& loop_axes = {},
                          bool allow_for_partial_preload = false,
                          std::string shuffle_var_name = "",
                          cldnn::format orig_output_layout = cldnn::format::any)
        : suffix(std::move(suffix)),
          bfzyx_idx_order(bfzyx_idx_order),
          input_var_name(std::move(input_var_name)),
          input_dt(input_dt),
          vec_size(vec_size),
          vec_axis(vec_axis),
          load_type(load_type),
          boundary_check(boundary_check),
          index_type(index_type),
          loop_axes(loop_axes),
          allow_for_partial_preload(allow_for_partial_preload),
          shuffle_var_name(std::move(shuffle_var_name)),
          orig_output_layout(std::move(orig_output_layout)) {}

    FusedOpsConfiguration& set_vector_type(size_t val) {
        vec_size = val;
        return *this;
    }
    FusedOpsConfiguration& set_load_type(LoadType val) {
        load_type = val;
        return *this;
    }
    FusedOpsConfiguration& set_boundary_check(BoundaryCheck val) {
        boundary_check = val;
        return *this;
    }
    FusedOpsConfiguration& set_index_type(IndexType val) {
        index_type = val;
        return *this;
    }
    FusedOpsConfiguration& set_vector_axis(ChannelName val) {
        vec_axis = val;
        return *this;
    }
    FusedOpsConfiguration& set_loop_axes(std::vector<ChannelName> val, bool partial_preload = false) {
        loop_axes = std::move(val);
        allow_for_partial_preload = partial_preload;
        return *this;
    }

    FusedOpsConfiguration& set_shuffle_var_name(std::string val) {
        shuffle_var_name = std::move(val);
        return *this;
    }
    [[nodiscard]] bool is_post_reorder_fused(void) const {
        return orig_output_layout != cldnn::format::any;
    }

    [[nodiscard]] int get_dim_index_from_order(ChannelName val) const {
        return get_channel_index(val, bfzyx_idx_order.size());
    }
};

// Dependency(Input) type of fusing operation in fused node.
// There are different ways to generate input var name and type by the dependency(input) type in MakeOpJitConstants in jitter
// - ORIGINAL: The input of the operation is the fused node such as Conv
// - EXTERNAL: The input of the operation is the external node outside the fused node
// - INTERNAL: The input of the operation is the another fused operation in the fused node
enum class DependencyType { UNDEFINED = -1, ORIGINAL = 0, EXTERNAL = 1, INTERNAL = 2 };

// Dependency(Input) information of fusing operation which is used to generate input var name and type
// in MakeOpJitConstants in jitter
struct DependencyInfo {
    DependencyType dep_type = DependencyType::UNDEFINED;
    size_t op_id;
    ov::element::Type data_type;
};

// Instance of FusedPrimitiveDesc is added to fused_ops vector if a node has been fused to current one using program::fuse_nodes
// method. In order to process fused ops following modifications should be done in a kernel:
// option 1 - using common generator:
//     - create FusedOpsConfiguration object that contains configuration for common code generator.
//       Multiple objects can be created if a kernel uses different data types at the same time. E.g. kernels that contains scalar and
//       vector branches that are chosen in runtime. To handle this case, create 2 configurations with different suffixes, like
//       "_SCALAR" and "_VEC" and then use generated macros accordingly.
//     - add jit constants returned by KernelBase::MakeFusedOpsJitConstants method to the kernel's constants.
//     - insert generated macros in the ocl code:
//       in kernel declaration:
//         #if HAS_FUSED_OPS_DECLS
//           FUSED_OPS_DECLS,
//         #endif
//       in kernel body:
//         #if HAS_FUSED_OPS
//           FUSED_OPS<OPTIONAL_SUFFIX>;
//           <SOME_VARIABLE> = FUSED_OPS_RESULT<OPTIONAL_SUFFIX>;
//         #endif
//   In this case common generator creates set of definitions for each op which are called sequentially in FUSED_OP<OPTIONAL_SUFFIX>
//   macro. Example:
//     #define FUSED_OPS
//       FUSED_OP0_LOAD_VEC
//       FUSED_OP0_ACTION_VEC
//       FUSED_OP1_LOAD_VEC
//       FUSED_OP1_ACTION_VEC
//     #define FUSED_OP0_LOAD_VEC
//       MAKE_VECTOR_TYPE(FUSED_OP_0_INPUT0_TYPE,2) activation0_data0 = UNIT_BLOCK_READ(activation0_input0,
//                                                                      FUSED_OP_0_INPUT0_GET_INDEX_SAFE(0,(f_block*16),0,0));
//     #define FUSED_OP0_ACTION_VEC
//       float2 dst_0 = dst;
//       dst_0 = ACTIVATION_FUSED_OP0_VEC(dst_0, ACTIVATION_PARAMS_FUSED_OP0_VEC);
//     #define FUSED_OP1_LOAD_VEC
//       MAKE_VECTOR_TYPE(FUSED_OP_1_INPUT0_TYPE,2) eltwise1_data0 = UNIT_BLOCK_READ2(eltwise1_input0,
//                                                                   FUSED_OP_1_INPUT0_GET_INDEX_SAFE(0,(f_block*16),y,x));
//     #define FUSED_OP1_ACTION_VEC
//       float2 dst_0_2 = convert_float2(eltwise1_data0) + convert_float2(dst_0);
//     #define FUSED_OPS_RESULT_VEC dst_0_2
// option 2 - using custom generator in a kernel. It can be used if performance is not optimal in the common one or to handle
//            some difficult cases that can't be unified. Custom processing of fused ops can be written absolutely independently
//            in a kernel, but to make it easier set of helper functions exist:
//     - KernelBase::MakeFusedOpsDeclsJitConstants that creates arguments for kernel declaration and macro for all tensors used in
//       a fused op (requires FusedOpsConfiguration instance).
//     - FusedOpsCodeGenerator contains a bunch of methods to generate variable/pointer names, type conversions, data loads

class FusedOpsCodeGenerator {
public:
    explicit FusedOpsCodeGenerator(const FusedPrimitiveDesc& desc, const RuntimeParams& params, size_t op_idx) : desc(desc), params(params), op_idx(op_idx) {}

    struct IndexDesc {
        std::string b;
        std::string f;
        std::string v;
        std::string u;
        std::string w;
        std::string z;
        std::string y;
        std::string x;
        size_t dims = 0;
        explicit IndexDesc(std::vector<std::string> idx, const cldnn::layout& t)
            : b("0"),
              f("0"),
              v("0"),
              u("0"),
              w("0"),
              z("0"),
              y("0"),
              x("0"),
              dims(idx.size()) {
            switch (dims) {
            case 1:
                f = idx[0];
                break;
            case 2:
                b = idx[0];
                f = idx[1];
                break;
            case 3:
                b = idx[0];
                f = idx[1];
                y = idx[2];
                break;
            case 4:
                b = idx[0];
                f = idx[1];
                y = idx[2];
                x = idx[3];
                break;
            case 5:
                b = idx[0];
                f = idx[1];
                z = idx[2];
                y = idx[3];
                x = idx[4];
                break;
            case 6:
                b = idx[0];
                f = idx[1];
                w = idx[2];
                z = idx[3];
                y = idx[4];
                x = idx[5];
                break;
            case 7:
                b = idx[0];
                f = idx[1];
                u = idx[2];
                w = idx[3];
                z = idx[4];
                y = idx[5];
                x = idx[6];
                break;
            case 8:
                b = idx[0];
                f = idx[1];
                v = idx[2];
                u = idx[3];
                w = idx[4];
                z = idx[5];
                y = idx[6];
                x = idx[7];
                break;
            default:
                throw std::runtime_error("More than 8 dimenstions is not supported in fused op generator");
            }

            const std::map<ChannelName, std::string*> channels_map{
                {ChannelName::BATCH, &b},
                {ChannelName::FEATURE, &f},
                {ChannelName::V, &v},
                {ChannelName::U, &u},
                {ChannelName::W, &w},
                {ChannelName::Z, &z},
                {ChannelName::Y, &y},
                {ChannelName::X, &x},
            };

            for (const auto& [channel_name, dim] : channels_map) {
                if (extract_channel(channel_name, t) == 1) {
                    *dim = "0";
                }
            }
        }
    };

    [[nodiscard]] JitConstants make_fused_tensor_jit_constants(const FusedOpsConfiguration& conf) const;
    [[nodiscard]] JitConstants make_input_decls_jit_constants(const FusedOpsConfiguration& conf) const;
    [[nodiscard]] JitConstants make_load_jit_constants(const FusedOpsConfiguration& conf, const cldnn::layout& prim_output) const;
    [[nodiscard]] JitConstants make_op_jit_constants(const FusedOpsConfiguration& conf,
                                                     const JitTerm& in_var,
                                                     ov::element::Type_t in_type,
                                                     JitTerm& out_var) const;

    [[nodiscard]] bool can_preload_data(const FusedOpsConfiguration& conf) const;

    [[nodiscard]] JitTerm get_op_type() const;
    [[nodiscard]] JitTerm get_input_tensor_name(size_t input_id) const;
    [[nodiscard]] JitTerm get_output_tensor_name() const;
    [[nodiscard]] JitTerm get_jit_load(const FusedOpsConfiguration& conf,
                                       size_t input_id,
                                       const cldnn::layout& prim_output,
                                       bool reuse_index = false,
                                       const JitTerm& reused_idx = {}) const;
    [[nodiscard]] JitTerm get_idx(size_t input_id, const IndexDesc& idx, bool should_be_safe) const;
    [[nodiscard]] JitTerm get_input_ptr_name(size_t input_id) const;
    [[nodiscard]] JitTerm get_input_var_name(size_t input_id, bool is_shuffled = false, const JitTerm& shuffle_var = {}) const;
    [[nodiscard]] JitTerm get_output_var_name(const JitTerm& input_var_name, size_t op_id) const;
    [[nodiscard]] JitTerm convert_to_output_type(const JitTerm& var, size_t vec_size = 1) const;
    [[nodiscard]] JitTerm convert_to_output_type_sat(const JitTerm& var, size_t vec_size = 1) const;
    [[nodiscard]] JitTerm get_output_type(size_t vec_size = 1) const;

private:
    [[nodiscard]] std::vector<size_t> get_required_inputs() const;

    const FusedPrimitiveDesc& desc;
    const RuntimeParams& params;
    const size_t op_idx;
};

JitConstants make_fused_ops_jit_constants(const RuntimeParams& params, const std::vector<FusedOpsConfiguration>& conf);
JitConstants make_activation_jit_constants(const std::string& suffix,
                                           cldnn::activation_func activation_function,
                                           ov::element::Type_t calc_dt,
                                           ov::element::Type_t out_dt);
}  // namespace ov::intel_gpu::ocl
