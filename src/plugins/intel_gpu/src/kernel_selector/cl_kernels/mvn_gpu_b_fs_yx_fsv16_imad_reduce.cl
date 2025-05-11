// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

// ==============================================================================================================================
// DECLARE_SG_PACKED_REDUCE_ADD(Name, Type, VecSize, PostOp)
// DECLARE_WG_PACKED_REDUCE_ADD(Name, Type, VecSize, SgNum, PostOp)
//
// Declares function "Name" performing work-group reduction on vector data, using addition operator:
//   Type Name (Type<VecSize> value, __local Type* slm_acc)
// Returns reduction result as sub-group vector, for example when VecSize equals 4:
//   work-item for which get_sub_group_local_id() == 0 will hold reduced values from value.s0
//   work-item for which get_sub_group_local_id() == 1 will hold reduced values from value.s1
//   work-item for which get_sub_group_local_id() == 2 will hold reduced values from value.s2
//   work-item for which get_sub_group_local_id() == 3 will hold reduced values from value.s2
//  for other work-items in sub-group the result will be undefined.
// All work-items in sub-group must enter declared function.
//
// DECLARE_SG_PACKED_REDUCE_ADD - declares function with same behaviour, but specialized for case with single sub-group
// and not using local memory. It is declared as:
//   Type Name (Type<VecSize> value)
//
// Template arguments:
//   Name    - Name of function to declare.
//   Type    - Type of values to reduce.  Can't be vector type. Examples: int, float, half.
//   VecSize - Vector size of input, one of 2,4,8,16. Must be smaller or equal to sub-group size.
//   SgNum   - Number of sub-groups inside work-group.
//   PostOp  - Operation to perform on reduced values.
//             Called as PostOp(value), where "value" is reduction result, and call should evaluate to expression returning final result.
//
// Function arguments:
//   value   - vector of "VecSize" elements of "Type" holding values to reduce.
//   slm_acc - pointer to local memory used for reduction. Must have size of at least ("SgNum" - 1) * "VecSize".
//
// Pseudocode:
//  function Name(value, slm_acc) {
//      Type result;
//      for (uint vi = 0; vi < VecSize; ++vi) {
//          Type tmp = work_group_reduce_add(value[vi]);
//          if (get_sub_group_local_id() == vi) {
//              result = tmp;
//          }
//      }
//      return result;
// }
//
// Notes:
//   If local memory is going to be reused additiona barrier(CLK_LOCAL_MEM_FENCE) is required to ensure that all usage inside
//   declared function has finished.
// ==============================================================================================================================

#define REDUCE_NO_POST_OP(val) (val)

#define DECLARE_SG_PACKED_REDUCE_ADD(Name, Type, VecSize, PostOp)                                                       \
    inline Type FUNC(Name) (MAKE_VECTOR_TYPE(Type, VecSize) value) {                                                    \
        typedef MAKE_VECTOR_TYPE(Type, VecSize) packed_t;                                                               \
                                                                                                                        \
        Type result;                                                                                                    \
                                                                                                                        \
        /* [uniform] Current sub-groups id */                                                                           \
        const uint sgid = get_sub_group_id();                                                                           \
        /* Id of work-item inside sub-group */                                                                          \
        const uint sglid = get_sub_group_local_id();                                                                    \
        /* [constexpr] Maximum simd/sub-group size */                                                                   \
        const uint simd = get_max_sub_group_size();                                                                     \
                                                                                                                        \
        /* Accumulation inside sub-group */                                                                             \
        packed_t acc;  /* [uniform] Accumulator variable */                                                             \
        __attribute__((opencl_unroll_hint))                                                                             \
        for (uint idx = 0; idx < VecSize; ++idx) {                                                                      \
            acc[idx] = sub_group_reduce_add(value[idx]);                                                                \
        }                                                                                                               \
        /* Transpose the data to correct layout */                                                                      \
        if (sglid < VecSize || simd == VecSize) {                                                                       \
            result = PostOp(acc[sglid]);                                                                                \
        }                                                                                                               \
        return result;                                                                                                  \
    }

#define DECLARE_WG_PACKED_REDUCE_ADD(Name, Type, VecSize, SgNum, PostOp)                                                \
    inline Type FUNC(Name) (MAKE_VECTOR_TYPE(Type, VecSize) value, __local Type* slm_acc) {                             \
        typedef MAKE_VECTOR_TYPE(Type, VecSize) packed_t;                                                               \
                                                                                                                        \
        Type result;                                                                                                    \
                                                                                                                        \
        /* [uniform] Current sub-groups id */                                                                           \
        const uint sgid = get_sub_group_id();                                                                           \
        /* Id of work-item inside sub-group */                                                                          \
        const uint sglid = get_sub_group_local_id();                                                                    \
        /* [constexpr] Maximum simd/sub-group size */                                                                   \
        const uint simd = get_max_sub_group_size();                                                                     \
                                                                                                                        \
        /* Accumulation inside sub-group */                                                                             \
        packed_t acc;  /* [uniform] Accumulator variable */                                                             \
        __attribute__((opencl_unroll_hint))                                                                             \
        for (uint idx = 0; idx < VecSize; ++idx) {                                                                      \
            acc[idx] = sub_group_reduce_add(value[idx]);                                                                \
        }                                                                                                               \
        /* More than one sub-group in work-group, reduce using local memory */                                          \
        /* Store partial results into local memory from sub-groups other than first one */                              \
        if (sgid != 0 && (sglid < VecSize || simd == VecSize)) {                                                        \
            slm_acc[(sgid - 1) * VecSize + sglid] = acc[sglid];                                                         \
        }                                                                                                               \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                                   \
        /* Accumulate partial results inside first sub-group */                                                         \
        if (sgid == 0) {                                                                                                \
            __attribute__((opencl_unroll_hint))                                                                         \
            for (uint vi = 0; vi < VecSize; ++vi) {                                                                     \
                /* Accumulate single vector element using sub_group_reduce_add */                                       \
                /* Last work-item inside sub-group holds previous value (iteration or sub-group reduction stage) */     \
                                                                                                                        \
                Type tmp = acc[vi];                                                                                     \
                __attribute__((opencl_unroll_hint))                                                                     \
                for (uint sg = 0; sg < (SgNum) - 1; sg += (simd - 1)) {                                                 \
                    bool last_sglid = sglid == simd - 1;                                                                \
                    bool sglid_inside_sgs = sg + simd - 1 <= (SgNum) - 1 || sg + sglid < (SgNum) - 1;                   \
                    Type tmp_in_slm = slm_acc[sg * VecSize + sglid * VecSize + vi];                                     \
                    tmp = last_sglid ? tmp :                                                                            \
                          sglid_inside_sgs ? tmp_in_slm                                                                 \
                          : 0;                                                                                          \
                    tmp = sub_group_reduce_add(tmp);                                                                    \
                }                                                                                                       \
                acc[vi] = tmp;                                                                                          \
            }                                                                                                           \
            if (sglid < VecSize || simd == VecSize) {                                                                   \
                result = PostOp(acc[sglid]);                                                                            \
                slm_acc[sglid] = result;                                                                                \
            }                                                                                                           \
        }                                                                                                               \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                                   \
        /* Read result in all other sub-groups */                                                                       \
        if (sgid != 0 && (sglid < VecSize || simd == VecSize)) {                                                        \
            result = slm_acc[sglid];                                                                                    \
        }                                                                                                               \
        return result;                                                                                                  \
    }
