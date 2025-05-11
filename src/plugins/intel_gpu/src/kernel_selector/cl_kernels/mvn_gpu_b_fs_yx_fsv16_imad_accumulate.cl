// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


// ==============================================================================================================================
// DECLARE_PACKED_ACCUMULATE(Name, AccT, InputT, SliceSize, SlicePitch, Items, Workers, AccOp)
// DECLARE_PACKED_ACCUMULATE_EARGS(Name, AccT, InputT, SliceSize, SlicePitch, Items, Workers, AccOp, ExtraArgsDecl, ExtraArgs)
//
// Declares function "Name" performing parallel packed accumulation:
// AccT<SliceSize> Name (const __global InputT* input, uint offset, uint worker_id  ExtraArgsDecl)
//
// Template arguments:
//   Name             - Name of function to declare.
//   AccT             - Type of accumulator variable. Can't be vector type. Examples: int, float, half.
//   InputT           - Type of input data. Can't be vector type. Examples: int, float, half.
//   SliceSize        - Number values in packed slice to accumulate in each work-item. One of: 2, 4, 8, 16.
//   SlicePitch       - Pitch between consecutive input slices in "input".
//   Items            - Total number of items to accumulate across all work-items.
//   Workers          - Number of work-items performing accumulation.
//   AccOp              Name of operation used to perform accumulation.
//                      Calling it "function-like" must return value of new accumulation variable.
//                      Expected interface:
//                          AccT AccOp(AccT current, InputT val, uint index ExtraArgs)
//                          current - current accumulation value
//                          val - currently processed input value
//                          index - number of item inside slice currently processed
//                          ExtraArgs - optional extra arguments passed as is from template argument
//                          returns: new accumulator value after accumulating "val" with "current"
//   ExtraArgsDecl    - Optional extra arguments declaration to pass to function.
//   ExtraArgs        - Optional extra arguments to pass to "AccOp" using names declared in "ExtraArgsDecl".
//
// Function arguments:
//   input          - Pointer to global memory from which values will be read to accumulate
//   offset         - Offset into "input" from where accumulation should start
//   worker_id      - Number of current work-item
//   ExtraArgsDecl  - Optional extra arguments, declared from template argument.
//
// Pseduocode:
//  function Name(input, offset, worker_id, ExtraArgs... eargs) {
//      AccT<SliceSize> accumulator = 0;
//      for (uint idx = worker_id; idx < Items; idx += Workers) {
//          InputT<SliceSize> in = vload<SliceSize>(0, &input[offset + idx * SlicePitch];
//          for (uint si = 0; si < SliceSize; ++si) {
//              accumulator[si] = AccOp(accumulator[si], in[si], si, eargs...)
//          }
//      }
//      return accumulator;
//  }
//
// ==============================================================================================================================

#define ACCUMULATE_SUM(a, b, idx)       ((a) + (b))
#define ACCUMULATE_SUM_SQ(a, b, idx)    ((a) + ((b) * (b)))

#define DECLARE_PACKED_ACCUMULATE_EARGS(Name, AccT, InputT, SliceSize, SlicePitch, Items, Workers, AccOp, ExtraArgsDecl, ExtraArgs)     \
inline MAKE_VECTOR_TYPE(AccT, SliceSize) FUNC(Name)(const __global InputT* input,                                                       \
                                                    uint offset,                                                                        \
                                                    uint worker_id                                                                      \
                                                    ExtraArgsDecl) {                                                                    \
    typedef MAKE_VECTOR_TYPE(InputT, SliceSize) packed_in_t;                                                                            \
    typedef MAKE_VECTOR_TYPE(AccT, SliceSize) packed_acc_t;                                                                             \
                                                                                                                                        \
    packed_acc_t acc = 0;  /* Accumulation variable */                                                                                  \
                                                                                                                                        \
    uint input_offset = offset + worker_id * (SlicePitch);  /* Current input offset */                                                  \
                                                                                                                                        \
    /* Uniform loop to help compiler in unrolling */                                                                                    \
    for (uint spatial_idx = 0; spatial_idx < (Items) / (Workers); ++spatial_idx) {                                                      \
        packed_in_t in_pack = ((const __global packed_in_t*)(input + input_offset))[0];                                                 \
                                                                                                                                        \
        input_offset += (Workers) * (SlicePitch);                                                                                       \
                                                                                                                                        \
        __attribute__((opencl_unroll_hint))                                                                                             \
        for (uint set_idx = 0; set_idx < (SliceSize); ++set_idx) {                                                                      \
            acc[set_idx] = AccOp(acc[set_idx], in_pack[set_idx], set_idx  ExtraArgs);                                                   \
        }                                                                                                                               \
    }                                                                                                                                   \
                                                                                                                                        \
    /* [constexpr] Number of leftovers after all uniform iterations */                                                                  \
    const uint leftovers = (Items) % (Workers);                                                                                         \
                                                                                                                                        \
    if (leftovers > 0 && worker_id < leftovers) {                                                                                       \
        packed_in_t in_pack = ((const __global packed_in_t*)(input + input_offset))[0];                                                 \
                                                                                                                                        \
        __attribute__((opencl_unroll_hint))                                                                                             \
        for (uint set_idx = 0; set_idx < (SliceSize); ++set_idx) {                                                                      \
            acc[set_idx] = AccOp(acc[set_idx], in_pack[set_idx], set_idx  ExtraArgs);                                                   \
        }                                                                                                                               \
    }                                                                                                                                   \
                                                                                                                                        \
    return acc;                                                                                                                         \
}

#define DECLARE_PACKED_ACCUMULATE(Name, AccT, InputT, SliceSize, SlicePitch, Items, Workers, AccOp)                                     \
    DECLARE_PACKED_ACCUMULATE_EARGS(Name, AccT, InputT, SliceSize, SlicePitch, Items, Workers, AccOp, , )
