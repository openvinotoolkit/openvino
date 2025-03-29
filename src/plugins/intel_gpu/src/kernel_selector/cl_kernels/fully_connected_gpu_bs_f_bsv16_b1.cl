// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"

// ---------------------------------------------------------------------------------------------------------------------
// Just-in-time macro definitions:
// ---------------------------------------------------------------------------------------------------------------------

// Required JIT constants:
//  - FP16_SUPPORTED        - [0/1] Value indicating whether device supports FP16 OpenCL extension (cl_khr_fp16).
//  - FP16_UNIT_USED        - [0/1] Value indicating that current kernel should use FP16.
//  - UNIT_TYPE             - Type of unit of input/output/weight/bias.
//  - UNIT_VAL_ZERO         - Literal of current UNIT_TYPE that represents 0.
//  - INPUT0_BATCH_NUM      - [int] Batch size for input. Number of input sets of spatial and feature data that
//                                  are grouped to be processed in single batch.
//  - INPUT0_ELEMENTS_COUNT - [int] Cumulative number of elements in single data set from batch.
//  - FILTER_OFM_NUM        - [int] Cumulative number of elements that are outputted for single input set from batch.
//                           Number of layer responses per single input set from batch.
//  - RELU                  - [0/1] Indicates that ReLU activation function should be used on output.
//  - NEGATIVE_SLOPE        - [float] Factor for negative output values (required when ReLU is specified).
//
//  - SUB_GROUP_SIZE        - [int] Size of used subgroup (SIMD).
//  - UNIT_BYTE_SIZE        - [int] Size of unit of input/output/weight/bias in bytes.
//  - CHUNK_TYPE            - Type of chunk of data read by work item using sub-group operation (OpenCL scalar type).
//  - CHUNK_BYTE_SIZE       - [int] Size of chunk of data read by work item using sub-group operation in bytes.
//  - UNITS_PER_CHUNK       - [int] Number of units stored in single chunk of read data.
//                                  Must be equal CHUNK_BYTE_SIZE / UNIT_BYTE_SIZE (and this division must not have
//                                  remainder). Added as helper for manual loop unrolling.
//  - BYTES_PER_SG_READ     - [int] Number of bytes read by single sub-group read operation (read by entire sub-group).
//                                  Must be equal (CHUNK_BYTE_SIZE * SUB_GROUP_SIZE). Added as helper for manual loop
//                                  unrolling.
//  - UNITS_PER_SG_READ     - [int] Number of units read by single sub-group read operation (read by entire sub-group).
//                                  Must be equal (UNIT_BYTE_SIZE * SUB_GROUP_SIZE). Added as helper for manual loop
//                                  unrolling.
//
//  - RESPONSES_PER_SG_EXEC      - [int] Number of neural responses processed/executed by single sub-group.
//  - IN_CHUNK_PREFETCH_SIZE     - [int] Size of array of CHUNK_TYPE use to cache/prefetch input data.
//  - FILTER_CHUNK_PREFETCH_SIZE - [int] Size of array of CHUNK_TYPE use to cache/prefetch filter/weights data.

// ---------------------------------------------------------------------------------------------------------------------
// Helpers:
// ---------------------------------------------------------------------------------------------------------------------

#define CONCAT_TOKEN_HANDLER1(prefix, suffix) prefix##suffix

// Expands and concatenates two tokens into one.
#define CONCAT_TOKEN(prefix, suffix) CONCAT_TOKEN_HANDLER1(prefix, suffix)

// ---------------------------------------------------------------------------------------------------------------------

// Converts scalar expression to scalar of unit type.
#if FP16_UNIT_USED
    #define CVT_UNIT(expression) CONCAT_TOKEN(convert_, UNIT_TYPE)(expression)
#else
    #define CVT_UNIT(expression) (expression)
#endif

// ---------------------------------------------------------------------------------------------------------------------

// - CHUNK_UNITS_TYPE - Type of scalar or vector of UNIT_TYPE that can be reinterpreted as CHUNK_TYPE.
#if UNITS_PER_CHUNK == 1
    #define CHUNK_UNITS_TYPE UNIT_TYPE
#else
    #define CHUNK_UNITS_TYPE MAKE_VECTOR_TYPE(UNIT_TYPE, UNITS_PER_CHUNK)
#endif

// ---------------------------------------------------------------------------------------------------------------------

// Reinterpretation between CHUNK_TYPE and CHUNK_UNITS_TYPE.
#define AS_CHUNK(expression) CONCAT_TOKEN(as_, CHUNK_TYPE)(expression)
#define AS_UNITS(expression) CONCAT_TOKEN(as_, CHUNK_UNITS_TYPE)(expression)

// ---------------------------------------------------------------------------------------------------------------------

// Extracts one scalar element of UNIT_TYPE from work-item chunk;
//     chunk - name of chunk variable, idx - 0-based index of element.
#if UNITS_PER_CHUNK == 4
    #define CHUNK_UNIT_SELECT(chunk, idx) ((idx) > 1 ? ((idx) > 2 ? AS_UNITS(chunk).s3 : AS_UNITS(chunk).s2) : ((idx) ? AS_UNITS(chunk).s1 : AS_UNITS(chunk).s0))
#elif UNITS_PER_CHUNK == 2
    #define CHUNK_UNIT_SELECT(chunk, idx) ((idx) ? AS_UNITS(chunk).s1 : AS_UNITS(chunk).s0)
#elif UNITS_PER_CHUNK == 1
    #define CHUNK_UNIT_SELECT(chunk, idx) AS_UNITS(chunk)
#else
    #error Unsupported number of units per chunk.
#endif

// ---------------------------------------------------------------------------------------------------------------------
// Sub-group operations:
// ---------------------------------------------------------------------------------------------------------------------

// Extracts one scalar element of UNIT_TYPE from sub-group chunk;
//     chunk - name of chunk variable, idx - 0-based index of element.
#define SG_UNIT_SELECT(chunk, idx) CHUNK_UNIT_SELECT(_sub_group_shuffle(chunk, (idx) / UNITS_PER_CHUNK), (idx) % UNITS_PER_CHUNK)

// ---------------------------------------------------------------------------------------------------------------------
// Reads / Writes:
// ---------------------------------------------------------------------------------------------------------------------

// Type of chunk salar/vector returned by or passed to read/write macros.
#define CHUNK_VEC1_TYPE CHUNK_TYPE
#define CHUNK_VEC2_TYPE MAKE_VECTOR_TYPE(CHUNK_TYPE, 2)
#define CHUNK_VEC4_TYPE MAKE_VECTOR_TYPE(CHUNK_TYPE, 4)
#define CHUNK_VEC8_TYPE MAKE_VECTOR_TYPE(CHUNK_TYPE, 8)

// Expands vector of chunks to array of chunks (using order of components);
//     array - name of chunk array variable, idx - 0-based start index in array where vector should be expanded,
//     chunk_vec - vector to expand.
#define EXPAND_CHUNK_VEC1_TO_CHUNK_ARRAY(array, idx, chunk_vec) ((void)((array)[(idx)] = chunk_vec))
#define EXPAND_CHUNK_VEC2_TO_CHUNK_ARRAY(array, idx, chunk_vec) ((void)((array)[(idx)] = chunk_vec.s0, (array)[(idx) + 1] = chunk_vec.s1))
#define EXPAND_CHUNK_VEC4_TO_CHUNK_ARRAY(array, idx, chunk_vec) ((void)((array)[(idx)]     = chunk_vec.s0, (array)[(idx) + 1] = chunk_vec.s1, \
                                                                        (array)[(idx) + 2] = chunk_vec.s2, (array)[(idx) + 3] = chunk_vec.s3))
#define EXPAND_CHUNK_VEC8_TO_CHUNK_ARRAY(array, idx, chunk_vec) ((void)((array)[(idx)]     = chunk_vec.s0, (array)[(idx) + 1] = chunk_vec.s1, \
                                                                        (array)[(idx) + 2] = chunk_vec.s2, (array)[(idx) + 3] = chunk_vec.s3, \
                                                                        (array)[(idx) + 4] = chunk_vec.s4, (array)[(idx) + 5] = chunk_vec.s5, \
                                                                        (array)[(idx) + 6] = chunk_vec.s6, (array)[(idx) + 7] = chunk_vec.s7))

// Currently block read is 4 bytes aligned.
#define ALIGNED_BLOCK_READ1(ptr, byte_offset) _sub_group_block_read((const __global CHUNK_TYPE*)((const __global char*)(ptr) + (byte_offset)))
#define ALIGNED_BLOCK_READ2(ptr, byte_offset) _sub_group_block_read2((const __global CHUNK_TYPE*)((const __global char*)(ptr) + (byte_offset)))
#define ALIGNED_BLOCK_READ4(ptr, byte_offset) _sub_group_block_read4((const __global CHUNK_TYPE*)((const __global char*)(ptr) + (byte_offset)))
#define ALIGNED_BLOCK_READ8(ptr, byte_offset) _sub_group_block_read8((const __global CHUNK_TYPE*)((const __global char*)(ptr) + (byte_offset)))

// Currently read is 4 bytes aligned.
#define ALIGNED_READ1(ptr, byte_offset) (*(const __global CHUNK_TYPE*)((const __global char*)(ptr) + (byte_offset)))
#define ALIGNED_READ2(ptr, byte_offset) vload2(0, (const __global CHUNK_TYPE*)((const __global char*)(ptr) + (byte_offset)))
#define ALIGNED_READ4(ptr, byte_offset) vload4(0, (const __global CHUNK_TYPE*)((const __global char*)(ptr) + (byte_offset)))
#define ALIGNED_READ8(ptr, byte_offset) vload8(0, (const __global CHUNK_TYPE*)((const __global char*)(ptr) + (byte_offset)))

// Currently block write is 16 bytes aligned.
#define ALIGNED_BLOCK_WRITE1(ptr, byte_offset, val) _sub_group_block_write((__global CHUNK_TYPE*)((__global char*)(ptr) + (byte_offset)), (val))
#define ALIGNED_BLOCK_WRITE2(ptr, byte_offset, val) _sub_group_block_write2((__global CHUNK_TYPE*)((__global char*)(ptr) + (byte_offset)), (val))
#define ALIGNED_BLOCK_WRITE4(ptr, byte_offset, val) _sub_group_block_write4((__global CHUNK_TYPE*)((__global char*)(ptr) + (byte_offset)), (val))
#define ALIGNED_BLOCK_WRITE8(ptr, byte_offset, val) _sub_group_block_write8((__global CHUNK_TYPE*)((__global char*)(ptr) + (byte_offset)), (val))

// Currently block write is 4 bytes aligned.
#define ALIGNED_WRITE1(ptr, byte_offset, val) ((void)(*(__global CHUNK_TYPE*)((__global char*)(ptr) + (byte_offset)) = (val)))
#define ALIGNED_WRITE2(ptr, byte_offset, val) vstore2((val), 0, (__global CHUNK_TYPE*)((__global char*)(ptr) + (byte_offset)))
#define ALIGNED_WRITE4(ptr, byte_offset, val) vstore4((val), 0, (__global CHUNK_TYPE*)((__global char*)(ptr) + (byte_offset)))
#define ALIGNED_WRITE8(ptr, byte_offset, val) vstore8((val), 0, (__global CHUNK_TYPE*)((__global char*)(ptr) + (byte_offset)))



// Kernel-specific JIT requirements.
#if INPUT0_BATCH_NUM != 1
    #error Kernel does not support specified input batch size.
#endif
#if UNITS_PER_SG_READ <= 0 || RESPONSES_PER_SG_EXEC <= 0 || UNITS_PER_CHUNK <= 0 || UNITS_PER_SG_READ % RESPONSES_PER_SG_EXEC != 0 || RESPONSES_PER_SG_EXEC % UNITS_PER_CHUNK != 0
    #error Kernel does not support specified number of responses processed by single sub-group.
#endif
#if IN_CHUNK_PREFETCH_SIZE <= 0 || FILTER_CHUNK_PREFETCH_SIZE <= 0 || (IN_CHUNK_PREFETCH_SIZE * RESPONSES_PER_SG_EXEC) % FILTER_CHUNK_PREFETCH_SIZE != 0
    #error Kernel does not support specified prefetch sizes.
#endif



REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
KERNEL (fully_connected_gpu_bx_bs_f_bsv16_b1)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* weight
#if BIAS_TERM
    , __global UNIT_TYPE* bias)
#else
    )
#endif
{
    // constexpr:
    const uint input_byte_size  = INPUT0_ELEMENTS_COUNT * UNIT_BYTE_SIZE;

    const uint output_size      = FILTER_OFM_NUM;

    // Identifier of work item element in processing sub-group.
    const uint sg_elem_id       = get_sub_group_local_id();

// ---------------------------------------------------------------------------------------------------------------------

    // non-constexpr:
    // Identifier of processing sub-group (each sub-group process RESPONSES_PER_SG_EXEC output responses).
    const uint sg_id          = get_group_id(0);

    // Input base offset in bytes (bfyx/bx format of input).
    const uint input_base     = 0;

    // Filter base offset in bytes (bs_f_bsv16 format of weights).
    const uint filter_base    = sg_id * input_byte_size * RESPONSES_PER_SG_EXEC;

    // [SCATTERED] Output base identifier/element offset to use (bx format of output).
    const uint output_base_id = sg_id * RESPONSES_PER_SG_EXEC + sg_elem_id;
#if BIAS_TERM
    // [SCATTERED] Bias base identifier/element offset to use (x/f format of biases).
    const uint bias_base_id = output_base_id;
#endif //
    // Filter/input byte offsets in sub-group used duering read/write operations.
    const uint sg_elem_offset = sg_elem_id * CHUNK_BYTE_SIZE;


    // Accumulator for fully connected. Contains one or more sum of multiples on yxf plane. If there is more than one it needs to be sum up to single one.
    CHUNK_TYPE acc = 0;

    uint input_offset = input_base;   // Non-scattered offset (all work items in sub-group must have the same value, so the loop will not diverge in sub-group).
    uint filter_offset = filter_base; // Non-scattered (to support different sub-group scatter in remainder processing and avoid TPM issue during shuffling).
    while (input_offset + IN_CHUNK_PREFETCH_SIZE * BYTES_PER_SG_READ <= input_byte_size)
    {
        // Contains chached IN_CHUNK_PREFETCH_SIZE * UNITS_PER_SG_READ input elements.
        // Currently consecutive fyx data elements are stored in consecutive work-items in sub-group (elems in array seen from work-item are offseted by UNITS_PER_SG_READ).
        CHUNK_TYPE input_val[IN_CHUNK_PREFETCH_SIZE];

#if IN_CHUNK_PREFETCH_SIZE % 8 == 0
        unroll_for(uint input_val_idx = 0; input_val_idx < IN_CHUNK_PREFETCH_SIZE; input_val_idx += 8)
        {
            CHUNK_VEC8_TYPE input_vals = ALIGNED_BLOCK_READ8(input, input_offset + 8 * sg_elem_offset);
            input_offset += 8 * BYTES_PER_SG_READ;
            EXPAND_CHUNK_VEC8_TO_CHUNK_ARRAY(input_val, input_val_idx, input_vals);
        }
#elif IN_CHUNK_PREFETCH_SIZE % 4 == 0
        unroll_for(uint input_val_idx = 0; input_val_idx < IN_CHUNK_PREFETCH_SIZE; input_val_idx += 4)
        {
            CHUNK_VEC4_TYPE input_vals = ALIGNED_BLOCK_READ4(input, input_offset + 4 * sg_elem_offset);
            input_offset += 4 * BYTES_PER_SG_READ;
            EXPAND_CHUNK_VEC4_TO_CHUNK_ARRAY(input_val, input_val_idx, input_vals);
        }
#elif IN_CHUNK_PREFETCH_SIZE % 2 == 0
        unroll_for(uint input_val_idx = 0; input_val_idx < IN_CHUNK_PREFETCH_SIZE; input_val_idx += 2)
        {
            CHUNK_VEC2_TYPE input_vals = ALIGNED_BLOCK_READ2(input, input_offset + 2 * sg_elem_offset);
            input_offset += 2 * BYTES_PER_SG_READ;
            EXPAND_CHUNK_VEC2_TO_CHUNK_ARRAY(input_val, input_val_idx, input_vals);
        }
#else
        unroll_for(uint input_val_idx = 0; input_val_idx < IN_CHUNK_PREFETCH_SIZE; input_val_idx += 1)
        {
            CHUNK_VEC1_TYPE input_vals = ALIGNED_BLOCK_READ1(input, input_offset + sg_elem_offset);
            input_offset += BYTES_PER_SG_READ;
            EXPAND_CHUNK_VEC1_TO_CHUNK_ARRAY(input_val, input_val_idx, input_vals);
        }
#endif

        unroll_for(uint elem_base_idx = 0; elem_base_idx < IN_CHUNK_PREFETCH_SIZE * UNITS_PER_SG_READ; elem_base_idx += FILTER_CHUNK_PREFETCH_SIZE * UNITS_PER_SG_READ / RESPONSES_PER_SG_EXEC)
        {
            // Contains group of weights for RESPONSES_PER_SG_EXEC responses and for (FILTER_CHUNK_PREFETCH_SIZE * UNITS_PER_SG_READ / RESPONSES_PER_SG_EXEC) spatial points.
            // Currently for floats:
            // sub-group-id |        0 |        1 |        2 | ... |        15
            // -------------+----------+----------+----------+-----+----------
            // [0]          | (s0, r0) | (s0, r1) | (s0, r2) | ... | (s0, r15)
            // [1]          | (s1, r0) | (s1, r1) | (s1, r2) | ... | (s1, r15)
            // [2]          | (s2, r0) | (s2, r1) | (s2, r2) | ... | (s2, r15)
            // ...          |   ...    |   ...    |   ...    | ... |   ...
            // Currently for halfs:
            // sub-group-id |          0 |          1 |          2 | ... |            7 |          8 |          9 |         10 | ... |           15
            // -------------+------------+------------+------------+-----+--------------+------------+------------+------------+-----+-------------
            // [0]          | (s0, r0-1) | (s0, r2-3) | (s0, r4-5) | ... | (s0, r14-15) | (s1, r0-1) | (s1, r2-3) | (s1, r4-5) | ... | (s1, r14-15)
            // [1]          | (s2, r0-1) | (s2, r2-3) | (s2, r4-5) | ... | (s2, r14-15) | (s3, r0-1) | (s3, r2-3) | (s3, r4-5) | ... | (s3, r14-15)
            // [2]          | (s4, r0-1) | (s4, r2-3) | (s4, r4-5) | ... | (s4, r14-15) | (s5, r0-1) | (s5, r2-3) | (s5, r4-5) | ... | (s5, r14-15)
            // ...          |    ...     |    ...     |    ...     | ... |     ...      |    ...     |    ...     |    ...     | ... |     ...
            CHUNK_TYPE filter_val[FILTER_CHUNK_PREFETCH_SIZE];

#if FILTER_CHUNK_PREFETCH_SIZE % 8 == 0
            unroll_for (uint filter_val_idx = 0; filter_val_idx < FILTER_CHUNK_PREFETCH_SIZE; filter_val_idx += 8)
            {
                CHUNK_VEC8_TYPE filter_vals = ALIGNED_BLOCK_READ8(weight, filter_offset + 8 * sg_elem_offset);
                filter_offset += 8 * BYTES_PER_SG_READ;
                EXPAND_CHUNK_VEC8_TO_CHUNK_ARRAY(filter_val, filter_val_idx, filter_vals);
            }
#elif FILTER_CHUNK_PREFETCH_SIZE % 4 == 0
            unroll_for (uint filter_val_idx = 0; filter_val_idx < FILTER_CHUNK_PREFETCH_SIZE; filter_val_idx += 4)
            {
                CHUNK_VEC4_TYPE filter_vals = ALIGNED_BLOCK_READ4(weight, filter_offset + 4 * sg_elem_offset);
                filter_offset += 4 * BYTES_PER_SG_READ;
                EXPAND_CHUNK_VEC4_TO_CHUNK_ARRAY(filter_val, filter_val_idx, filter_vals);
            }
#elif FILTER_CHUNK_PREFETCH_SIZE % 2 == 0
            unroll_for (uint filter_val_idx = 0; filter_val_idx < FILTER_CHUNK_PREFETCH_SIZE; filter_val_idx += 2)
            {
                CHUNK_VEC2_TYPE filter_vals = ALIGNED_BLOCK_READ2(weight, filter_offset + 2 * sg_elem_offset);
                filter_offset += 2 * BYTES_PER_SG_READ;
                EXPAND_CHUNK_VEC2_TO_CHUNK_ARRAY(filter_val, filter_val_idx, filter_vals);
            }
#else
            unroll_for (uint filter_val_idx = 0; filter_val_idx < FILTER_CHUNK_PREFETCH_SIZE; filter_val_idx += 1)
            {
                CHUNK_VEC1_TYPE filter_vals = ALIGNED_BLOCK_READ1(weight, filter_offset + sg_elem_offset);
                filter_offset += BYTES_PER_SG_READ;
                EXPAND_CHUNK_VEC1_TO_CHUNK_ARRAY(filter_val, filter_val_idx, filter_vals);
            }
#endif

            // Processing of cached filter chunks.
            unroll_for (uint filter_val_idx = 0; filter_val_idx < FILTER_CHUNK_PREFETCH_SIZE; ++filter_val_idx)
            {
                const uint input_base_elem_idx = elem_base_idx + filter_val_idx * UNITS_PER_SG_READ / RESPONSES_PER_SG_EXEC;

                // Select different input for every SUB_GROUP_SIZE * RESPONSES_PER_SG_EXEC / UNITS_PER_SG_READ work-items in sub-group.
                // This code is suboptimal because get_sub_group_local_id() is not treated as constexpr (compiler issue).
#if UNITS_PER_SG_READ / RESPONSES_PER_SG_EXEC == 4
                UNIT_TYPE rearranged_input = sg_elem_id < SUB_GROUP_SIZE / 2
                    ? (sg_elem_id < SUB_GROUP_SIZE / 4
                        ? (SG_UNIT_SELECT(input_val[input_base_elem_idx / UNITS_PER_SG_READ], input_base_elem_idx % UNITS_PER_SG_READ))
                        : (SG_UNIT_SELECT(input_val[(input_base_elem_idx + 1) / UNITS_PER_SG_READ], (input_base_elem_idx + 1) % UNITS_PER_SG_READ)))
                    : (sg_elem_id < 3 * SUB_GROUP_SIZE / 4
                        ? (SG_UNIT_SELECT(input_val[(input_base_elem_idx + 2) / UNITS_PER_SG_READ], (input_base_elem_idx + 2) % UNITS_PER_SG_READ))
                        : (SG_UNIT_SELECT(input_val[(input_base_elem_idx + 3) / UNITS_PER_SG_READ], (input_base_elem_idx + 3) % UNITS_PER_SG_READ)));
#elif UNITS_PER_SG_READ / RESPONSES_PER_SG_EXEC == 2
                UNIT_TYPE rearranged_input = sg_elem_id < SUB_GROUP_SIZE / 2
                    ? (SG_UNIT_SELECT(input_val[input_base_elem_idx / UNITS_PER_SG_READ], input_base_elem_idx % UNITS_PER_SG_READ))
                    : (SG_UNIT_SELECT(input_val[(input_base_elem_idx + 1) / UNITS_PER_SG_READ], (input_base_elem_idx + 1) % UNITS_PER_SG_READ));
#elif UNITS_PER_SG_READ / RESPONSES_PER_SG_EXEC == 1
                UNIT_TYPE rearranged_input = SG_UNIT_SELECT(input_val[input_base_elem_idx / UNITS_PER_SG_READ], input_base_elem_idx % UNITS_PER_SG_READ);
#else
    #error Selected RESPONSES_PER_SG_EXEC is not supported.
#endif

                acc = AS_CHUNK(fma(rearranged_input, AS_UNITS(filter_val[filter_val_idx]), AS_UNITS(acc)));
            }
        }
    }


// Processing input remainder (if needed).
#define INPUT0_ELEMENTS_REMAINDER             (INPUT0_ELEMENTS_COUNT % (IN_CHUNK_PREFETCH_SIZE * UNITS_PER_SG_READ))
#define IN_CHUNK_PREFETCH_REMAINDER_REQ_SIZE ((INPUT0_ELEMENTS_REMAINDER + UNITS_PER_SG_READ - 1) / UNITS_PER_SG_READ)
#if INPUT0_ELEMENTS_REMAINDER != 0

    {
        CHUNK_TYPE input_val[IN_CHUNK_PREFETCH_SIZE];

    #if IN_CHUNK_PREFETCH_SIZE % 8 == 0 && (IN_CHUNK_PREFETCH_REMAINDER_REQ_SIZE % 8 == 0 || IN_CHUNK_PREFETCH_REMAINDER_REQ_SIZE >= 16)
        unroll_for(uint input_val_idx = 0; input_val_idx < IN_CHUNK_PREFETCH_REMAINDER_REQ_SIZE; input_val_idx += 8)
        {
            CHUNK_VEC8_TYPE input_vals = ALIGNED_BLOCK_READ8(input, input_offset + 8 * sg_elem_offset);
            input_offset += 8 * BYTES_PER_SG_READ;
            EXPAND_CHUNK_VEC8_TO_CHUNK_ARRAY(input_val, input_val_idx, input_vals);
        }
    #elif IN_CHUNK_PREFETCH_SIZE % 4 == 0 && (IN_CHUNK_PREFETCH_REMAINDER_REQ_SIZE % 4 == 0 || IN_CHUNK_PREFETCH_REMAINDER_REQ_SIZE >= 8)
        unroll_for(uint input_val_idx = 0; input_val_idx < IN_CHUNK_PREFETCH_REMAINDER_REQ_SIZE; input_val_idx += 4)
        {
            CHUNK_VEC4_TYPE input_vals = ALIGNED_BLOCK_READ4(input, input_offset + 4 * sg_elem_offset);
            input_offset += 4 * BYTES_PER_SG_READ;
            EXPAND_CHUNK_VEC4_TO_CHUNK_ARRAY(input_val, input_val_idx, input_vals);
        }
    #elif IN_CHUNK_PREFETCH_SIZE % 2 == 0 && (IN_CHUNK_PREFETCH_REMAINDER_REQ_SIZE % 2 == 0 || IN_CHUNK_PREFETCH_REMAINDER_REQ_SIZE >= 4)
        unroll_for(uint input_val_idx = 0; input_val_idx < IN_CHUNK_PREFETCH_REMAINDER_REQ_SIZE; input_val_idx += 2)
        {
            CHUNK_VEC2_TYPE input_vals = ALIGNED_BLOCK_READ2(input, input_offset + 2 * sg_elem_offset);
            input_offset += 2 * BYTES_PER_SG_READ;
            EXPAND_CHUNK_VEC2_TO_CHUNK_ARRAY(input_val, input_val_idx, input_vals);
        }
    #else
        unroll_for(uint input_val_idx = 0; input_val_idx < IN_CHUNK_PREFETCH_REMAINDER_REQ_SIZE; input_val_idx += 1)
        {
            CHUNK_VEC1_TYPE input_vals = ALIGNED_BLOCK_READ1(input, input_offset + sg_elem_offset);
            input_offset += BYTES_PER_SG_READ;
            EXPAND_CHUNK_VEC1_TO_CHUNK_ARRAY(input_val, input_val_idx, input_vals);
        }
    #endif

        unroll_for(uint elem_base_idx = 0; elem_base_idx < INPUT0_ELEMENTS_REMAINDER; elem_base_idx += FILTER_CHUNK_PREFETCH_SIZE * UNITS_PER_SG_READ / RESPONSES_PER_SG_EXEC)
        {
            // Size of array of CHUNK_TYPE needed to contain filter elements for input elements in range [elem_base_idx; INPUT0_ELEMENTS_REMAINDER).
            const uint filter_chunk_remainder_size = ((INPUT0_ELEMENTS_REMAINDER - elem_base_idx) * RESPONSES_PER_SG_EXEC + UNITS_PER_SG_READ - 1) / UNITS_PER_SG_READ;
            const uint filter_chunk_prefetch_req_size = filter_chunk_remainder_size < FILTER_CHUNK_PREFETCH_SIZE ? filter_chunk_remainder_size : FILTER_CHUNK_PREFETCH_SIZE;

            CHUNK_TYPE filter_val[FILTER_CHUNK_PREFETCH_SIZE];

#if FILTER_CHUNK_PREFETCH_SIZE % 8 == 0
            unroll_for (uint filter_val_idx = 0; filter_val_idx < filter_chunk_prefetch_req_size; filter_val_idx += 8)
            {
                CHUNK_VEC8_TYPE filter_vals = ALIGNED_BLOCK_READ8(weight, filter_offset + 8 * sg_elem_offset);
                filter_offset += 8 * BYTES_PER_SG_READ;
                EXPAND_CHUNK_VEC8_TO_CHUNK_ARRAY(filter_val, filter_val_idx, filter_vals);
            }
#elif FILTER_CHUNK_PREFETCH_SIZE % 4 == 0
            unroll_for (uint filter_val_idx = 0; filter_val_idx < filter_chunk_prefetch_req_size; filter_val_idx += 4)
            {
                CHUNK_VEC4_TYPE filter_vals = ALIGNED_BLOCK_READ4(weight, filter_offset + 4 * sg_elem_offset);
                filter_offset += 4 * BYTES_PER_SG_READ;
                EXPAND_CHUNK_VEC4_TO_CHUNK_ARRAY(filter_val, filter_val_idx, filter_vals);
            }
#elif FILTER_CHUNK_PREFETCH_SIZE % 2 == 0
            unroll_for (uint filter_val_idx = 0; filter_val_idx < filter_chunk_prefetch_req_size; filter_val_idx += 2)
            {
                CHUNK_VEC2_TYPE filter_vals = ALIGNED_BLOCK_READ2(weight, filter_offset + 2 * sg_elem_offset);
                filter_offset += 2 * BYTES_PER_SG_READ;
                EXPAND_CHUNK_VEC2_TO_CHUNK_ARRAY(filter_val, filter_val_idx, filter_vals);
            }
#else
            unroll_for (uint filter_val_idx = 0; filter_val_idx < filter_chunk_prefetch_req_size; filter_val_idx += 1)
            {
                CHUNK_VEC1_TYPE filter_vals = ALIGNED_BLOCK_READ1(weight, filter_offset + sg_elem_offset);
                filter_offset += BYTES_PER_SG_READ;
                EXPAND_CHUNK_VEC1_TO_CHUNK_ARRAY(filter_val, filter_val_idx, filter_vals);
            }
#endif

            // Processing of cached filter chunks.
            unroll_for (uint filter_val_idx = 0; filter_val_idx < filter_chunk_prefetch_req_size; ++filter_val_idx)
            {
                const uint input_base_elem_idx = elem_base_idx + filter_val_idx * UNITS_PER_SG_READ / RESPONSES_PER_SG_EXEC;

                // Select different input for every SUB_GROUP_SIZE * RESPONSES_PER_SG_EXEC / UNITS_PER_SG_READ work-items in sub-group.
                // This code is suboptimal because get_sub_group_local_id() is not treated as constexpr (compiler issue).
#if UNITS_PER_SG_READ / RESPONSES_PER_SG_EXEC == 4
                UNIT_TYPE rearranged_input = sg_elem_id < SUB_GROUP_SIZE / 2
                    ? (sg_elem_id < SUB_GROUP_SIZE / 4
                        ? (input_base_elem_idx     < INPUT0_ELEMENTS_REMAINDER ? SG_UNIT_SELECT(input_val[input_base_elem_idx       / UNITS_PER_SG_READ], input_base_elem_idx % UNITS_PER_SG_READ)       : UNIT_VAL_ZERO)
                        : (input_base_elem_idx + 1 < INPUT0_ELEMENTS_REMAINDER ? SG_UNIT_SELECT(input_val[(input_base_elem_idx + 1) / UNITS_PER_SG_READ], (input_base_elem_idx + 1) % UNITS_PER_SG_READ) : UNIT_VAL_ZERO))
                    : (sg_elem_id < 3 * SUB_GROUP_SIZE / 4
                        ? (input_base_elem_idx + 2 < INPUT0_ELEMENTS_REMAINDER ? SG_UNIT_SELECT(input_val[(input_base_elem_idx + 2) / UNITS_PER_SG_READ], (input_base_elem_idx + 2) % UNITS_PER_SG_READ) : UNIT_VAL_ZERO)
                        : (input_base_elem_idx + 3 < INPUT0_ELEMENTS_REMAINDER ? SG_UNIT_SELECT(input_val[(input_base_elem_idx + 3) / UNITS_PER_SG_READ], (input_base_elem_idx + 3) % UNITS_PER_SG_READ) : UNIT_VAL_ZERO));
#elif UNITS_PER_SG_READ / RESPONSES_PER_SG_EXEC == 2
                UNIT_TYPE rearranged_input = sg_elem_id < SUB_GROUP_SIZE / 2
                    ? (input_base_elem_idx     < INPUT0_ELEMENTS_REMAINDER ? SG_UNIT_SELECT(input_val[input_base_elem_idx       / UNITS_PER_SG_READ], input_base_elem_idx % UNITS_PER_SG_READ)       : UNIT_VAL_ZERO)
                    : (input_base_elem_idx + 1 < INPUT0_ELEMENTS_REMAINDER ? SG_UNIT_SELECT(input_val[(input_base_elem_idx + 1) / UNITS_PER_SG_READ], (input_base_elem_idx + 1) % UNITS_PER_SG_READ) : UNIT_VAL_ZERO);
#elif UNITS_PER_SG_READ / RESPONSES_PER_SG_EXEC == 1
                UNIT_TYPE rearranged_input = input_base_elem_idx < INPUT0_ELEMENTS_REMAINDER ? SG_UNIT_SELECT(input_val[input_base_elem_idx / UNITS_PER_SG_READ], input_base_elem_idx % UNITS_PER_SG_READ) : UNIT_VAL_ZERO;
#else
    #error Selected RESPONSES_PER_SG_EXEC is not supported.
#endif

                acc = AS_CHUNK(fma(rearranged_input, AS_UNITS(filter_val[filter_val_idx]), AS_UNITS(acc)));
            }
        }
    }

#endif
#undef INPUT0_ELEMENTS_REMAINDER
#undef IN_CHUNK_PREFETCH_REMAINDER_REQ_SIZE


    // Secondary accumulator that will contain final sum (special reducing over work-items in sub-group).
    CHUNK_TYPE reduced_acc = acc;
    CHUNK_TYPE zero = 0;

    for (uint sg_reduce_offset = SUB_GROUP_SIZE * RESPONSES_PER_SG_EXEC / UNITS_PER_SG_READ;
         sg_reduce_offset < SUB_GROUP_SIZE;
         sg_reduce_offset += SUB_GROUP_SIZE * RESPONSES_PER_SG_EXEC / UNITS_PER_SG_READ)
    {
        reduced_acc = AS_CHUNK(AS_UNITS(reduced_acc) + AS_UNITS(_sub_group_shuffle_down(acc, zero, sg_reduce_offset)));
    }


    // Expand accumulator chunks to units.
    const uint expanded_acc_size = (RESPONSES_PER_SG_EXEC + SUB_GROUP_SIZE - 1) / SUB_GROUP_SIZE;

    unroll_for (uint expanded_acc_idx = 0; expanded_acc_idx < expanded_acc_size; ++expanded_acc_idx)
    {
        const uint output_id = output_base_id + expanded_acc_idx * SUB_GROUP_SIZE;
#if BIAS_TERM
        const uint bias_id = bias_base_id + expanded_acc_idx * SUB_GROUP_SIZE;
#endif
        UNIT_TYPE expanded_acc = SG_UNIT_SELECT(reduced_acc, expanded_acc_idx * SUB_GROUP_SIZE + sg_elem_id);

        if (output_id < output_size)
        {
#if BIAS_TERM
            expanded_acc += bias[bias_id];
#endif
            output[output_id] = ACTIVATION(expanded_acc, ACTIVATION_PARAMS);
        }
    }
}

#undef CONCAT_TOKEN_HANDLER1
#undef CONCAT_TOKEN
#undef CVT_UNIT
#undef CHUNK_UNITS_TYPE
#undef AS_CHUNK
#undef AS_UNITS
#undef CHUNK_UNIT_SELECT

#undef SG_UNIT_SELECT
#undef CHUNK_VEC1_TYPE
#undef CHUNK_VEC2_TYPE
#undef CHUNK_VEC4_TYPE
#undef CHUNK_VEC8_TYPE
#undef EXPAND_CHUNK_VEC1_TO_CHUNK_ARRAY
#undef EXPAND_CHUNK_VEC2_TO_CHUNK_ARRAY
#undef EXPAND_CHUNK_VEC4_TO_CHUNK_ARRAY
#undef EXPAND_CHUNK_VEC8_TO_CHUNK_ARRAY
#undef ALIGNED_BLOCK_READ1
#undef ALIGNED_BLOCK_READ2
#undef ALIGNED_BLOCK_READ4
#undef ALIGNED_BLOCK_READ8
#undef ALIGNED_READ1
#undef ALIGNED_READ2
#undef ALIGNED_READ4
#undef ALIGNED_READ8
#undef ALIGNED_BLOCK_WRITE1
#undef ALIGNED_BLOCK_WRITE2
#undef ALIGNED_BLOCK_WRITE4
#undef ALIGNED_BLOCK_WRITE8
#undef ALIGNED_WRITE1
#undef ALIGNED_WRITE2
#undef ALIGNED_WRITE4
#undef ALIGNED_WRITE8
