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
//  - FP16_SUPPORTED       - [0/1] Value indicating whether device supports FP16 OpenCL extension (cl_khr_fp16).
//  - FP16_UNIT_USED       - [0/1] Value indicating that current kernel should use FP16.
//  - UNIT_TYPE            - Type of unit of input/output/weight/bias.
//  - UNIT_VAL_ZERO        - Literal of current UNIT_TYPE that represents 0.
//  - INPUT_BATCH_NUM      - [int] Batch size for input. Number of input sets of spatial and feature data that
//                           are grouped to be processed in single batch.
//  - INPUT_ELEMENTS_COUNT - [int] Cumulative number of elements in single data set from batch.
//  - FILTER_OFM_NUM       - [int] Cumulative number of elements that are outputted for single input set from batch.
//                           Number of layer responses per single input set from batch.
//  - RELU                 - [0/1] Indicates that ReLU activation function should be used on output.
//  - NEGATIVE_SLOPE       - [float] Factor for negative output values (required when ReLU is specified).
//
//  - SUB_GROUP_SIZE       - [int] Size of used subgroup (SIMD).
//  - WORK_ITEMS_PER_BATCH - [int] Number of work items needed to process at least one element in all data sets
//                           from batch.
//  - UNIT_BYTE_SIZE       - [int] Size of unit of input/output/weight/bias in bytes.
//  - CHUNK_TYPE           - Type of chunk of data read by work item using sub-group operation.
//  - CHUNK_BYTE_SIZE      - [int] Size of chunk of data read by work item using sub-group operation in bytes.
//  - UNITS_PER_CHUNK      - [int] Number of units stored in single chunk of read data.
//  - BYTES_PER_SG_READ    - [int] Number of bytes read by single sub-group read operation (read by entire sub-group).
//  - UNITS_PER_SG_READ    - [int] Number of units read by single sub-group read operation (read by entire sub-group).
//  - RG_COUNT             - [int] Number of response groups. Each group (except last) writes units_per_sg_read
//                           responses for at least one input data set from batch.
//  - LAST_RG_SIZE         - [int] Number of responses in last group of written responses.
//                           Responses are grouped in UNITS_PER_SG_READ-sized groups. The parameter describes how
//                           many responses are in last group or 0, if group is full.

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
#if UNITS_PER_CHUNK == 2
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

// Currently block read is 4 bytes aligned.
#define ALIGNED_BLOCK_READ(ptr, byte_offset) _sub_group_block_read((const __global CHUNK_TYPE*)((const __global char*)(ptr) + (byte_offset)))

// Currently read is 4 bytes aligned.
#define ALIGNED_READ(ptr, byte_offset) (*(const __global CHUNK_TYPE*)((const __global char*)(ptr) + (byte_offset)))

// Currently block write is 16 bytes aligned.
#define ALIGNED_BLOCK_WRITE(ptr, byte_offset, val) _sub_group_block_write((__global CHUNK_TYPE*)((__global char*)(ptr) + (byte_offset)), (val))

// Currently block write is 4 bytes aligned.
#define ALIGNED_WRITE(ptr, byte_offset, val) ((void)(*(__global CHUNK_TYPE*)((__global char*)(ptr) + (byte_offset)) = (val)))

// Depends on batch size (aligned to greatest power of 2 which divides INPUT0_BATCH_NUM).
#define INPUT0_READ(ptr, byte_offset) ALIGNED_READ(ptr, byte_offset)
// Depends on number of responses (aligned to greatest power of 2 which divides FILTER_OFM_NUM).
#define FILTER_READ(ptr, byte_offset) ALIGNED_READ(ptr, byte_offset)
// Aligned to BYTES_PER_SG_READ.
#define BIAS_READ(ptr, byte_offset) ALIGNED_READ(ptr, byte_offset)
// Depends on batch size (aligned to greatest power of 2 which divides INPUT0_BATCH_NUM).
#define OUTPUT_WRITE(ptr, byte_offset, val) ALIGNED_WRITE(ptr, byte_offset, val)


/*
#if FILTER_OFM_NUM % (2 * SUB_GROUP_SIZE) == 0 || (!FP16_UNIT_USED && FILTER_OFM_NUM % SUB_GROUP_SIZE == 0)
    #define FILTER_READ(ptr, byte_offset) ALIGNED_BLOCK_READ(ptr, byte_offset)
#elifs
    #define FILTER_READ(ptr, byte_offset) ALIGNED_BLOCK_READ(ptr, byte_offset)
#elif FILTER_OFM_NUM % 8 == 0
#else
#endif




#if FP16_UNIT_USED
    #define ALIGNED_FILTER_BLOCK_READ(ptr, byte_offset) as_half2(_sub_group_block_read((const __global uint*)((const __global char*)(ptr) + (byte_offset))))
    #define FILTER_TYPE half2
#else
    #define ALIGNED_FILTER_BLOCK_READ(ptr, byte_offset) as_float(_sub_group_block_read((const __global uint*)((const __global char*)(ptr) + (byte_offset))))
    #define FILTER_TYPE float
#endif
*/


#if INPUT0_BATCH_NUM > 0 && INPUT0_BATCH_NUM % (SUB_GROUP_SIZE * CHUNK_BYTE_SIZE / UNIT_BYTE_SIZE) == 0
#else
    #error Kernel does not support specified input batch size.
#endif



REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
KERNEL (fully_connected_gpu_xb_xb_block_fp16)(
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
    const uint input_batch_byte_size       = INPUT0_BATCH_NUM * UNIT_BYTE_SIZE;
    const uint input_byte_size             = INPUT0_ELEMENTS_COUNT * input_batch_byte_size;
    const uint input_yxf_elems_per_sg_read = INPUT0_BATCH_NUM < UNITS_PER_SG_READ
                                               ? UNITS_PER_SG_READ / INPUT0_BATCH_NUM
                                               : 1;
    const uint input_sg_reads_distance     = WORK_ITEMS_PER_BATCH * BYTES_PER_SG_READ;

    // Size in bytes of all responses for single spatial/feature data point (the same as filter_yxf_elems_distance).
    // Distance between two nearest xyf elements with the same response id.
    const uint filter_response_byte_size = FILTER_OFM_NUM * UNIT_BYTE_SIZE;
    // Cumulative size in bytes of all weights/filters.
    const uint filters_byte_size         = INPUT0_ELEMENTS_COUNT * filter_response_byte_size;

    const uint output_batch_byte_size = input_batch_byte_size;
    const uint output_byte_size = FILTER_OFM_NUM * output_batch_byte_size;

// ---------------------------------------------------------------------------------------------------------------------

    // non-constexpr:
    // Identifier of processing sub-group (each sub-group process UNITS_PER_SG_READ output responses for at least
    // one data set in batch).
    const uint sg_id          = get_group_id(0);
    // Identifier of batch group (each batch group process up to UNITS_PER_SG_READ data sets from batch).
    const uint batch_group_id = get_global_id(1);
    // Identifier of work item element in processing sub-group.
    const uint sg_elem_id     = get_sub_group_local_id();

    // Input base offset in bytes (yxfb/xb format of input).
    const uint input_base     = batch_group_id * BYTES_PER_SG_READ;

    // Filter base offset in bytes (yxfb/xb format of weights).
    const uint filter_base    = sg_id * BYTES_PER_SG_READ;

    // Filter base offset in bytes (x/f format of biases).
#if BIAS_TERM
    const uint bias_base = filter_base;
#endif
    // Output base offset in bytes (xb format of output). INPUT0_BATCH_NUM is the same as OUTPUT_BATCH_NUM.
    const uint output_base    = (sg_id * INPUT0_BATCH_NUM + batch_group_id) * BYTES_PER_SG_READ;

    // Filter/input byte offsets in sub-group used duering read/write operations.
    const uint sg_elem_offset = sg_elem_id * CHUNK_BYTE_SIZE;


    // Accumulator over batch and response elements.
    CHUNK_TYPE acc[UNITS_PER_SG_READ] = {};

    // Iterate over yxf linear plane (both filters/weights and input).
    uint input_offset = input_base;
    uint filter_offset = filter_base;
    do {
        CHUNK_TYPE input_val = INPUT0_READ(input, input_offset + sg_elem_offset);

        // Iterate over filters needed to process input read by sub-group.
        for(uint elem_idx = 0; elem_idx < input_yxf_elems_per_sg_read; ++elem_idx)
        {
            CHUNK_TYPE filter_val = FILTER_READ(weight, filter_offset + sg_elem_offset);
            filter_offset += filter_response_byte_size;

            // MULTIPLY
            // BATCH = 32x? (HF) / 16x? (F)
            // Iterate over output features (indexed by acc_pos). acc[i] accumulates entire batch group for output feature i.
            __attribute__((opencl_unroll_hint(UNITS_PER_SG_READ)))
            for (uint acc_pos = 0; acc_pos < UNITS_PER_SG_READ; ++acc_pos)
            {
                acc[acc_pos] = AS_CHUNK(fma(AS_UNITS(input_val), SG_UNIT_SELECT(filter_val, acc_pos), AS_UNITS(acc[acc_pos])));
            }
        }

        input_offset += input_sg_reads_distance;
    } while (input_offset < input_byte_size);

    // WRITE OUTPUT
    // BATCH = 32x? (HF) / 16x? (F)
#if LAST_RG_SIZE > 0
    if (sg_id < RG_COUNT - 1)
#endif
    {
#if BIAS_TERM
        CHUNK_TYPE bias_val = BIAS_READ(bias, bias_base + sg_elem_offset);
#endif
        uint output_offset = output_base;
        __attribute__((opencl_unroll_hint(UNITS_PER_SG_READ)))
        for (uint acc_pos = 0; acc_pos < UNITS_PER_SG_READ; ++acc_pos)
        {
#if BIAS_TERM
            CHUNK_UNITS_TYPE output_val = AS_UNITS(acc[acc_pos]) + SG_UNIT_SELECT(bias_val, acc_pos);
#else
            CHUNK_UNITS_TYPE output_val = AS_UNITS(acc[acc_pos]);
#endif
            output_val = ACTIVATION(output_val, ACTIVATION_PARAMS);
            OUTPUT_WRITE(output, output_offset + sg_elem_offset, AS_CHUNK(output_val));
            output_offset += output_batch_byte_size;
        }
    }
#if LAST_RG_SIZE > 0
    else
    {
#if BIAS_TERM
        CHUNK_TYPE bias_val = BIAS_READ(bias, bias_base + sg_elem_offset);
#endif

        uint output_offset = output_base;
        __attribute__((opencl_unroll_hint(LAST_RG_SIZE)))
        for (uint acc_pos = 0; acc_pos < LAST_RG_SIZE; ++acc_pos)
        {
#if BIAS_TERM
            CHUNK_UNITS_TYPE output_val = AS_UNITS(acc[acc_pos]) + SG_UNIT_SELECT(bias_val, acc_pos);
#else
            CHUNK_UNITS_TYPE output_val = AS_UNITS(acc[acc_pos]);
#endif
            output_val = ACTIVATION(output_val, ACTIVATION_PARAMS);
            OUTPUT_WRITE(output, output_offset + sg_elem_offset, AS_CHUNK(output_val));
            output_offset += output_batch_byte_size;
        }
    }
#endif
}

#undef CONCAT_TOKEN_HANDLER1
#undef CONCAT_TOKEN
#undef CVT_UNIT
#undef CHUNK_UNITS_TYPE
#undef AS_CHUNK
#undef AS_UNITS
#undef CHUNK_UNIT_SELECT

#undef SG_UNIT_SELECT
#undef ALIGNED_BLOCK_READ
#undef ALIGNED_BLOCK_WRITE
#undef INPUT0_READ
#undef FILTER_READ
#undef BIAS_READ
#undef OUTPUT_WRITE
