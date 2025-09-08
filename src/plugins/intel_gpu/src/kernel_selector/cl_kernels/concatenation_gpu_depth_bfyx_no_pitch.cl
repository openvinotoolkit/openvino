// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/fetch_data.cl"

//
// In this kernel we are processing "fyx" as flatten 1D "elements".
// As long as we can we use block read/write.
// For last SIMD in which we have to write only partial data we use normal read/write to buffer.
//

// must be 8 as long as we use block_read8/write8
#define ELEMENTS_PER_WORK_ITEM 8
#define WORK_GROUP_SIZE 16
#define INPUT0_ELEMENTS_COUNT (INPUT0_LENGTH/INPUT0_BATCH_NUM)

__attribute__((reqd_work_group_size(1, WORK_GROUP_SIZE, 1)))
REQD_SUB_GROUP_SIZE(WORK_GROUP_SIZE)
KERNEL(concatenation_gpu_depth_bfyx_no_pitch)(__global INPUT0_TYPE* input, __global OUTPUT_TYPE* output, uint output_offset_in_concat_axis)
{
    const uint batch_id = get_group_id(0);

    // Which pack of 16*8 elements we are processing.
    uint element_group_id = get_group_id(1);
    uint element_offset = (uint)get_global_id(1) * ELEMENTS_PER_WORK_ITEM;

    const uint element_group_offset = element_group_id * WORK_GROUP_SIZE * ELEMENTS_PER_WORK_ITEM;

    const uint input_offset = INPUT0_OFFSET + element_group_offset + batch_id * INPUT0_BATCH_PITCH;
    const uint output_batch_offset = batch_id * OUTPUT_BATCH_PITCH;
    const uint output_offset = OUTPUT_OFFSET + element_group_offset + output_batch_offset + output_offset_in_concat_axis*OUTPUT_PITCHES[CONCAT_AXIS_INDEX];

    //Check if current group in batch starts from 16-byte aligned pos. If not then move block read to 16-byte aligned position.
    //Requirement for _sub_group_block_write8.
    uint align_offset = 0;
    const uint group_start_pos = output_offset;
    if(group_start_pos % WORK_GROUP_SIZE != 0)
    {
        uint next_aligned_pos = group_start_pos / WORK_GROUP_SIZE * WORK_GROUP_SIZE + WORK_GROUP_SIZE;
        align_offset = next_aligned_pos - group_start_pos;
    }

    if(element_group_offset + align_offset + WORK_GROUP_SIZE * ELEMENTS_PER_WORK_ITEM < INPUT0_ELEMENTS_COUNT)
    {
        MAKE_VECTOR_TYPE(INPUT0_TYPE, 8) in = DT_INPUT_BLOCK_READ8(input, input_offset + align_offset);
        DT_OUTPUT_BLOCK_WRITE8(output, output_offset + align_offset, ACTIVATION(in, ACTIVATION_PARAMS));

        //Fill the values that were missed upon adding align_offset
        if((align_offset != 0) && (element_offset + output_batch_offset < group_start_pos + align_offset))
        {
            for(uint i = 0; i < align_offset; i++)
                output[output_offset + i] = ACTIVATION(input[input_offset + i], ACTIVATION_PARAMS);
        }
    }
    else
    {
        // This is the last SIMD that needs to write only partial data.
        uint element_offset_in_workitem = element_offset - element_group_offset;
        for(uint i = 0; i < ELEMENTS_PER_WORK_ITEM; i++)
        {
            if(element_offset + i >= INPUT0_ELEMENTS_COUNT)
                return;

            output[output_offset + element_offset_in_workitem] = ACTIVATION(input[input_offset + element_offset_in_workitem], ACTIVATION_PARAMS);
            element_offset_in_workitem++;
        }
    }
}

#undef INPUT0_ELEMENTS_COUNT
#undef WORK_GROUP_SIZE
#undef ELEMENTS_PER_WORK_ITEM
#undef ALIGNED_BLOCK_READ8
#undef ALIGNED_BLOCK_WRITE8
