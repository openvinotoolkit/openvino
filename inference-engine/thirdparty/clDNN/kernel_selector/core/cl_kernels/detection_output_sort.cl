// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "include/include_all.cl"
#include "include/detection_output_common.cl"

UNIT_TYPE FUNC(get_score_sort)(__global UNIT_TYPE* input_bboxes, const uint idx_bbox, const uint idx_image)
{
    if (idx_bbox == KEEP_BBOXES_NUM)
    {
        // Idx set to dummy value, return -1 to exclude this element from sorting
        return -1;
    }
    else
    {
        return input_bboxes[(idx_bbox + idx_image * NUM_OF_IMAGE_BBOXES) * OUTPUT_ROW_SIZE + INPUT_OFFSET + SCORE_OFFSET];
    }
}

KERNEL (detection_output_sort)(__global UNIT_TYPE* input_bboxes, __global UNIT_TYPE* output)
{
    __local uint indexes[NUM_CLASSES_IN];
    __local bool stillSorting;
    __local uint output_count;
    __local uint num_out_per_class[NUM_CLASSES_IN];

    output_count = 0;
    num_out_per_class[get_local_id(0)] = 0;

    const uint image_id = (uint)get_global_id(0) / NUM_CLASSES_IN;
    const uint local_id = (uint)get_local_id(0) * NUM_OF_ITEMS_SORT; // All bboxes from one image in work group

    uint image_offset_input = image_id * NUM_OF_IMAGE_BBOXES;

    uint count_sum = 0;
    for (uint i = 0; i < image_id; i++)
    {
        count_sum += (input_bboxes[i] < KEEP_TOP_K)? input_bboxes[i] : KEEP_TOP_K;
    }

    uint image_offset_output = count_sum * OUTPUT_ROW_SIZE;

    // If there is less elements than needed, write input to output
    if (input_bboxes[image_id] <= KEEP_TOP_K)
    {
        if (local_id == 0)
        {
            for (uint class = 0; class < NUM_CLASSES_IN; class++)
            {
                if (class == BACKGROUND_LABEL_ID && !HIDDEN_CLASS)
                {
                    continue;
                }
                for (uint i = 0; i < NUM_OF_CLASS_BBOXES; i++)
                {
                    uint input_idx = (i + image_offset_input + class * NUM_OF_CLASS_BBOXES) * OUTPUT_ROW_SIZE + INPUT_OFFSET;
                    if (input_bboxes[input_idx] != -1)
                    {
                        uint out_idx = output_count * OUTPUT_ROW_SIZE + image_offset_output;

                        for (uint idx = 0; idx < OUTPUT_ROW_SIZE; idx++)
                        {
                            output[out_idx + idx] = input_bboxes[input_idx + idx];
                        }

                        output_count++;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }
    }
    else
    {
        uint sorted_output[KEEP_TOP_K * NUM_CLASSES_IN];

        for (uint it = 0; it < NUM_OF_ITEMS_SORT; it++)
        {
            indexes[local_id + it] = (local_id + it) * NUM_OF_CLASS_BBOXES;
        }

        while (output_count < KEEP_BBOXES_NUM)
        {
            stillSorting = true;

            while(stillSorting)
            {
                barrier(CLK_LOCAL_MEM_FENCE);
                stillSorting = false;
                for (uint it = 0; it < NUM_OF_ITEMS_SORT; it++)
                {
                    uint item_id = local_id + it;
                    for (uint i = 0; i < 2; i++)
                    {

                        uint idx1 = indexes[item_id];
                        uint idx2 = indexes[item_id+1];
                        bool perform = false;
                        if ((((i % 2) && (item_id % 2)) ||
                            ((!(i % 2)) && (!(item_id % 2)))) &&
                            (item_id != (NUM_CLASSES_IN - 1)))
                        {
                            perform = true;
                        }

                        if (perform &&
                            (FUNC_CALL(get_score_sort)(input_bboxes, idx1, image_id) <
                             FUNC_CALL(get_score_sort)(input_bboxes, idx2, image_id)))
                        {
                            indexes[item_id] = idx2;
                            indexes[item_id+1] = idx1;
                            stillSorting = true;
                        }
                        barrier(CLK_LOCAL_MEM_FENCE);
                    }
                }
            }

            if (local_id == 0)
            {
                UNIT_TYPE top_score = FUNC_CALL(get_score_sort)(input_bboxes, indexes[0], image_id);

                if (top_score != 0)
                {
                    for (uint it = 0; (it < NUM_CLASSES_IN) && (output_count < KEEP_BBOXES_NUM); it++)
                    {
                        if (FUNC_CALL(get_score_sort)(input_bboxes, indexes[it], image_id) == top_score)
                        {
                            // write to output, create counter, and check if keep_top_k is satisfied.
                            uint input_idx = (indexes[it] + image_offset_input) * OUTPUT_ROW_SIZE + INPUT_OFFSET;
                            uint class_idx = input_bboxes[input_idx + 1] - HIDDEN_CLASS;

                            sorted_output[class_idx * KEEP_TOP_K + num_out_per_class[class_idx]] = input_idx;
                            num_out_per_class[class_idx]++;

                            indexes[it]++;
                            output_count++;

                            // If all class elements are written to output, set dummy value to exclude class from sorting.
                            if ((indexes[it] % NUM_OF_CLASS_BBOXES) == 0)
                            {
                                indexes[it] = KEEP_BBOXES_NUM;
                            }
                        }
                    }
                }
                else
                {
                    // There is no more significant results to sort.
                    output_count = KEEP_BBOXES_NUM;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (local_id == 0)
        {
            output_count = 0;
            for (uint i = 0; i < NUM_CLASSES_IN; i++)
            {
                for (uint j = 0; j < num_out_per_class[i]; j++)
                {

                    uint out_idx = output_count * OUTPUT_ROW_SIZE + image_offset_output;
                    for (uint idx = 0; idx < OUTPUT_ROW_SIZE; idx++)
                    {
                        output[out_idx + idx] = input_bboxes[sorted_output[i * KEEP_TOP_K + j] + idx];
                    }
                    output_count++;
                }
           }
           uint image_count_sum = (input_bboxes[image_id] < KEEP_TOP_K)? input_bboxes[image_id] : KEEP_TOP_K;
           for (output_count; output_count < image_count_sum; output_count++)
           {
                uint out_idx = output_count * OUTPUT_ROW_SIZE + image_offset_output;
                output[out_idx] = -1.0;
                output[out_idx + 1] = 0.0;
                output[out_idx + 2] = 0.0;
                output[out_idx + 3] = 0.0;
                output[out_idx + 4] = 0.0;
                output[out_idx + 5] = 0.0;
                output[out_idx + 6] = 0.0;
           }
        }
    }

    if (local_id == 0 &&
        image_id == (NUM_IMAGES - 1))
    {
        for (output_count += count_sum; output_count < (KEEP_TOP_K *  NUM_IMAGES); output_count++ )
        {
            uint out_idx = output_count * OUTPUT_ROW_SIZE;
            output[out_idx] = -1.0;
            output[out_idx + 1] = 0.0;
            output[out_idx + 2] = 0.0;
            output[out_idx + 3] = 0.0;
            output[out_idx + 4] = 0.0;
            output[out_idx + 5] = 0.0;
            output[out_idx + 6] = 0.0;
        }
    }

}
