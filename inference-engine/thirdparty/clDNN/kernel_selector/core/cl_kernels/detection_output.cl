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

KERNEL (detection_output)(__global UNIT_TYPE* input_location, __global UNIT_TYPE* output, __global UNIT_TYPE* input_confidence, __global UNIT_TYPE* input_prior_box)
{
    const uint idx = get_global_id(0);              // bbox idx
    const uint local_id = (uint)get_local_id(0) * NUM_OF_ITEMS; // All bboxes from one image in work group
    const uint idx_image = idx / NUM_OF_ITERATIONS;  // idx of current image

    __local uint indexes[NUM_OF_PRIORS];
    __local uint scores_size[NUM_CLASSES * NUM_OF_IMAGES];
    __local bool stillSorting;

    uint indexes_class_0[NUM_OF_PRIORS];

    int last_bbox_in_class = NUM_OF_ITEMS;
    bool is_last_bbox_in_class = false;
    for (uint it = 0; it < NUM_OF_ITEMS; it ++)
    {
        if (((local_id + it + 1) % NUM_OF_PRIORS) == 0 )
        {
            last_bbox_in_class = it;
            is_last_bbox_in_class = true;
            break;
        }
    }

    for (uint idx_class = 0; idx_class < NUM_CLASSES; idx_class++)
    {
        if (idx_class == BACKGROUND_LABEL_ID)
        {
            continue;
        }

        for (uint it = 0;  it < NUM_OF_ITEMS; it++)
        {
            indexes[local_id + it] = local_id + it; 
        }

        stillSorting = true;
        barrier(CLK_LOCAL_MEM_FENCE);

        bool is_last_bbox_in_image = (is_last_bbox_in_class) && (idx_class == (NUM_CLASSES - 1));

        while(stillSorting)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            stillSorting = false;

            for (uint i = 0; i < 2; i++)
            {
                for (uint it = 0; it < NUM_OF_ITEMS; it++)
                {
                    uint item_id = local_id + it;
     
                    uint idx1 = indexes[item_id];
                    uint idx2 = indexes[item_id+1];
                    bool perform = false;
                    if ((((i % 2) && (item_id % 2)) ||
                        ((!(i % 2)) && (!(item_id % 2)))) &&
                        (it < last_bbox_in_class))
                    {
                        perform = true;
                    }

                    if (perform &&
                        (FUNC_CALL(get_score)(input_confidence, idx1, idx_class, idx_image) <
                         FUNC_CALL(get_score)(input_confidence, idx2, idx_class, idx_image)))
                    {
                        indexes[item_id] = idx2;
                        indexes[item_id+1] = idx1;
                        stillSorting = true;
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
        }

        // Do it only once per class in image
        if (is_last_bbox_in_class)
        {
            UNIT_TYPE adaptive_threshold = NMS_THRESHOLD;
            uint post_nms_count = 0;
            const uint shared_class = (SHARE_LOCATION)? 0 : idx_class;
            scores_size[idx_class] = 0;

            // Do the "keep" algorithm only for classes with confidence greater than CONFIDENCE_THRESHOLD.
            // Check first, the biggest one (after sort) element in class.
            if (FUNC_CALL(get_score)(input_confidence, indexes[0], idx_class, idx_image) != 0.0f)
            {
                for (uint i = 0; i < SCORES_COUNT; i++)
                {
                    const uint bb_idx = indexes[i];
                    bool keep = true;
                    for (uint j = 0; j < post_nms_count; j++)
                    {
                        if (!keep)
                        {
                            break;
                        }

                        UNIT_TYPE overlap = 0.0;
                        const uint bb_idx2 = indexes[j];

                        UNIT_TYPE decoded_bbox1[4];
                        FUNC_CALL(get_decoded_bbox)(decoded_bbox1, input_location, input_prior_box, bb_idx, shared_class, idx_image);
                        UNIT_TYPE decoded_bbox2[4];
                        FUNC_CALL(get_decoded_bbox)(decoded_bbox2, input_location, input_prior_box, bb_idx2, shared_class, idx_image);
                        bool intersecting =
                            (decoded_bbox1[0] < decoded_bbox2[2]) &
                            (decoded_bbox2[0] < decoded_bbox1[2]) &
                            (decoded_bbox1[1] < decoded_bbox2[3]) &
                            (decoded_bbox2[1] < decoded_bbox1[3]);

                        if (intersecting)
                        {
                            const UNIT_TYPE intersect_width = min(decoded_bbox1[2], decoded_bbox2[2]) - max(decoded_bbox1[0], decoded_bbox2[0]);
                            const UNIT_TYPE intersect_height = min(decoded_bbox1[3], decoded_bbox2[3]) - max(decoded_bbox1[1], decoded_bbox2[1]);
                            const UNIT_TYPE intersect_size = intersect_width * intersect_height;
                            const UNIT_TYPE bbox1_area = (decoded_bbox1[2] - decoded_bbox1[0]) * (decoded_bbox1[3] - decoded_bbox1[1]);
                            const UNIT_TYPE bbox2_area = (decoded_bbox2[2] - decoded_bbox2[0]) * (decoded_bbox2[3] - decoded_bbox2[1]);
                            overlap = intersect_size / (bbox1_area + bbox2_area - intersect_size);
                        }
                        keep = (overlap <= adaptive_threshold);
                    }
                    if (keep)
                    {
                        indexes[post_nms_count] = indexes[i];
                        ++post_nms_count;
                    }
                    if ((keep) && (ETA < 1) && (adaptive_threshold > 0.5))
                    {
                        adaptive_threshold *= ETA;
                    }
                }
            }
            // Write number of scores to global memory, for proper output order in separated work groups
            scores_size[idx_class] = post_nms_count;
        }

        stillSorting = true;
        // Wait for scores number from all classes in images
        barrier(CLK_LOCAL_MEM_FENCE);

        uint output_offset = (idx_image * NUM_CLASSES_OUT + idx_class - HIDDEN_CLASS) * SCORES_COUNT;

        for (uint it = 0; it < NUM_OF_ITEMS; it++)
        {
            const uint local_id_out = local_id + it;
            
            if (local_id_out < scores_size[idx_class])
            {
                const uint score_idx = indexes[local_id_out];
                uint bb_idx = indexes[local_id_out];
                const uint shared_class = (SHARE_LOCATION)? 0 : idx_class;
                UNIT_TYPE decoded_bbox[4];
                FUNC_CALL(get_decoded_bbox)(decoded_bbox, input_location, input_prior_box, bb_idx, shared_class, idx_image);

                const uint out_idx = (local_id_out + output_offset) * OUTPUT_ROW_SIZE + OUTPUT_OFFSET;
                output[out_idx] = TO_UNIT_TYPE(idx_image);
                output[out_idx + 1] = TO_UNIT_TYPE(idx_class);
                output[out_idx + 2] = FUNC_CALL(get_score)(input_confidence, score_idx, idx_class, idx_image);
                output[out_idx + 3] = decoded_bbox[0];
                output[out_idx + 4] = decoded_bbox[1];
                output[out_idx + 5] = decoded_bbox[2];
                output[out_idx + 6] = decoded_bbox[3];
            }
        }

        // If work item is processing last bbox in image (we already know the number of all detections),
        // use it to fill rest of keep_top_k items if number of detections is smaller
        if (is_last_bbox_in_class)
        {
            uint out_idx = output_offset + scores_size[idx_class];

            uint current_top_k = output_offset + SCORES_COUNT;
            for (uint i = out_idx; i < current_top_k; i++)
            {
                out_idx = i * OUTPUT_ROW_SIZE + OUTPUT_OFFSET;
                output[out_idx] = -1.0;
                output[out_idx + 1] = 0.0;
                output[out_idx + 2] = 0.0;
                output[out_idx + 3] = 0.0;
                output[out_idx + 4] = 0.0;
                output[out_idx + 5] = 0.0;
                output[out_idx + 6] = 0.0;
            }
        }

        // Write number of scores kept in first step of detection output
        if (is_last_bbox_in_image)
        {
            uint scores_sum = 0;
            for (uint i = 0; i < NUM_CLASSES; i++)
            {
                scores_sum += scores_size[i];
            }
            output[idx_image] = scores_sum;

        }
    }
}
