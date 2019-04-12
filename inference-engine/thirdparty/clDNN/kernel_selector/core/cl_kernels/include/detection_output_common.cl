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

#define PRIOR_BOX_SIZE 4 // Each prior-box consists of [xmin, ymin, xmax, ymax].
#define OUTPUT_ROW_SIZE 7 // Each detection consists of [image_id, label, confidence, xmin, ymin, xmax, ymax].

#define CODE_TYPE_CORNER 0
#define CODE_TYPE_CENTER_SIZE 1
#define CODE_TYPE_CORNER_SIZE 2

#define HIDDEN_CLASS ((BACKGROUND_LABEL_ID == 0 && SHARE_LOCATION)?  1 : 0)
#define NUM_OF_IMAGES INPUT0_BATCH_NUM
#define NUM_LOC_CLASSES ((SHARE_LOCATION)? 1 : NUM_CLASSES)
#define NUM_CLASSES_OUT ((HIDDEN_CLASS == 1)? NUM_CLASSES - 1 : NUM_CLASSES)
#define NUM_OF_PRIORS (INPUT0_LENGTH / (NUM_OF_IMAGES * NUM_LOC_CLASSES * PRIOR_BOX_SIZE))
#define NUM_OF_ITEMS ((NUM_OF_PRIORS / 256) + 1)
#define NUM_OF_ITERATIONS ((NUM_OF_PRIORS % NUM_OF_ITEMS == 0)? (NUM_OF_PRIORS / NUM_OF_ITEMS) : ((NUM_OF_PRIORS / NUM_OF_ITEMS) + 1))

#define X_SIZE INPUT0_Y_PITCH
#define Y_SIZE (INPUT0_FEATURE_PITCH/INPUT0_Y_PITCH)
#define LOCATION_PADDING (INPUT0_PAD_BEFORE_SIZE_Y * X_SIZE + INPUT0_PAD_BEFORE_SIZE_X)
#define LOC_XY_SIZE_PRODUCT (X_SIZE * Y_SIZE)
#define CONF_PADDING (CONF_PADDING_Y * CONF_SIZE_X + CONF_PADDING_X)
#define CONF_XY_SIZE_PRODUCT (CONF_SIZE_X * CONF_SIZE_Y)

#define NUM_OF_PRIOR_COMPONENTS (NUM_OF_PRIORS * PRIOR_BOX_SIZE)
#define NUM_OF_IMAGE_CONF (INPUT0_LENGTH/NUM_OF_IMAGES/PRIOR_BOX_SIZE)

#define SCORES_COUNT (((TOP_K != -1) && (TOP_K < NUM_OF_PRIORS))? TOP_K : NUM_OF_PRIORS)

#define OUTPUT_OFFSET (((NUM_OF_IMAGES + 15) / 16) * 16)
#define SCORE_OFFSET 2

#define INPUT_OFFSET (((NUM_IMAGES + 15) / 16) * 16)
#define INPUT_BBOXES_COUNT ((INPUT0_LENGTH - INPUT_OFFSET) / OUTPUT_ROW_SIZE)
#define NUM_CLASSES_IN NUM_CLASSES_OUT
#define BBOXES_NUM_BASED_TOP_K (TOP_K * NUM_CLASSES_IN * NUM_IMAGES)
#define INPUT_BBOXES_LENGTH (((TOP_K != -1) && (BBOXES_NUM_BASED_TOP_K < INPUT_BBOXES_COUNT))? BBOXES_NUM_BASED_TOP_K : INPUT_BBOXES_COUNT)
#define NUM_OF_CLASS_BBOXES (INPUT_BBOXES_LENGTH / (NUM_IMAGES * NUM_CLASSES_IN))
#define NUM_OF_IMAGE_BBOXES (INPUT_BBOXES_LENGTH / NUM_IMAGES)
#define NUM_OF_ITEMS_SORT ((NUM_CLASSES_IN / 256) + 1)


// Number of bboxes to keep in output
#define KEEP_BBOXES_NUM ((KEEP_TOP_K < NUM_OF_IMAGE_BBOXES)? KEEP_TOP_K : NUM_OF_IMAGE_BBOXES)

void FUNC(get_decoded_bbox)(UNIT_TYPE* decoded_bbox, __global UNIT_TYPE* input_location, __global UNIT_TYPE* input_prior_box, const uint idx_prior, const uint idx_class, const uint idx_image)
{
    const uint prior_offset = idx_prior * PRIOR_INFO_SIZE + PRIOR_COORD_OFFSET;
    uint location_offset =
        (NUM_LOC_CLASSES * (idx_prior * PRIOR_BOX_SIZE) + idx_image * INPUT0_FEATURE_NUM + idx_class * PRIOR_BOX_SIZE) *
        LOC_XY_SIZE_PRODUCT +
        LOCATION_PADDING;

    UNIT_TYPE prior_bboxes[4] = {
        input_prior_box[prior_offset],
        input_prior_box[prior_offset + 1],
        input_prior_box[prior_offset + 2],
        input_prior_box[prior_offset + 3]};

    if (!PRIOR_IS_NORMALIZED)
    {
        prior_bboxes[0] /= IMAGE_WIDTH;
        prior_bboxes[1] /= IMAGE_HEIGH;
        prior_bboxes[2] /= IMAGE_WIDTH;
        prior_bboxes[3] /= IMAGE_HEIGH;
    }

    if (CODE_TYPE == CODE_TYPE_CORNER)
    {
        if (VARIANCE_ENCODED_IN_TARGET)
        {
            // variance is encoded in target, we simply need to add the offset predictions.
            for(uint i = 0; i < PRIOR_BOX_SIZE; i++)
            {
                decoded_bbox[i] =
                    prior_bboxes[i] +
                    input_location[location_offset];

                location_offset += LOC_XY_SIZE_PRODUCT;
            }
        }
        else
        {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            for(uint i = 0; i < PRIOR_BOX_SIZE; i++)
            {
                decoded_bbox[i] = 
                    mad(input_prior_box[NUM_OF_PRIOR_COMPONENTS + i], // prior variances are places after prior bboxes
                        input_location[location_offset],
                        prior_bboxes[i]);

                location_offset += LOC_XY_SIZE_PRODUCT;
            }
        }
    }
    else if (CODE_TYPE == CODE_TYPE_CENTER_SIZE)
    {
        const UNIT_TYPE prior_width = prior_bboxes[2] - prior_bboxes[0];
        const UNIT_TYPE prior_height = prior_bboxes[3] - prior_bboxes[1];
        const UNIT_TYPE prior_center_x = (prior_bboxes[0] + prior_bboxes[2]) / 2;
        const UNIT_TYPE prior_center_y = (prior_bboxes[1] + prior_bboxes[3]) / 2;
        const UNIT_TYPE bbox_xmin = input_location[location_offset];
        const UNIT_TYPE bbox_ymin = input_location[location_offset + LOC_XY_SIZE_PRODUCT];
        const UNIT_TYPE bbox_xmax = input_location[location_offset + 2 * LOC_XY_SIZE_PRODUCT];
        const UNIT_TYPE bbox_ymax = input_location[location_offset + 3 * LOC_XY_SIZE_PRODUCT];
        UNIT_TYPE decode_bbox_center_x, decode_bbox_center_y;
        UNIT_TYPE decode_bbox_width, decode_bbox_height;

        if (VARIANCE_ENCODED_IN_TARGET)
        {
            // variance is encoded in target, we simply need to restore the offset predictions.
            decode_bbox_center_x = bbox_xmin * prior_width + prior_center_x;
            decode_bbox_center_y = bbox_ymin * prior_height + prior_center_y;
            decode_bbox_width = (exp(bbox_xmax) * prior_width) / 2;
            decode_bbox_height = (exp(bbox_ymax) * prior_height) / 2;
        }
        else
        {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox_center_x = input_prior_box[NUM_OF_PRIOR_COMPONENTS] * bbox_xmin * prior_width + prior_center_x;
            decode_bbox_center_y = input_prior_box[NUM_OF_PRIOR_COMPONENTS + 1] * bbox_ymin * prior_height + prior_center_y;
            decode_bbox_width = (exp(input_prior_box[NUM_OF_PRIOR_COMPONENTS + 2] * bbox_xmax) * prior_width) / 2;
            decode_bbox_height = (exp(input_prior_box[NUM_OF_PRIOR_COMPONENTS + 3] * bbox_ymax) * prior_height) / 2;
        }

        decoded_bbox[0] = decode_bbox_center_x - decode_bbox_width;
        decoded_bbox[1] = decode_bbox_center_y - decode_bbox_height;
        decoded_bbox[2] = decode_bbox_center_x + decode_bbox_width;
        decoded_bbox[3] = decode_bbox_center_y + decode_bbox_height;
    }
    else
    {
        const UNIT_TYPE prior_width = prior_bboxes[2] - prior_bboxes[0];
        const UNIT_TYPE prior_height = prior_bboxes[3] - prior_bboxes[1];
        const UNIT_TYPE bbox_xmin = input_location[location_offset];
        const UNIT_TYPE bbox_ymin = input_location[location_offset + LOC_XY_SIZE_PRODUCT];
        const UNIT_TYPE bbox_xmax = input_location[location_offset + 2 * LOC_XY_SIZE_PRODUCT];
        const UNIT_TYPE bbox_ymax = input_location[location_offset + 3 * LOC_XY_SIZE_PRODUCT];

        if (VARIANCE_ENCODED_IN_TARGET)
        {
            // variance is encoded in target, we simply need to add the offset predictions.
            decoded_bbox[0] = prior_bboxes[0] + bbox_xmin * prior_width;
            decoded_bbox[1] = prior_bboxes[1] + bbox_ymin * prior_height;
            decoded_bbox[2] = prior_bboxes[2] + bbox_xmax * prior_width;
            decoded_bbox[3] = prior_bboxes[3] + bbox_ymax * prior_height;
        }
        else
        {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decoded_bbox[0] = prior_bboxes[0] + input_prior_box[NUM_OF_PRIOR_COMPONENTS] * bbox_xmin * prior_width;
            decoded_bbox[1] = prior_bboxes[1] + input_prior_box[NUM_OF_PRIOR_COMPONENTS + 1] * bbox_ymin * prior_height;
            decoded_bbox[2] = prior_bboxes[2] + input_prior_box[NUM_OF_PRIOR_COMPONENTS + 2] * bbox_xmax * prior_width;
            decoded_bbox[3] = prior_bboxes[3] + input_prior_box[NUM_OF_PRIOR_COMPONENTS + 3] * bbox_ymax * prior_height;
        }
    } 
}

UNIT_TYPE FUNC(get_score)(__global UNIT_TYPE* input_confidence, const uint idx_prior, const uint idx_class, const uint idx_image)
{
    const uint confidence_offset =                    // offset in kernel input 'input_confidence'
            (idx_prior * NUM_CLASSES + idx_image * NUM_OF_PRIORS * NUM_CLASSES + idx_class) *
            CONF_XY_SIZE_PRODUCT +
            CONF_PADDING;

    return (input_confidence[confidence_offset] > CONFIDENCE_THRESHOLD)? input_confidence[confidence_offset] : 0;
}

