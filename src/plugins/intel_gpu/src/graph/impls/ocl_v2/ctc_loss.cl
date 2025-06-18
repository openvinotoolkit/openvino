// Copyright (C) 2022-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define BATCH_NUM          INPUT0_BATCH_NUM
#define MAX_TIME           INPUT0_FEATURE_NUM
#define CLASSES_NUM        INPUT0_SIZE_Y
#define MAX_DECODED_LENGTH (MAX_TIME * 2 + 1)

#ifdef INPUT4_TYPE
#    define BLANK_INDEX blank_index[INPUT4_GET_INDEX(0, 0, 0, 0)]
#else
#    define BLANK_INDEX (CLASSES_NUM - 1)
#endif

inline OUTPUT_TYPE FUNC(sumLogs)(OUTPUT_TYPE log1, OUTPUT_TYPE log2) {
    if (log1 == -INFINITY) {
        return log2;
    } else if (log2 == -INFINITY) {
        return log1;
    } else {
        if (log1 > log2) {
            return log1 + log1p(exp(log2 - log1));
        } else {
            return log2 + log1p(exp(log1 - log2));
        }
    }
}

// Based on the CPU plugin implementation that uses the backward dynamic programming algorithm
KERNEL(ctc_loss_ref)
(const __global INPUT0_TYPE* logits,
 const __global INPUT1_TYPE* logit_length,
 const __global INPUT2_TYPE* labels,
 const __global INPUT3_TYPE* label_length,
#ifdef INPUT4_TYPE
 const __global INPUT4_TYPE* blank_index,
#endif
 __global OUTPUT_TYPE* output) {
    const uint b = get_global_id(0);

    const INPUT1_TYPE actual_logit_length = logit_length[INPUT1_GET_INDEX(0, b, 0, 0)];
    const INPUT3_TYPE actual_target_length = label_length[INPUT3_GET_INDEX(0, b, 0, 0)];

    INPUT2_TYPE decoded_target[MAX_DECODED_LENGTH] = {};
    INPUT3_TYPE decoded_target_length = 0;
#if UNIQUE
    bool founded_values[CLASSES_NUM] = {};
    for (uint t = 0; t < actual_target_length; ++t) {
        const INPUT2_TYPE value = labels[INPUT2_GET_INDEX(b, t, 0, 0)];
        if (founded_values[value]) {
            continue;
        }
        founded_values[value] = true;
        decoded_target[decoded_target_length++] = BLANK_INDEX;
        decoded_target[decoded_target_length++] = value;
    }
    decoded_target[decoded_target_length++] = BLANK_INDEX;
#elif PREPROCESS_COLLAPSE_REPEATED
    INPUT2_TYPE previous_value = labels[INPUT2_GET_INDEX(b, 0, 0, 0)];
    decoded_target[decoded_target_length++] = BLANK_INDEX;
    decoded_target[decoded_target_length++] = previous_value;
    for (uint t = 1; t < actual_target_length; ++t) {
        const INPUT2_TYPE value = labels[INPUT2_GET_INDEX(b, t, 0, 0)];
        if (value == previous_value) {
            continue;
        }
        decoded_target[decoded_target_length++] = BLANK_INDEX;
        decoded_target[decoded_target_length++] = value;
        previous_value = value;
    }
    decoded_target[decoded_target_length++] = BLANK_INDEX;
#else
    for (uint t = 0; t < actual_target_length; ++t) {
        decoded_target[decoded_target_length++] = BLANK_INDEX;
        decoded_target[decoded_target_length++] = labels[INPUT2_GET_INDEX(b, t, 0, 0)];
    }
    decoded_target[decoded_target_length++] = BLANK_INDEX;
#endif

    OUTPUT_TYPE log_probabilities[MAX_TIME][MAX_DECODED_LENGTH] = {};
    for (uint t = 0; t < actual_logit_length; ++t) {
        OUTPUT_TYPE exp_sum = OUTPUT_VAL_ZERO;
        for (uint c = 0; c < CLASSES_NUM; ++c) {
            exp_sum += exp(logits[INPUT0_GET_INDEX(b, t, c, 0)]);
        }
        for (uint s = 0; s < decoded_target_length; ++s) {
            log_probabilities[t][s] = logits[INPUT0_GET_INDEX(b, t, decoded_target[s], 0)] - log(exp_sum);
        }
    }

    OUTPUT_TYPE log_backward[MAX_DECODED_LENGTH][MAX_TIME] = {};
    for (uint i = 0; i < MAX_DECODED_LENGTH; ++i) {
        for (uint j = 0; j < MAX_TIME; ++j) {
            log_backward[i][j] = -INFINITY;
        }
    }
    log_backward[decoded_target_length - 1][actual_logit_length - 1] = OUTPUT_VAL_ZERO;
    log_backward[decoded_target_length - 2][actual_logit_length - 1] = OUTPUT_VAL_ZERO;

    for (INPUT1_TYPE t = actual_logit_length - 2; t >= 0; t--) {
        const INPUT1_TYPE t_1 = t + 1;
        for (INPUT1_TYPE s = max(INPUT1_VAL_ZERO, decoded_target_length - (2 * (actual_logit_length - t)));
             s < min(decoded_target_length, 2 * (t_1));
             s++) {
            if (CTC_MERGE_REPEATED || decoded_target[s] == BLANK_INDEX) {
                log_backward[s][t] =
                    FUNC_CALL(sumLogs)(log_backward[s][t], log_backward[s][t_1] + log_probabilities[t_1][s]);
            }

            if (s + 1 < decoded_target_length) {
                log_backward[s][t] =
                    FUNC_CALL(sumLogs)(log_backward[s][t], log_backward[s + 1][t_1] + log_probabilities[t_1][s + 1]);
            }

            if (s + 2 < decoded_target_length) {
                if (decoded_target[s] != BLANK_INDEX &&
                    (!CTC_MERGE_REPEATED || (decoded_target[s] != decoded_target[s + 2]))) {
                    log_backward[s][t] = FUNC_CALL(sumLogs)(log_backward[s][t],
                                                            log_backward[s + 2][t_1] + log_probabilities[t_1][s + 2]);
                }
            }
        }
    }

    log_backward[0][0] += log_probabilities[0][0];
    log_backward[1][0] += log_probabilities[0][1];

    output[OUTPUT_GET_INDEX(b, 0, 0, 0)] = -FUNC_CALL(sumLogs)(log_backward[0][0], log_backward[1][0]);
}

#undef BATCH_NUM
#undef MAX_TIME
#undef CLASSES_NUM
#undef MAX_DECODED_LENGTH
#undef BLANK_INDEX
