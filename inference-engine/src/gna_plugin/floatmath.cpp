// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "floatmath.h"
#include "pwl.h"
#include "gna_plugin_log.hpp"
#include <cmath>


void CNNFilter32(intel_dnn_component_t *component) {
    float *ptr_filters = reinterpret_cast<float *>(component->op.conv1D.ptr_filters);
    float *ptr_biases = reinterpret_cast<float *>(component->op.conv1D.ptr_biases);
    float *ptr_inputs = reinterpret_cast<float *>(component->ptr_inputs);
    float *ptr_outputs = reinterpret_cast<float *>(component->ptr_outputs);
    uint32_t num_group = component->num_rows_in;
    uint32_t num_filter_outputs = component->op.conv1D.num_feature_map_rows - component->op.conv1D.num_filter_rows + 1;
    uint32_t
        num_inputs_band_stride = component->op.conv1D.num_feature_maps * component->op.conv1D.num_feature_map_columns;
    uint32_t num_filter_coefficients = component->op.conv1D.num_filter_coefficients;

    if ((component->num_rows_in != 1) || (component->num_rows_out != 1)
        || (component->num_columns_out != num_filter_outputs * component->op.conv1D.num_filters)) {
        THROW_GNA_EXCEPTION << "Bad problem dimensions in CNNFilter32!";
    }

    for (uint32_t j = 0; j < num_filter_outputs; j++) {
        float *ptr_in = ptr_inputs + j * num_inputs_band_stride;
        for (uint32_t i = 0; i < component->op.conv1D.num_filters; i++) {
            float *ptr_coef = ptr_filters + i * num_filter_coefficients;
            float sum = ptr_biases[i];
            for (uint32_t k = 0; k < num_filter_coefficients; k++) {
                sum += ptr_in[k] * ptr_coef[k];
            }
            ptr_outputs[j * component->op.conv1D.num_filters + i] = sum;
        }
    }
}

void CNNMaxPool(intel_dnn_component_t *component, intel_dnn_number_type_t number_type) {
    if (number_type == kDnnInt) {
        int32_t *ptr_inputs = reinterpret_cast<int32_t *>(component->ptr_inputs);
        int32_t *ptr_outputs = reinterpret_cast<int32_t *>(component->ptr_outputs);
        uint32_t num_inputs = component->num_columns_in;
        uint32_t num_columns = component->op.maxpool.num_inputs_stride;
        uint32_t num_pool_size = component->op.maxpool.num_inputs;
        uint32_t num_pool_step = component->op.maxpool.num_inputs_step;
        uint32_t num_rows_in = num_inputs / component->op.maxpool.num_inputs_stride;
        uint32_t num_rows_out = num_rows_in / num_pool_step;

        for (uint32_t i = 0; i < num_columns; i++) {
            int32_t m = 0;
            if (component->op.maxpool.do_sum_not_max) {
                uint32_t num_saturate = 0;
                for (uint32_t j = 0; j < num_rows_in; j += num_pool_step) {
                    int64_t sum = 0;
                    uint32_t num_end = (j + num_pool_size > num_rows_in) ? num_rows_in : j + num_pool_size;
                    for (uint32_t k = j; k < num_end; k++) {
                        sum += ptr_inputs[k * num_columns + i];
                    }
                    if (sum > 2147483647.0) {
                        ptr_outputs[m * num_columns + i] = 2147483647L;
                        num_saturate++;
                    } else if (sum < -2147483648.0) {
                        ptr_outputs[m * num_columns + i] = -2147483648L;
                        num_saturate++;
                    } else {
                        ptr_outputs[m * num_columns + i] = (int32_t) sum;
                    }
                    m++;
                }
                if (num_saturate > 0) {
                    fprintf(stderr, "Warning:  %d saturations in CNNMaxPool()\n", num_saturate);
                }
            } else {
                for (uint32_t j = 0; j < num_rows_in; j += num_pool_step) {
                    int32_t max = INT32_MIN;
                    uint32_t num_end = (j + num_pool_size > num_rows_in) ? num_rows_in : j + num_pool_size;
                    for (uint32_t k = j; k < num_end; k++) {
                        if (ptr_inputs[k * num_columns + i] > max) max = ptr_inputs[k * num_columns + i];
                    }
                    ptr_outputs[m * num_columns + i] = max;
                    m++;
                }
            }
        }
    } else {
        float *ptr_inputs = reinterpret_cast<float *>(component->ptr_inputs);
        float *ptr_outputs = reinterpret_cast<float *>(component->ptr_outputs);
        uint32_t num_inputs = component->num_columns_in;
        uint32_t num_columns = component->op.maxpool.num_inputs_stride;
        uint32_t num_pool_size = component->op.maxpool.num_inputs;
        uint32_t num_pool_step = component->op.maxpool.num_inputs_step;
        uint32_t num_rows_in = num_inputs / component->op.maxpool.num_inputs_stride;
        uint32_t num_rows_out = num_rows_in / num_pool_step;

        for (uint32_t i = 0; i < num_columns; i++) {
            int32_t m = 0;
            if (component->op.maxpool.do_sum_not_max) {
                for (uint32_t j = 0; j < num_rows_in; j += num_pool_step) {
                    float sum = 0.0;
                    uint32_t num_end = (j + num_pool_size > num_rows_in) ? num_rows_in : j + num_pool_size;
                    for (uint32_t k = j; k < num_end; k++) {
                        sum += ptr_inputs[k * num_columns + i];
                    }
                    ptr_outputs[m * num_columns + i] = sum;
                    m++;
                }
            } else {
                for (uint32_t j = 0; j < num_rows_in; j += num_pool_step) {
                    float max = -1e20f;
                    uint32_t num_end = (j + num_pool_size > num_rows_in) ? num_rows_in : j + num_pool_size;
                    for (uint32_t k = j; k < num_end; k++) {
                        if (ptr_inputs[k * num_columns + i] > max) max = ptr_inputs[k * num_columns + i];
                    }
                    ptr_outputs[m * num_columns + i] = max;
                    m++;
                }
            }
        }
    }
}

void PwlApply16(intel_dnn_component_t *component, uint32_t num_subset_size) {
    if (component->orientation_in == kDnnInterleavedOrientation) {  // subsets only supported in interleaved orientation
        PwlApply16(component, 0, num_subset_size - 1, 0, component->num_columns_in - 1);
    } else {
        PwlApply16(component, 0, component->num_rows_in - 1, 0, component->num_columns_in - 1);
    }
}

void PwlApply16(intel_dnn_component_t *component,
                uint32_t num_row_start,
                uint32_t num_row_end,
                uint32_t num_col_start,
                uint32_t num_col_end) {
    uint32_t num_saturate = 0;
    uint32_t num_segments = component->op.pwl.num_segments;
    if (num_segments > 0) {
        intel_pwl_segment_t *ptr_segment = component->op.pwl.ptr_segments;
        for (int i = num_row_start; i <= num_row_end; i++) {
            int32_t *ptr_input = reinterpret_cast<int32_t *>(component->ptr_inputs) + i * component->num_columns_in;
            int16_t *ptr_output = reinterpret_cast<int16_t *>(component->ptr_outputs) + i * component->num_columns_in;
            for (int j = num_col_start; j <= num_col_end; j++) {
                int32_t xbase = (int32_t) (ptr_segment[0].xBase & XBASEMASK);
                int32_t input = ptr_input[j];
                if (input <= xbase) {
                    ptr_output[j] = ptr_segment[0].yBase;
                } else {
                    uint32_t slope_shift;
                    int16_t slope, ybase;
                    int64_t diff, prod, prod_shift, sum;
                    uint32_t k = num_segments / 2;
                    uint32_t k_upper = num_segments;
                    uint32_t k_lower = 0;
                    while (k_upper > k_lower + 1) {
                        xbase = (int32_t) (ptr_segment[k].xBase & XBASEMASK);
                        if (xbase > input) {
                            k_upper = k;
                            k = (k + k_lower) / 2;
                        } else {
                            k_lower = k;
                            k = (k_upper + k) / 2;
                        }
                    }
                    xbase = (int32_t) (ptr_segment[k].xBase & XBASEMASK);
                    slope_shift = ((ptr_segment[k].xBase & ~XBASEMASK) + 1) * 8;
                    slope = ptr_segment[k].slope;
                    ybase = ptr_segment[k].yBase;
                    diff = (int64_t) input - (int64_t) xbase;
                    prod = diff * slope;
                    prod_shift = prod >> slope_shift;
                    sum = prod_shift + (int64_t) ybase;
                    if (sum > 32767LL) {
                        ptr_output[j] = 32767;
                        num_saturate++;
                    } else if (sum < -32768LL) {
                        ptr_output[j] = -32768;
                        num_saturate++;
                    } else {
                        ptr_output[j] = (int16_t) sum;
                    }
                }
            }
        }
    }

    if (num_saturate > 0) {
        fprintf(stderr, "Warning:  %d saturations in PwlApply16!\n", num_saturate);
    }
}

void PwlApply32(intel_dnn_component_t *component, uint32_t num_subset_size) {
    if (component->orientation_in == kDnnInterleavedOrientation) {  // subsets only supported in interleaved orientation
        PwlApply32(component, 0, num_subset_size - 1, 0, component->num_columns_in - 1);
    } else {
        PwlApply32(component, 0, component->num_rows_in - 1, 0, component->num_columns_in - 1);
    }
}

void PwlApply32(intel_dnn_component_t *component,
                uint32_t num_row_start,
                uint32_t num_row_end,
                uint32_t num_col_start,
                uint32_t num_col_end) {
    intel_piecewiselinear_t *transform = reinterpret_cast<intel_piecewiselinear_t *>(&component->op.pwl);
    float *ptr_in = reinterpret_cast<float *>(component->ptr_inputs);
    float *ptr_out = reinterpret_cast<float *>(component->ptr_outputs);
    uint32_t num_columns = component->num_columns_in;
    switch (transform->func_id.type) {
        case kActSigmoid:
            for (uint32_t i = num_row_start; i <= num_row_end; i++) {
                for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                    ptr_out[i * num_columns + j] = 0.5 * (1.0 + tanh(0.5 * ptr_in[i * num_columns + j]));
                }
            }
            break;
        case kActTanh:
            for (uint32_t i = num_row_start; i <= num_row_end; i++) {
                for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                    ptr_out[i * num_columns + j] = tanh(ptr_in[i * num_columns + j]);
                }
            }
            break;
        case kActRelu:
            for (uint32_t i = num_row_start; i <= num_row_end; i++) {
                for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                    ptr_out[i * num_columns + j] =
                        (ptr_in[i * num_columns + j] < 0.0f) ? ptr_in[i * num_columns + j] * transform->func_id.negative_slope : ptr_in[i * num_columns + j];
                }
            }
            break;
        case kActIdentity:
            for (uint32_t i = num_row_start; i <= num_row_end; i++) {
                for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                    ptr_out[i * num_columns + j] = ptr_in[i * num_columns + j];
                }
            }
            break;
        case kActKaldiLstmClipping:
            for (uint32_t i = num_row_start; i <= num_row_end; i++) {
                for (uint32_t j = num_col_start; j <= num_col_end; j++) {
                    float val = ptr_in[i * num_columns + j];
                    if (val > KALDI_LSTM_CLIP_UPPER) {
                        ptr_out[i * num_columns + j] = KALDI_LSTM_CLIP_UPPER;
                    } else if (val < KALDI_LSTM_CLIP_LOWER) {
                        ptr_out[i * num_columns + j] = KALDI_LSTM_CLIP_LOWER;
                    } else {
                        ptr_out[i * num_columns + j] = val;
                    }
                }
            }
            break;
        case kActCustom:
            // break;
        default:fprintf(stderr, "Unknown piecewise linear function type!\n");
            throw -1;
    }
}

#ifdef __cplusplus
extern "C" {  // API uses C linkage so that it can be used by C and C++ applications
#endif

#ifdef _NO_MKL_
void cblas_sgemm1(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                  const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                  const MKL_INT K, const float alpha, const float *A,
                  const MKL_INT lda, const float *B, const MKL_INT ldb,
                  const float beta, float *C, const MKL_INT ldc) {
    int i, j, k;

    if (Layout != CblasRowMajor) {
        fprintf(stderr, "Only row major is supported in cblas_sgemm!\n");
        throw -1;
    }

    if ((TransA == CblasNoTrans) && (TransB == CblasNoTrans)) {
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                float sum = (beta == 1.0) ? C[i * ldc + j] : 0;
                for (k = 0; k < K; k++) {
                    sum += A[i * lda + k] * B[k * ldb + j];
                }
                C[i * ldc + j] = sum;
            }
        }
    } else if ((TransA == CblasNoTrans) && (TransB == CblasTrans)) {
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                float sum;
                sum = beta * C[i * ldc + j];
                for (k = 0; k < K; k++) {
                    sum += alpha * A[i * lda + k] * B[j * ldb + k];
                }
                C[i * ldc + j] = sum;
            }
        }
    } else if ((TransA == CblasTrans) && (TransB == CblasNoTrans)) {
        for (i = 0; i < M; i++) {
            for (j = 0; j < N; j++) {
                float sum = (beta == 1.0) ? C[i * ldc + j] : 0;
                for (k = 0; k < K; k++) {
                    sum += A[k * lda + i] * B[k * ldb + j];
                }
                C[i * ldc + j] = sum;
            }
        }
    } else {
        fprintf(stderr, "Expected A not transposed in cblas_sgemm!\n");
        throw -1;
    }
}
void cblas_ssbmv1(const CBLAS_LAYOUT Layout, const CBLAS_UPLO Uplo,
                  const MKL_INT N, const MKL_INT K, const float alpha, const float *A,
                  const MKL_INT lda, const float *X, const MKL_INT incX,
                  const float beta, float *Y, const MKL_INT incY) {
    int i, j, k;

    if (Layout != CblasRowMajor) {
        fprintf(stderr, "Only row major is supported in cblas_ssbmv!\n");
        throw -1;
    }
    if (Uplo != CblasLower) {
        fprintf(stderr, "Only lower format is supported in cblas_ssbmv!\n");
        throw -1;
    }
    if (K != 0) {
        fprintf(stderr, "Only diagonal matrices supported in cblas_ssbmv at this time!\n");
        throw -1;
    }
    if ((alpha == 1.0) && (beta == 1.0) && (incX == 1) && (incY == 1)) {
        for (i = 0; i < N; i++) {
            Y[i] += A[i] * X[i];
        }
    } else {
        fprintf(stderr, "Only alpha=1, beta=1, incX=1, incY=1, LDA=1 supported in cblas_ssbmv at this time!\n");
        throw -1;
    }
}
#endif  // #ifdef _NO_MKL_

void cblas_sgemm_subset(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB, const MKL_INT M, const MKL_INT N,
                        const MKL_INT K, const float alpha, const float *A,
                        const MKL_INT lda, const float *B, const MKL_INT ldb,
                        const float beta, float *C, const MKL_INT ldc,
                        const uint32_t *OutputList, const MKL_INT L) {
    int i, j, k, l;

    if (Layout != CblasRowMajor) {
        fprintf(stderr, "Only row major is supported in cblas_sgemm_subset!\n");
        throw -1;
    }

    if ((TransA == CblasNoTrans) && (TransB == CblasNoTrans)) {
        for (l = 0; l < L; l++) {
            i = OutputList[l];
            for (j = 0; j < N; j++) {
                float sum = (beta == 1.0) ? C[l * ldc + j] : 0;
                for (k = 0; k < K; k++) {
                    sum += A[i * lda + k] * B[k * ldb + j];
                }
                C[l * ldc + j] = sum;
            }
        }
    } else if ((TransA == CblasNoTrans) && (TransB == CblasTrans)) {
        for (i = 0; i < M; i++) {
            for (l = 0; l < L; l++) {
                float sum;
                j = OutputList[l];
                sum = beta * C[i * ldc + l];
                for (k = 0; k < K; k++) {
                    sum += alpha * A[i * lda + k] * B[j * ldb + k];
                }
                C[i * ldc + l] = sum;
            }
        }
    } else if ((TransA == CblasTrans) && (TransB == CblasNoTrans)) {
        for (l = 0; l < L; l++) {
            i = OutputList[l];
            for (j = 0; j < N; j++) {
                float sum = (beta == 1.0) ? C[l * ldc + j] : 0;
                for (k = 0; k < K; k++) {
                    sum += A[k * lda + i] * B[k * ldb + j];
                }
                C[l * ldc + j] = sum;
            }
        }
    } else {
        fprintf(stderr, "Expected A not transposed in cblas_sgemm_subset!\n");
        throw -1;
    }
}

// C = [ A1 A2 ] * X + B
void sgemv_split(const uint32_t N,
                 const uint32_t K1,
                 const uint32_t K2,
                 const float *A1,
                 const float *A2,
                 const float *X,
                 const float *B,
                 float *C) {
    uint32_t num_columns = K1 + K2;
    uint32_t num_rows = N;
    uint32_t i, j;

    for (i = 0; i < num_rows; i++) {
        float sum = B[i];
        for (j = 0; j < K1; j++) {
            sum += A1[j] * X[i * num_columns + j];
        }
        for (j = K1; j < num_columns; j++) {
            sum += A2[j - K1] * X[i * num_columns + j];
        }
        C[i] = sum;
    }
}

#ifdef __cplusplus
}  // end extern "C"
#endif
