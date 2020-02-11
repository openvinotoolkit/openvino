// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <cmath>

#include <details/ie_exception.hpp>

#if GNA_LIB_VER == 2
#include <gna2-model-api.h>
#include "gna2_model_helper.hpp"
#include "gna2_model_debug_log.hpp"
#else
#include <gna-api-types-xnn.h>

#endif

#ifndef _NO_MKL_
#include <mkl_dnn.h>
#endif

#ifdef INTEGER_REF
#include "convnet.h"
#include "igemv16.h"
#include "igemv8.h"
#include "sgemm.h"
#else
#include "runtime/floatmath.h"
#endif

#include "dnn.hpp"
#include "gna_plugin_log.hpp"
#include "runtime/pwl.h"
#include "runtime/cnn.h"


void GNAPluginNS::backend::ApplyAffineTransform(intel_dnn_component_t *component, uint32_t *list, uint32_t listsize) {
    auto transform = &component->op.affine;
    int m = component->num_rows_out;
    int n = component->num_columns_in;
    int k = component->num_rows_in;
    int lda = component->num_rows_in;
    int ldb = component->num_columns_in;
    int ldc = component->num_columns_out;

    switch (component->num_bytes_per_input) {
#ifdef INTEGER_REF
        case 2:
            if (component->op.affine.num_bytes_per_weight == 1) {
                int8_t *A = reinterpret_cast<int8_t*>(transform->ptr_weights);
                int16_t *B = reinterpret_cast<int16_t*>(component->ptr_inputs);
                int32_t *C = reinterpret_cast<int32_t*>(component->ptr_outputs);
                intel_compound_bias_t *bias = reinterpret_cast<intel_compound_bias_t*>(transform->ptr_biases);
                if (list == nullptr) {
                    //  PrintMatrixInt8("W int8", W, k, m, ldw, component->op.affine.weight_scale_factor);
                    //  PrintMatrixInt16("X int16", X, k, n, ldx, component->op.affine.weight_scale_factor);
                    //  PrintMatrixInt32("Y int32", Y, m, n, ldy, component->output_scale_factor);
                    igemm8_gna(m, n, k, A, lda, B, ldb, bias, C, ldc);
                } else {
                    //  PrintMatrixInt8("W int8", W, k, m, ldw, component->op.affine.weight_scale_factor);
                    //  PrintMatrixInt16("X int16", X, k, n, ldx, component->op.affine.weight_scale_factor);
                    //  PrintMatrixInt32("Y int32", Y, m, n, ldy, component->output_scale_factor);
                    igemm8_gna_subset(m, n, k, A, lda, B, ldb, bias, C, ldc, list, listsize);
                }
                //  PrintMatrixInt32("C int32", C, m, n, ldc, component->output_scale_factor);
            } else if (component->op.affine.num_bytes_per_weight == 2) {
                int16_t *A = reinterpret_cast<int16_t*>(transform->ptr_weights);
                int16_t *B = reinterpret_cast<int16_t*>(component->ptr_inputs);
                int32_t *C = reinterpret_cast<int32_t*>(component->ptr_outputs);
                int32_t *bias = reinterpret_cast<int32_t*>(transform->ptr_biases);
                if (list == nullptr) {
                    for (uint32_t i = 0; i < m; i++) {
                        for (uint32_t j = 0; j < n; j++) {
                            C[i*ldc+j] = bias[i];
                        }
                    }
                    //  PrintMatrixInt16("A int16", A, k, m, lda, component->op.affine.weight_scale_factor);
                    //  PrintMatrixInt16("trans(B) int16", B, k, n, ldb, component->op.affine.weight_scale_factor);
                    //  PrintMatrixInt32("C int32", C, m, n, ldc, component->output_scale_factor);
                    cblas_igemm16(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, lda, B, ldb, 1.0, C, ldc);
                } else {
                    for (int l = 0; l < listsize; l++) {
                        int i = list[l];
                        for (uint32_t j = 0; j < n; j++) {
                            C[l*ldc+j] = bias[i];
                        }
                    }
                    //  PrintMatrixInt16("A int16", A, k, m, lda, component->op.affine.scale_factor);
                    //  PrintMatrixInt16("trans(B) int16", B, k, n, ldb, component->op.affine.scale_factor);
                    //  PrintMatrixInt32("C int32", C, m, n, ldc, component->op.affine.scale_factor * component->op.affine.scale_factor);
                    cblas_igemm16_subset(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, lda, B, ldb, 1.0, C, ldc, list, listsize);
                }
                //  PrintMatrixInt32("C int32", C, m, n, ldc, component->output_scale_factor);
            } else {
                fprintf(stderr, "Bad weight width in ApplyAffineTransform!\n");
                throw -1;
            }
            break;
#endif  // #ifdef INTEGER_REF
        case 4: {
            auto A = reinterpret_cast<float *>(transform->ptr_weights);
            auto B = reinterpret_cast<float *>(component->ptr_inputs);
            auto C = reinterpret_cast<float *>(component->ptr_outputs);
            auto bias = reinterpret_cast<float *>(transform->ptr_biases);
            if (list == nullptr) {
                for (uint32_t i = 0; i < m; i++) {
                    for (uint32_t j = 0; j < n; j++) {
                        C[i * ldc + j] = bias[i];
                    }
                }
                //  if (global_debug) PrintMatrixFloat32("A float", A, m, k, lda);
                //  if (global_debug) PrintMatrixFloat32("B float", B, k, n, ldb);
                //  if (global_debug) PrintMatrixFloat32("C float before", C, m, n, ldc);
                cblas_sgemm1(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, lda, B, ldb, 1.0, C, ldc);
                //  if (global_debug) PrintMatrixFloat32("C float after", C, m, n, ldc);
            } else {
                for (int l = 0; l < listsize; l++) {
                    int i = list[l];
                    for (uint32_t j = 0; j < n; j++) {
                        C[l * ldc + j] = bias[i];
                    }
                }
                //  PrintMatrixFloat32("A float", A, k, m, lda);
                //  PrintMatrixFloat32("trans(B) float", B, k, n, ldb);
                //  PrintMatrixFloat32("C float before", C, listsize, n, ldc);
                cblas_sgemm_subset(CblasRowMajor,
                                   CblasNoTrans,
                                   CblasNoTrans,
                                   m,
                                   n,
                                   k,
                                   1.0,
                                   A,
                                   lda,
                                   B,
                                   ldb,
                                   1.0,
                                   C,
                                   ldc,
                                   list,
                                   listsize);
                //  PrintMatrixFloat32("C float after", C, listsize, n, ldc);
            }
        }
            break;
        default:
            THROW_GNA_EXCEPTION << "Bad data width in ApplyAffineTransform: " << component->num_bytes_per_input;
    }
}

void GNAPluginNS::backend::ApplyDiagonalTransform(intel_dnn_component_t *component) {
    auto transform = &component->op.affine;
    int m = component->num_rows_out;
    int n = component->num_columns_in;
    int ldb = component->num_columns_in;
    int ldc = component->num_columns_out;

    switch (component->num_bytes_per_input) {
#ifdef INTEGER_REF
        case 2:
            if (component->op.affine.num_bytes_per_weight == 1) {
                int8_t *A = reinterpret_cast<int8_t*>(transform->ptr_weights);
                int16_t *B = reinterpret_cast<int16_t*>(component->ptr_inputs);
                int32_t *C = reinterpret_cast<int32_t*>(component->ptr_outputs);
                intel_compound_bias_t *bias = reinterpret_cast<intel_compound_bias_t*>(transform->ptr_biases);
                //  PrintMatrixInt8("W int8", W, k, m, ldw, component->op.affine.weight_scale_factor);
                //  PrintMatrixInt16("X int16", X, k, n, ldx, component->op.affine.weight_scale_factor);
                //  PrintMatrixInt32("Y int32", Y, m, n, ldy, component->output_scale_factor);
                isbmm8_gna(m, n, A, lda, B, ldb, bias, C, ldc);
                //  PrintMatrixInt32("C int32", C, m, n, ldc, component->output_scale_factor);
            } else if (component->op.affine.num_bytes_per_weight == 2) {
                int16_t *A = reinterpret_cast<int16_t*>(transform->ptr_weights);
                int16_t *B = reinterpret_cast<int16_t*>(component->ptr_inputs);
                int32_t *C = reinterpret_cast<int32_t*>(component->ptr_outputs);
                int32_t *bias = reinterpret_cast<int32_t*>(transform->ptr_biases);
                for (uint32_t i = 0; i < m; i++) {
                    for (uint32_t j = 0; j < n; j++) {
                        C[i*ldc+j] = bias[i];
                    }
                }
                //  PrintMatrixInt16("A int16", A, 1, m, lda, component->op.affine.weight_scale_factor);
                //  PrintMatrixInt16("trans(B) int16", B, k, n, ldb, component->op.affine.weight_scale_factor);
                //  PrintMatrixInt32("C int32", C, m, n, ldc, component->output_scale_factor);
                cblas_isbmm16(m, n, A, lda, B, ldb, C, ldc);
                //  PrintMatrixInt32("C int32", C, m, n, ldc, component->output_scale_factor);
            } else {
                fprintf(stderr, "Bad weight width in ApplyDiagonalTransform!\n");
                throw -1;
            }
            break;
#endif  // #ifdef INTEGER_REF
        case 4: {
            auto A = reinterpret_cast<float *>(transform->ptr_weights);
            auto B = reinterpret_cast<float *>(component->ptr_inputs);
            auto C = reinterpret_cast<float *>(component->ptr_outputs);
            auto bias = reinterpret_cast<float *>(transform->ptr_biases);
            for (uint32_t i = 0; i < m; i++) {
                for (uint32_t j = 0; j < n; j++) {
                    C[i * ldc + j] = bias[i];
                }
            }
            //  PrintMatrixFloat32("A float", A, 1, m, lda);
            //  PrintMatrixFloat32("B float", B, k, n, ldb);
            //  PrintMatrixFloat32("C float before", C, m, n, ldc);
            for (uint32_t j = 0; j < n; j++) {
                float *Bcol = B + j * ldb;
                float *Ccol = C + j * ldc;
                cblas_ssbmv1(CblasRowMajor, CblasLower, m, 0, 1.0, A, 1, Bcol, 1, 1.0, Ccol, 1);
            }
            //  PrintMatrixFloat32("C float after", C, m, n, ldc);
            break;
        }
        default:
            THROW_GNA_EXCEPTION << "Bad data width in ApplyDiagonalTransform: " << component->num_bytes_per_input;
    }
}

void GNAPluginNS::backend::ApplyRecurrentTransform(intel_dnn_component_t *component, uint32_t row, void *ptr_feedbacks) {
    intel_recurrent_t *transform = &component->op.recurrent;
    int k1 = component->num_columns_in;
    int k2 = component->num_columns_out;
    int n = k2;

    if (component->op.recurrent.ptr_feedbacks == nullptr) {
        fprintf(stderr, "nullptr feedback pointer in ApplyRecurrentTransform()!\n");
        throw -1;
    }

    switch (component->num_bytes_per_input) {
#ifdef INTEGER_REF
        case 2:
            if (component->op.recurrent.num_bytes_per_weight == 1) {
                int16_t *A1 = reinterpret_cast<int16_t*>(component->ptr_inputs) + row * component->num_columns_in;
                int16_t *A2 = reinterpret_cast<int16_t*>(ptr_feedbacks);
                int8_t *X = reinterpret_cast<int8_t*>(transform->ptr_weights);
                intel_compound_bias_t *B = reinterpret_cast<intel_compound_bias_t*>(transform->ptr_biases);
                int32_t *C = reinterpret_cast<int32_t*>(component->ptr_outputs) + row * component->num_columns_out;
                //  PrintMatrixInt16("A1 int", A1, 1, k1, k1, component->op.recurrent.weight_scale_factor);
                //  PrintMatrixInt16("A2 int", A2, 1, k2, k2);
                //  PrintMatrixInt8("X int", X, k, n, n, component->op.recurrent.weight_scale_factor);
                //  PrintMatrixInt32("B int", B, 1, 2*n, 2*n, component->output_scale_factor);
                igemv8_gna_split(n, k1, k2, A1, A2, X, B, C);
                //  PrintMatrixInt32("C int", C, 1, n, n, component->output_scale_factor);
            } else if (component->op.recurrent.num_bytes_per_weight == 2) {
                int16_t *A1 = reinterpret_cast<int16_t*>(component->ptr_inputs) + row * component->num_columns_in;
                int16_t *A2 = reinterpret_cast<int16_t*>(ptr_feedbacks);
                int16_t *X = reinterpret_cast<int16_t*>(transform->ptr_weights);
                int32_t *B = reinterpret_cast<int32_t*>(transform->ptr_biases);
                int32_t *C = reinterpret_cast<int32_t*>(component->ptr_outputs) + row * component->num_columns_out;
                //  PrintMatrixInt16("A1 int", A1, 1, k1, k1, component->op.recurrent.weight_scale_factor);
                //  PrintMatrixInt16("A2 int", A2, 1, k2, k2, component->op.recurrent.weight_scale_factor);
                //  PrintMatrixInt16("X int", X, k, n, n, component->op.recurrent.weight_scale_factor);
                //  PrintMatrixInt32("B int", B, 1, n, n, component->output_scale_factor);
                igemv16_split(n, k1, k2, A1, A2, X, B, C);
                //  PrintMatrixInt32("C int", C, 1, n, n, component->output_scale_factor);
            } else {
                fprintf(stderr, "Weight width not supported in ApplyRecurrentTransform!\n");
                throw -1;
            }
            break;
#endif  // #ifdef INTEGER_REF
        case 4: {
            auto A1 = reinterpret_cast<float *>(component->ptr_inputs) + row * component->num_columns_in;
            auto A2 = reinterpret_cast<float *>(ptr_feedbacks);
            auto X = reinterpret_cast<float *>(transform->ptr_weights);
            auto B = reinterpret_cast<float *>(transform->ptr_biases);
            auto C = reinterpret_cast<float *>(component->ptr_outputs) + row * component->num_columns_out;
            //  PrintMatrixFloat32("A1 float", A1, 1, k1, k1);
            //  PrintMatrixFloat32("A2 float", A2, 1, k2, k2);
            //  PrintMatrixFloat32("X float", X, k, n, n);
            //  PrintMatrixFloat32("B float", B, 1, n, n);
            sgemv_split(n, k1, k2, A1, A2, X, B, C);
            //  PrintMatrixFloat32("C float", C, 1, n, n);
            break;
        }
        default:
            THROW_GNA_EXCEPTION << "Bad data width in ApplyRecurrentTransform: " << component->num_bytes_per_input;
    }
}

void GNAPluginNS::backend::ApplyConvolutional1DTransform(intel_dnn_component_t *component) {
    switch (component->num_bytes_per_input) {
#ifdef INTEGER_REF
        case 2:
            CNNFilter16(component);
            break;
#endif  // #ifdef INTEGER_REF
        case 4:
            //  PrintMatrixFloat32("Input float", reinterpret_cast<float*>(component->ptr_inputs),
            //  component->num_rows_in, component->num_columns_in, component->num_columns_in);
            //  PrintMatrixFloat32("Filt float", reinterpret_cast<float*>(component->op.conv1D.ptr_filters),
            //  component->op.conv1D.num_filters,
            //  component->op.conv1D.num_filter_rows*component->op.conv1D.num_feature_map_columns*component->op.conv1D.num_feature_maps,
            //  component->op.conv1D.num_filter_rows*component->op.conv1D.num_feature_map_columns*component->op.conv1D.num_feature_maps);
            //  PrintMatrixFloat32("Bias float", reinterpret_cast<float*>(component->op.conv1D.ptr_biases), 1,
            // component->op.conv1D.num_filters, component->op.conv1D.num_filters);
            CNNFilter32(component);
            //  PrintMatrixFloat32("Output float", reinterpret_cast<float*>(component->ptr_outputs, component->num_rows_out,
            // component->num_columns_out, component->num_columns_out);
            break;
        default:
            THROW_GNA_EXCEPTION << "Bad data width in ApplyConvolutionalTransform: " << component->num_bytes_per_input;
    }
}

void GNAPluginNS::backend::ApplyPiecewiseLinearTransform(intel_dnn_component_t *component,
                                            intel_dnn_number_type_t number_type,
                                            uint32_t listsize) {
    if (number_type == kDnnFloat) {
        // PrintMatrixFloat32("PWL Input float", reinterpret_cast<float*>(component->ptr_inputs), component->num_rows_in,
        // component->num_columns_in, component->num_columns_in);
        PwlApply32(component, listsize);
        // PrintMatrixFloat32("PWL Output float", reinterpret_cast<float*>(component->ptr_outputs), component->num_rows_out,
        // component->num_columns_out, component->num_columns_out);
#ifdef INTEGER_REF
        } else if (component->num_bytes_per_output == 2) {
            PwlApply16(component, listsize);
#endif  // #ifdef INTEGER_REF
    } else {
        THROW_GNA_EXCEPTION << "Bad data width in ApplyPiecewiseLinearTransform: " << number_type;
    }
}

void GNAPluginNS::backend::ApplyPiecewiseLinearTransform(intel_dnn_component_t *component,
                                            intel_dnn_number_type_t number_type,
                                            uint32_t listsize,
                                            uint32_t num_row) {
    if (number_type == kDnnFloat) {
        PwlApply32(component, num_row, num_row, 0, listsize - 1);
#ifdef INTEGER_REF
        } else if (component->num_bytes_per_output == 2) {
            PwlApply16(component, num_row, num_row, 0, listsize-1);
#endif  // #ifdef INTEGER_REF
    } else {
        THROW_GNA_EXCEPTION << "Bad data width in ApplyPiecewiseLinearTransform: " << number_type;
    }
}

void GNAPluginNS::backend::ApplyMaxPoolTransform(intel_dnn_component_t *component, intel_dnn_number_type_t number_type) {
    if (component->num_bytes_per_input == 4) {
        // PrintMatrixFloat32("Input float", reinterpret_cast<float*>(component->ptr_inputs), component->num_rows_in,
        // component->num_columns_in, component->num_columns_in);
        CNNMaxPool(component, number_type);
        // PrintMatrixFloat32("Output float", reinterpret_cast<float*>(component->ptr_outputs), component->num_rows_out,
        // component->num_columns_out, component->num_columns_out);
    } else {
        fprintf(stderr, "Bad data width in ApplyMaxPoolTransform!\n");
        throw -1;
    }
}

void GNAPluginNS::backend::ApplyTranspose(intel_dnn_component_t *component) {
    int m = component->num_rows_in;
    int n = component->num_columns_in;
    int lda = component->num_columns_in;
    int ldb = component->num_columns_out;
    // B = Transpose(A) where A is mxn and B is nxm
    switch (component->num_bytes_per_input) {
#ifdef INTEGER_REF
        case 1:
            {
                int8_t *A = reinterpret_cast<int8_t*>(component->ptr_inputs);
                int8_t *B = reinterpret_cast<int8_t*>(component->ptr_outputs);
                for (uint32_t row = 0; row < m; row++) {
                    for (uint32_t col = 0; col < n; col++) {
                        B[col*ldb+row] = A[row*lda+col];
                    }
                }
            }
            break;
        case 2:
            {
                int16_t *A = reinterpret_cast<int16_t*>(component->ptr_inputs);
                int16_t *B = reinterpret_cast<int16_t*>(component->ptr_outputs);
                for (uint32_t row = 0; row < m; row++) {
                    for (uint32_t col = 0; col < n; col++) {
                        B[col*ldb+row] = A[row*lda+col];
                    }
                }
            }
            break;
#endif  // #ifdef INTEGER_REF
        case 4: {
            auto A = reinterpret_cast<float *>(component->ptr_inputs);
            auto B = reinterpret_cast<float *>(component->ptr_outputs);
            for (uint32_t row = 0; row < m; row++) {
                for (uint32_t col = 0; col < n; col++) {
                    B[col * ldb + row] = A[row * lda + col];
                }
            }
        }
            break;
        default:fprintf(stderr, "Bad data width in ApplyInterleave!\n");
            throw -1;
    }
}

void GNAPluginNS::backend::ApplyCopy(intel_dnn_component_t *component) {
    auto src = reinterpret_cast<uint8_t *>(component->ptr_inputs);
    auto dst = reinterpret_cast<uint8_t *>(component->ptr_outputs);
    int32_t m = component->op.copy.num_copy_rows;
    int32_t n = component->op.copy.num_copy_columns;
    int32_t lda = component->num_columns_in;
    int32_t ldb = component->num_columns_out;
    if (m > component->num_rows_in) {
        fprintf(stderr, "Error:  attempt to copy more columns than matrix has!\n");
        throw -1;
    } else {
        switch (component->num_bytes_per_input) {
#ifdef INTEGER_REF
            case 2:
                {
                    int16_t *A = reinterpret_cast<int16_t*>(src);
                    int16_t *B = reinterpret_cast<int16_t*>(dst);
                    for (uint32_t row = 0; row < m; row++) {
                        for (uint32_t col = 0; col < n; col++) {
                            B[row*ldb + col] = A[row*lda + col];
                        }
                    }
                }
                break;
#endif  // #ifdef INTEGER_REF
            case 4: {
                auto A = reinterpret_cast<float *>(src);
                auto B = reinterpret_cast<float *>(dst);
                for (uint32_t row = 0; row < m; row++) {
                    for (uint32_t col = 0; col < n; col++) {
                        B[row * ldb + col] = A[row * lda + col];
                    }
                }
            }
                break;
            default:fprintf(stderr, "Bad data width in ApplyCopy!\n");
                throw -1;
        }
    }
}

bool GNAPluginNS::backend::isCompatibleDnn(GNAPluginNS::backend::AMIntelDNN dnn1, GNAPluginNS::backend::AMIntelDNN dnn2) {
    bool isCompatible = true;

    // compare basic structures to see if they are compatible
    if (dnn1.num_components() != dnn2.num_components()) isCompatible = false;
    for (int i = 0; i < dnn1.num_components(); i++) {
        if (dnn1.component[i].num_rows_in != dnn2.component[i].num_rows_in) isCompatible = false;
        if (dnn1.component[i].num_columns_in != dnn2.component[i].num_columns_in) isCompatible = false;
        if (dnn1.component[i].num_rows_out != dnn2.component[i].num_rows_out) isCompatible = false;
        if (dnn1.component[i].num_columns_out != dnn2.component[i].num_columns_out) isCompatible = false;
        if (dnn1.component[i].operation != dnn2.component[i].operation) isCompatible = false;
    }

    return (isCompatible);
}

void GNAPluginNS::backend::ClearScoreError(intel_score_error_t *error) {
    error->num_scores = 0;
    error->num_errors = 0;
    error->max_error = 0.0;
    error->sum_error = 0.0;
    error->sum_squared_error = 0.0;
    error->max_rel_error = 0.0;
    error->sum_rel_error = 0.0;
    error->sum_squared_rel_error = 0.0;
}

void GNAPluginNS::backend::UpdateScoreError(intel_score_error_t *error, intel_score_error_t *total_error) {
    total_error->num_errors += error->num_errors;
    total_error->num_scores += error->num_scores;
    total_error->sum_error += error->sum_error;
    total_error->sum_squared_error += error->sum_squared_error;
    if (error->max_error > total_error->max_error) {
        total_error->max_error = error->max_error;
    }
    total_error->sum_rel_error += error->sum_rel_error;
    total_error->sum_squared_rel_error += error->sum_squared_rel_error;
    if (error->max_rel_error > total_error->max_rel_error) {
        total_error->max_rel_error = error->max_rel_error;
    }
}

void GNAPluginNS::backend::SoftmaxGoogle(float *ptr_output, float *ptr_input, const uint32_t num_outputs, const uint32_t num_inputs) {
    // Assumes input vector contains log likelihoods
    // The computes x[i] = x[i] - log(sum_j exp(x[j]))
    // This normalizes the likelihoods by the sum of likelihoods but stores them as log likelihoods

    float max_score = ptr_input[0];
    float sum = 0.0;
    float diff;
    // find max score for normalization to [0,1]
    for (uint32_t i = 0; i < num_inputs; i++) {
        if (ptr_input[i] > max_score) {
            max_score = ptr_input[i];
        }
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        sum += exp(ptr_input[i] - max_score);
    }
    if (sum < 1.0e-20) {
        fprintf(stderr, "Warning:  attempt to take log(0) in SoftmaxGoogle()!\n");
        sum = 1.0e-20;
    }
    diff = max_score + log(sum);
    for (uint32_t i = 0; i < num_outputs; i++) {
        ptr_output[i] = ptr_input[i] - diff;
    }
}
