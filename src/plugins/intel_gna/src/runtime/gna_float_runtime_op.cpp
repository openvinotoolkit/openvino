// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_float_runtime.hpp"
#include "pwl.h"
#include "cnn.h"
#include "floatmath.h"

using namespace GNAPluginNS;
using namespace GNAPluginNS::runtime;

void FP::ApplyAffineTransform(intel_dnn_component_t *component, uint32_t *list, uint32_t listsize) {
    if (4 != component->num_bytes_per_input) {
        THROW_GNA_EXCEPTION << "Bad data width: " << component->num_bytes_per_input;
    }

    auto transform = &component->op.affine;
    uint32_t m = component->num_rows_out;
    uint32_t n = component->num_columns_in;
    uint32_t k = component->num_rows_in;
    uint32_t lda = component->num_rows_in;
    uint32_t ldb = component->num_columns_in;
    uint32_t ldc = component->num_columns_out;

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
        cblas_sgemm1(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, lda, B, ldb, 1.0, C, ldc);
    } else {
        for (uint32_t l = 0; l < listsize; l++) {
            int i = list[l];
            for (uint32_t j = 0; j < n; j++) {
                C[l * ldc + j] = bias[i];
            }
        }
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
    }
}

void FP::ApplyDiagonalTransform(intel_dnn_component_t *component) {
    if (4 != component->num_bytes_per_input) {
        THROW_GNA_EXCEPTION << "Bad data width: " << component->num_bytes_per_input;
    }

    auto transform = &component->op.affine;
    uint32_t m = component->num_rows_out;
    uint32_t n = component->num_columns_in;
    uint32_t ldc = component->num_columns_out;

    auto A = reinterpret_cast<float *>(transform->ptr_weights);
    auto B = reinterpret_cast<float *>(component->ptr_inputs);
    auto C = reinterpret_cast<float *>(component->ptr_outputs);
    auto bias = reinterpret_cast<float *>(transform->ptr_biases);
    for (uint32_t i = 0; i < m; i++) {
        for (uint32_t j = 0; j < n; j++) {
            C[i * ldc + j] = bias[i];
        }
    }
    std::vector<float> Arow(n);
    for (uint32_t i = 0; i < m; i++) {
        float *Brow = B + i * n;
        float *Crow = C + i * ldc;
        std::fill(std::begin(Arow), std::end(Arow), A[i]);
        cblas_ssbmv1(CblasRowMajor, CblasLower, n, 0, 1.0, Arow.data(), 1, Brow, 1, 1.0, Crow, 1);
    }
}

void FP::ApplyRecurrentTransform(intel_dnn_component_t *component, uint32_t row, void *ptr_feedbacks) {
    if (4 != component->num_bytes_per_input) {
        THROW_GNA_EXCEPTION << "Bad data width: " << component->num_bytes_per_input;
    }

    intel_recurrent_t *transform = &component->op.recurrent;
    int k1 = component->num_columns_in;
    int k2 = component->num_columns_out;
    int n = k2;

    if (component->op.recurrent.ptr_feedbacks == nullptr) {
        THROW_GNA_EXCEPTION << "nullptr feedback pointer";
    }
    auto A1 = reinterpret_cast<float *>(component->ptr_inputs) + row * component->num_columns_in;
    auto A2 = reinterpret_cast<float *>(ptr_feedbacks);
    auto X = reinterpret_cast<float *>(transform->ptr_weights);
    auto B = reinterpret_cast<float *>(transform->ptr_biases);
    auto C = reinterpret_cast<float *>(component->ptr_outputs) + row * component->num_columns_out;
    sgemv_split(n, k1, k2, A1, A2, X, B, C);
}

void FP::ApplyConvolutional1DTransform(intel_dnn_component_t *component) {
    if (4 != component->num_bytes_per_input) {
        THROW_GNA_EXCEPTION << "Bad data width: " << component->num_bytes_per_input;
    }
    CNNFilter32(component);
}

void FP::ApplyConvolutional2DTransform(intel_dnn_component_t* component) {
    CNN2DFilter32(component);
}

void FP::ApplyPiecewiseLinearTransform(intel_dnn_component_t *component,
                                                         intel_dnn_number_type_t number_type,
                                                         uint32_t listsize) {
    if (kDnnFloat != number_type) {
        THROW_GNA_EXCEPTION << "Bad number type: " << number_type;
    }
    PwlApply32(component, listsize);
}

void FP::ApplyPiecewiseLinearTransform(intel_dnn_component_t *component,
                                                         intel_dnn_number_type_t number_type,
                                                         uint32_t listsize,
                                                         uint32_t num_row) {
    if (kDnnFloat != number_type) {
        THROW_GNA_EXCEPTION << "Bad number type: " << number_type;
    }
    PwlApply32(component, num_row, num_row, 0, listsize - 1);
}

void FP::ApplyMaxPoolTransform(intel_dnn_component_t *component, intel_dnn_number_type_t number_type) {
    if (4 != component->num_bytes_per_input) {
        THROW_GNA_EXCEPTION << "Bad data width: " << component->num_bytes_per_input;
    }
    CNNMaxPool(component, number_type);
}

void FP::ApplyTranspose(intel_dnn_component_t *component) {
    if (4 != component->num_bytes_per_input) {
        THROW_GNA_EXCEPTION << "Bad data width: " << component->num_bytes_per_input;
    }

    uint32_t m = component->num_rows_in;
    uint32_t n = component->num_columns_in;
    uint32_t lda = component->num_columns_in;
    uint32_t ldb = component->num_columns_out;
    // B = Transpose(A) where A is mxn and B is nxm
    auto A = reinterpret_cast<float *>(component->ptr_inputs);
    auto B = reinterpret_cast<float *>(component->ptr_outputs);
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t col = 0; col < n; col++) {
            B[col * ldb + row] = A[row * lda + col];
        }
    }
}

void FP::ApplyCopy(intel_dnn_component_t *component) {
    if (4 != component->num_bytes_per_input) {
        THROW_GNA_EXCEPTION << "Bad data width: " << component->num_bytes_per_input;
    }

    auto src = reinterpret_cast<uint8_t *>(component->ptr_inputs);
    auto dst = reinterpret_cast<uint8_t *>(component->ptr_outputs);
    uint32_t m = component->op.copy.num_copy_rows;
    uint32_t n = component->op.copy.num_copy_columns;
    uint32_t lda = component->num_columns_in;
    uint32_t ldb = component->num_columns_out;
    if (m > component->num_rows_in) {
        THROW_GNA_EXCEPTION << "Error:  attempt to copy more columns than matrix has";
    }
    auto A = reinterpret_cast<float *>(src);
    auto B = reinterpret_cast<float *>(dst);
    for (uint32_t row = 0; row < m; row++) {
        for (uint32_t col = 0; col < n; col++) {
            B[row * ldb + col] = A[row * lda + col];
        }
    }
}
