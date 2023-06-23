// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifndef _NO_MKL_
#    include <mkl_cblas.h>
#    include <mkl_dnn.h>
#endif

#ifndef CBLAS_LAYOUT
#    define CBLAS_LAYOUT CBLAS_ORDER
#endif

#ifdef _NO_MKL_
#    ifndef _MKL_H_
#        define _MKL_H_
typedef enum { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_LAYOUT;
typedef enum { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 } CBLAS_TRANSPOSE;
typedef enum { CblasUpper = 121, CblasLower = 122 } CBLAS_UPLO;
typedef enum { CblasNonUnit = 131, CblasUnit = 132 } CBLAS_DIAG;
typedef enum { CblasLeft = 141, CblasRight = 142 } CBLAS_SIDE;
typedef CBLAS_LAYOUT CBLAS_ORDER; /* this for backward compatibility with CBLAS_ORDER */
#        define MKL_INT int
#    endif  // #ifndef _MKL_H_
#endif      // #ifdef _NO_MKL_

#ifdef _NO_MKL_
void cblas_sgemm1(const CBLAS_LAYOUT Layout,
                  const CBLAS_TRANSPOSE TransA,
                  const CBLAS_TRANSPOSE TransB,
                  const MKL_INT M,
                  const MKL_INT N,
                  const MKL_INT K,
                  const float alpha,
                  const float* A,
                  const MKL_INT lda,
                  const float* B,
                  const MKL_INT ldb,
                  const float beta,
                  float* C,
                  const MKL_INT ldc);
void cblas_ssbmv1(const CBLAS_LAYOUT Layout,
                  const CBLAS_UPLO Uplo,
                  const MKL_INT N,
                  const MKL_INT K,
                  const float alpha,
                  const float* A,
                  const MKL_INT lda,
                  const float* X,
                  const MKL_INT incX,
                  const float beta,
                  float* Y,
                  const MKL_INT incY);
#endif  // #ifdef _NO_MKL_
void cblas_sgemm_subset(const CBLAS_LAYOUT Layout,
                        const CBLAS_TRANSPOSE TransA,
                        const CBLAS_TRANSPOSE TransB,
                        const MKL_INT M,
                        const MKL_INT N,
                        const MKL_INT K,
                        const float alpha,
                        const float* A,
                        const MKL_INT lda,
                        const float* B,
                        const MKL_INT ldb,
                        const float beta,
                        float* C,
                        const MKL_INT ldc,
                        const uint32_t* OutputList,
                        const MKL_INT L);
void sgemv_split(const uint32_t N,
                 const uint32_t K1,
                 const uint32_t K2,
                 const float* A1,
                 const float* A2,
                 const float* X,
                 const float* B,
                 float* C);
