/*******************************************************************************
 * Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Shared tile declarations for expert GEMM kernels.
// Requires DECORATOR (gm or moe) and the standard batch_headers/* common/vector/tile includes.

#define UG(suffix) CAT(CAT(ugemm_, DECORATOR), suffix)

// Aliases — DECLARE_2D_TILE doesn't expand nested CAT reliably across OCL preprocessors.
#define UGEMM_SG_TILE_M      UG(_sg_tile_m)
#define UGEMM_SG_TILE_N      UG(_sg_tile_n)
#define UGEMM_WG_TILE_M      UG(_wg_tile_m)
#define UGEMM_WG_TILE_N      UG(_wg_tile_n)
#define UGEMM_C_TYPE          UG(_c_type)
#define UGEMM_C_TYPE_HALF     UG(_c_type_half)
#define UGEMM_C_TYPE_BLOCK0   UG(_c_type_block0)
#define UGEMM_C_TYPE_BLOCK1   UG(_c_type_block1)
#define UGEMM_C_TYPE_NBLOCK0  UG(_c_type_nblock0)
#define UGEMM_C_TYPE_NBLOCK1  UG(_c_type_nblock1)

#ifdef BIAS_DT
DECLARE_2D_TILE(bias_tile_type, BIAS_DT, SUBGROUP_SIZE, UGEMM_SG_TILE_M, 1, 1, 1)
#endif

DECLARE_2D_TILE(UGEMM_C_TYPE_HALF,
                half,
                SUBGROUP_SIZE,
                UGEMM_C_TYPE_BLOCK0,
                UGEMM_C_TYPE_BLOCK1,
                UGEMM_C_TYPE_NBLOCK0,
                UGEMM_C_TYPE_NBLOCK1)

DECLARE_2D_TILE_COPY_REBLOCK(UGEMM_C_TYPE,
                             SUBGROUP_SIZE,
                             UGEMM_C_TYPE_BLOCK0,
                             UGEMM_C_TYPE_BLOCK1,
                             UGEMM_C_TYPE_NBLOCK0,
                             UGEMM_C_TYPE_NBLOCK1,
                             UGEMM_C_TYPE_HALF,
                             SUBGROUP_SIZE,
                             UGEMM_C_TYPE_BLOCK0,
                             UGEMM_C_TYPE_BLOCK1,
                             UGEMM_C_TYPE_NBLOCK0,
                             UGEMM_C_TYPE_NBLOCK1)

#define unroll_for __attribute__((opencl_unroll_hint)) for
