// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Abstraction over platform specific implementations
 * @file ie_system_conf.h
 */

#pragma once

#include "ie_api.h"
#include <vector>
#include <exception>

namespace InferenceEngine {

/**
 * @brief      Checks whether OpenMP environment variables are defined
 * @ingroup    ie_dev_api_system_conf
 *
 * @param[in]  includeOMPNumThreads  Indicates if the omp number threads is included
 * @return     `True` if any OpenMP environment variable is defined, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) checkOpenMpEnvVars(bool includeOMPNumThreads = true);

/**
 * @brief      Returns available CPU NUMA nodes (on Linux, and Windows [only with TBB], single node is assumed on all other OSes)
 * @ingroup    ie_dev_api_system_conf
 * @return     NUMA nodes
 */
INFERENCE_ENGINE_API_CPP(std::vector<int>) getAvailableNUMANodes();

/**
 * @brief      Returns available CPU cores types (on Linux, and Windows) and ONLY with TBB, single core type is assumed otherwise
 * @ingroup    ie_dev_api_system_conf
 * @return     Vector of core types
 */
INFERENCE_ENGINE_API_CPP(std::vector<int>) getAvailableCoresTypes();

/**
 * @brief      Returns number of CPU physical cores on Linux/Windows (which is considered to be more performance friendly for servers)
 *             (on other OSes it simply relies on the original parallel API of choice, which usually uses the logical cores).
 *                     call function with 'false' to get #phys cores of all types
 *                     call function with 'true' to get #phys 'Big' cores
 *                     number of 'Little' = 'all' - 'Big'
 * @ingroup    ie_dev_api_system_conf
 * @param[in]  bigCoresOnly Additionally limits the number of reported cores to the 'Big' cores only.
 * @return     Number of physical CPU cores.
 */
INFERENCE_ENGINE_API_CPP(int) getNumberOfCPUCores(bool bigCoresOnly = false);

/**
 * @brief      Checks whether CPU supports SSE 4.2 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is SSE 4.2 instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_sse42();

/**
 * @brief      Checks whether CPU supports AVX capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is AVX instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_avx();

/**
 * @brief      Checks whether CPU supports AVX2 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is AVX2 instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_avx2();

/**
 * @brief      Checks whether CPU supports AVX 512 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is AVX512F (foundation) instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_avx512f();

/**
 * @brief      Checks whether CPU supports AVX 512 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is AVX512F, AVX512BW, AVX512DQ instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_avx512_core();

/**
 * @brief      Checks whether CPU supports BFloat16 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is tAVX512_BF16 instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_bfloat16();

}  // namespace InferenceEngine
