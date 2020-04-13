// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Abstraction over platform specific implementations
 * @file ie_system_conf.h
 */

#pragma once

#include "ie_api.h"
#include <vector>

namespace InferenceEngine {

/**
 * @brief Provides the reference to static thread_local std::exception_ptr
 * @return A an exception pointer
 */
INFERENCE_ENGINE_API_CPP(std::exception_ptr&) CurrentException();

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
 * @brief      Returns number of CPU physical cores on Linux/Windows (which is considered to be more performance friendly for servers)
 *             (on other OSes it simply relies on the original parallel API of choice, which usually uses the logical cores )
 * @ingroup    ie_dev_api_system_conf
 * @return     Number of physical CPU cores.
 */
INFERENCE_ENGINE_API_CPP(int) getNumberOfCPUCores();

/**
 * @brief      Checks whether CPU supports SSE 4.2 capability
 * @ingroup    ie_dev_api_system_conf
 * @return     `True` is SSE 4.2 instructions are available, `false` otherwise
 */
INFERENCE_ENGINE_API_CPP(bool) with_cpu_x86_sse42();

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

}  // namespace InferenceEngine
