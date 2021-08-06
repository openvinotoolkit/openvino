// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_api.h>

#include <tuple>
#include <memory>

#if !(defined(__APPLE__) || defined(_WIN32))
#include <sched.h>
#endif

namespace InferenceEngine {
#if (defined(__APPLE__) || defined(_WIN32))
using cpu_set_t = void;
#endif  // (defined(__APPLE__) || defined(_WIN32))

/**
 * @brief      Release the cores affinity mask for the current process
 * @ingroup    ie_dev_api_threading
 *
 * @param      mask  The mask
 */
void ReleaseProcessMask(cpu_set_t* mask);

/**
 * @brief      Deleter for process mask
 * @ingroup    ie_dev_api_threading
 */
struct ReleaseProcessMaskDeleter {
    /**
     * @brief      A callable operator to release object
     *
     * @param      mask  The mask to release
     */
    void operator()(cpu_set_t* mask) const {
        ReleaseProcessMask(mask);
    }
};

/**
 * @brief A unique pointer to CPU set structure with the ReleaseProcessMaskDeleter deleter
 * @ingroup ie_dev_api_threading
 */
using CpuSet = std::unique_ptr<cpu_set_t, ReleaseProcessMaskDeleter>;

/**
 * @brief Get the cores affinity mask for the current process
 * @ingroup ie_dev_api_threading
 * @return A core affinity mask
 */
std::tuple<CpuSet, int> GetProcessMask();

/**
 * @brief      Pins current thread to a set of cores determined by the mask
 * @ingroup    ie_dev_api_threading
 *
 * @param[in]  thrIdx        The thr index
 * @param[in]  hyperThreads  The hyper threads
 * @param[in]  ncores        The ncores
 * @param[in]  processMask   The process mask
 * @return     `True` in case of success, `false` otherwise
 */
bool PinThreadToVacantCore(int thrIdx, int hyperThreads, int ncores, const CpuSet& processMask);

/**
 * @brief      Pins thread to a spare core in the round-robin scheme, while respecting the given process mask.
 *             The function can also handle the hyper-threading (by populating the physical cores first)
 * @ingroup    ie_dev_api_threading
 *
 * @param[in]  ncores       The ncores
 * @param[in]  processMask  The process mask
 * @return     `True` in case of success, `false` otherwise
 */
bool PinCurrentThreadByMask(int ncores, const CpuSet& processMask);

/**
 * @brief      Pins a current thread to a socket.
 * @ingroup    ie_dev_api_threading
 *
 * @param[in]  socket  The socket id
 * @return     `True` in case of success, `false` otherwise
 */
bool PinCurrentThreadToSocket(int socket);
}  //  namespace InferenceEngine
