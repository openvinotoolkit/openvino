// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <tuple>
#include <vector>

#if !(defined(__APPLE__) || defined(__EMSCRIPTEN__) || defined(_WIN32))
#    include <sched.h>
#endif
#if defined(_WIN32)
#    include <windows.h>

#    include <thread>
#endif

namespace ov {
namespace threading {

#if (defined(__APPLE__) || defined(__EMSCRIPTEN__))
using cpu_set_t = void;
#elif defined(_WIN32)
using cpu_set_t = DWORD_PTR;
#endif

/**
 * @brief      Release the cores affinity mask for the current process
 * @ingroup    ov_dev_api_threading
 *
 * @param      mask  The mask
 */
void release_process_mask(cpu_set_t* mask);

/**
 * @brief      Deleter for process mask
 * @ingroup    ov_dev_api_threading
 */
struct ReleaseProcessMaskDeleter {
    /**
     * @brief      A callable operator to release object
     *
     * @param      mask  The mask to release
     */
    void operator()(cpu_set_t* mask) const {
        release_process_mask(mask);
    }
};

/**
 * @brief A unique pointer to CPU set structure with the ReleaseProcessMaskDeleter deleter
 * @ingroup ov_dev_api_threading
 */
#if defined(_WIN32)
using CpuSet = std::unique_ptr<cpu_set_t>;
#else
using CpuSet = std::unique_ptr<cpu_set_t, ReleaseProcessMaskDeleter>;
#endif

/**
 * @brief Get the cores affinity mask for the current process
 * @ingroup ov_dev_api_threading
 * @return A core affinity mask
 */
std::tuple<CpuSet, int> get_process_mask();

/**
 * @brief      Pins current thread to a set of cores determined by the mask
 * @ingroup    ov_dev_api_threading
 *
 * @param[in]  thrIdx        The thr index
 * @param[in]  hyperThreads  The hyper threads
 * @param[in]  ncores        The ncores
 * @param[in]  processMask   The process mask
 * @return     `True` in case of success, `false` otherwise
 */
bool pin_thread_to_vacant_core(int thrIdx,
                               int hyperThreads,
                               int ncores,
                               const CpuSet& processMask,
                               const std::vector<int>& cpu_ids = {});

/**
 * @brief      Pins thread to a spare core in the round-robin scheme, while respecting the given process mask.
 *             The function can also handle the hyper-threading (by populating the physical cores first)
 * @ingroup    ov_dev_api_threading
 *
 * @param[in]  ncores       The ncores
 * @param[in]  processMask  The process mask
 * @return     `True` in case of success, `false` otherwise
 */
bool pin_current_thread_by_mask(int ncores, const CpuSet& processMask);

/**
 * @brief      Pins a current thread to a socket.
 * @ingroup    ov_dev_api_threading
 *
 * @param[in]  socket  The socket id
 * @return     `True` in case of success, `false` otherwise
 */
bool pin_current_thread_to_socket(int socket);
}  // namespace threading
}  // namespace ov
