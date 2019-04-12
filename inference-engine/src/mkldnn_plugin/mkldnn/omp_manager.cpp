// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>
#include <cstring>
#include "ie_parallel.hpp"
#include "omp_manager.h"

using namespace MKLDNNPlugin;
namespace MKLDNNPlugin {
namespace cpu {

static const char *openMpEnvVars[] = {
        "OMP_CANCELLATION", "OMP_DISPLAY_ENV", "OMP_DEFAULT_DEVICE", "OMP_DYNAMIC",
        "OMP_MAX_ACTIVE_LEVELS", "OMP_MAX_TASK_PRIORITY", "OMP_NESTED",
        "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_STACKSIZE",
        "OMP_SCHEDULE", "OMP_THREAD_LIMIT", "OMP_WAIT_POLICY", "GOMP_CPU_AFFINITY",
        "GOMP_DEBUG", "GOMP_STACKSIZE", "GOMP_SPINCOUNT", "GOMP_RTEMS_THREAD_POOLS",
        "KMP_AFFINITY", "KMP_NUM_THREADS", "MIC_KMP_AFFINITY",
        "MIC_OMP_NUM_THREADS", "MIC_OMP_PROC_BIND", "PHI_KMP_AFFINITY",
        "PHI_OMP_NUM_THREADS", "PHI_KMP_PLACE_THREADS", "MKL_NUM_THREADS",
        "MKL_DYNAMIC", "MKL_DOMAIN_NUM_THREADS"
};

static const unsigned numberOfOpenMpEnvVars =
        sizeof(openMpEnvVars) / sizeof(openMpEnvVars[0]);

bool checkOpenMpEnvVars(bool includeOMPNumThreads) {
    for (unsigned i = 0; i < numberOfOpenMpEnvVars; i++) {
        if (getenv(openMpEnvVars[i])) {
            if (0 != strcmp(openMpEnvVars[i], "OMP_NUM_THREADS") || includeOMPNumThreads)
                return true;
        }
    }
    return false;
}

#if !(defined(__APPLE__) || defined(_WIN32))
// getNumberOfCPUSockets/getNumberOfCPUCores are implemented in the lin_omp_manager.cpp
#else
int getNumberOfCPUSockets() {return 1;}
int getNumberOfCPUCores()   {return parallel_get_max_threads();}
#endif

}  // namespace cpu
}  // namespace MKLDNNPlugin
