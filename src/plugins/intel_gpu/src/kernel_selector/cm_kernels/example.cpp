// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

namespace KERNEL_NAME {

#include "include/example_header.h"

extern "C" _GENX_MAIN_ void KERNEL_NAME(svmptr_t x [[type("svmptr_t")]]) {
    // This kernel prints and exits
    if (cm_linear_global_id() == 0) {
        printf("Example CM kernel\n");
        printf("Pointer address: %p\n", (void*)x);

        // Call function from header
        print_lws_gws();

        // Check macro from batch header
#ifdef EXAMPLE_CM_MACRO
        printf("Batch header included\n");
#else
        printf("Batch header not included\n");
#endif
    }
}
}  // namespace KERNEL_NAME
