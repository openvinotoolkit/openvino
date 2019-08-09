/*******************************************************************************
* Copyright 2018-2019 Intel Corporation
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
*******************************************************************************/

#ifndef GEMM_DRIVER_HPP
#define GEMM_DRIVER_HPP

#include "mkldnn_types.h"

namespace mkldnn {
namespace impl {
namespace cpu {

template <typename a_type, typename b_type, typename c_type>
mkldnn_status_t gemm_driver(
        const char *transA, const char *transB, const char *offsetC,
        const int *m, const int *n, const int *k,
        const float *alpha, const a_type *a, const int *lda, const a_type *oa,
        const b_type *b, const int *ldb, const a_type *ob,
        const float *beta, c_type *c, const int *ldc, const c_type *oc,
        const bool force_jit_nocopy_gemm);

}
}
}

#endif // GEMM_DRIVER_HPP
