// Copyright (c) 2016-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include "include/include_all.cl"

KERNEL(eltwise_gpu_vload8)(INPUTS_DECLS
                           __global OUTPUT_TYPE* output)
{
    const uint global_id = get_global_id(0);

    VLOAD_DECLS

    MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8) res;

    DO_ELTWISE

    res = ACTIVATION(res, ACTIVATION_PARAMS);

    vstore8(res, global_id, output);

}
