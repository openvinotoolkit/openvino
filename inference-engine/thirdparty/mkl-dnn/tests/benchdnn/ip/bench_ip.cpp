/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "mkldnn.h"

#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

#include "ip/ip.hpp"

namespace ip {

int bench(int argc, char **argv) {
    ip::prb_t ipps[] = {
        { FWD_B, 16, 8, 1, 1, 32, mkldnn_u8, mkldnn_s8, mkldnn_s32, mkldnn_u8 },
    };

    const int ip_num = sizeof(ipps) / sizeof(ipps[0]);
    int ip_fails = 0;

    for (int id = 0; id < ip_num; ++id) {
        res_t res{};
        ip::doit(&ipps[id], &res);
        if (res.errors) {
            printf("[%3d] FAILED (errs: %lu out of %lu)\n", id,
                    (unsigned long)res.errors, (unsigned long)res.total);
        }
        ip_fails += res.errors != 0;
    }


    benchdnn_stat.failed = ip_fails;
    benchdnn_stat.tests = ip_num;

    return OK;
}

}
