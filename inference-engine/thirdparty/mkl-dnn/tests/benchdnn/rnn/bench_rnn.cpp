/*******************************************************************************
 * Copyright 2018 Intel Corporation
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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mkldnn.h"

#include "mkldnn_common.hpp"
#include "mkldnn_debug.hpp"
#include "mkldnn_memory.hpp"

#include "rnn/input_rnn.hpp"
#include "rnn/rnn.hpp"

namespace rnn {

int bench(int argc, char **argv) {
    // !!?? TODO: check consistence of direction, dir ...
    mkldnn_prop_kind_t direction = mkldnn_forward;
    dir_t dir = FWD_D;
    for (int arg = 0; arg < argc; ++arg) {
        if (!strncmp("--dir=", argv[arg], 6)) {
            dir = str2dir(argv[arg] + 6);
            if (dir == FWD_D)
                direction = mkldnn_forward;
            else if (dir == BWD_D)
                direction = mkldnn_backward;
            else
                assert("unknown dir");
        }
    }
    const int num_r = sizeof(rnns) / sizeof(rnns[0]);

    for (int r = 0; r < num_r; ++r) {
        const rnn_prb_t p(rnns[r], conf_f32, direction);
        check(&p);
    }

    return OK;
}

void check(const rnn_prb_t *p) {
    res_t res{};
    char pstr[max_prb_len];
    prb2str(p, &res, pstr);

    int status = rnn::doit(p, &res);

    prb2str(p, &res, pstr);
    bool want_perf_report = false;

    parse_result(res, want_perf_report, false, status, pstr);

    if (bench_mode & PERF)
        perf_report(p, &res, pstr);

    benchdnn_stat.tests++;
}

} // namespace rnn
