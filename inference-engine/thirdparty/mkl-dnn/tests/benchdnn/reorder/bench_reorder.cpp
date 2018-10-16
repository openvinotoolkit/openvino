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

#include "mkldnn.h"
#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

#include "reorder.hpp"
#include "input_reorder.hpp"

namespace reorder {

int bench(int argc, char **argv) {
    const int num_r = sizeof(reorders) / sizeof(reorders[0]);
    const int num_q = sizeof(q10ns) / sizeof(q10ns[0]);
    const int num_s = sizeof(default_scales) / sizeof(default_scales[0]);

    for (int q = 0; q < num_q; ++q) {
        if (q10ns[q].scale == 0) {
           for (int idx = 0; idx < num_s; ++idx) {
               q10ns[q].scale = default_scales[idx];
               for (int r = 0; r < num_r; ++r) {
                   const prb_t p(&reorders[r], &q10ns[q]);
                   check(&p);
               }
            }
        } else {
            for (int r = 0; r < num_r; ++r) {
                const prb_t p(&reorders[r], &q10ns[q]);
                check(&p);
            }
        }
    }

    return OK;
}

void check(const prb_t *p) {
    res_t res{};
    char pstr[max_prb_len];
    prb2str(p, &res, pstr);

    int status = reorder::doit(p, &res);

    prb2str(p, &res, pstr);
    bool want_perf_report = false;

    parse_result(res, want_perf_report, false, status, pstr);

    if (bench_mode & PERF)
        perf_report(p, &res, pstr);

    benchdnn_stat.tests++;
}

void perf_report(const prb_t *p, const res_t *r, const char *pstr) {
    const auto &t = r->timer;
    const int max_len = 400;
    char buffer[max_len], *buf = buffer;
    int rem_len = max_len - 1;

    #   define DPRINT(...) do { \
        int l = snprintf(buf, rem_len, __VA_ARGS__); \
        buf += l; rem_len -= l; \
    } while(0)

    DPRINT("perf,");
    DPRINT("%s,", pstr);
    DPRINT("min_ms=%g,", t.ms(benchdnn_timer_t::min));
    DPRINT("max_ms=%g", t.ms(benchdnn_timer_t::max));

#   undef DPRINT
    print(0, "%s\n", buffer);
}

}
