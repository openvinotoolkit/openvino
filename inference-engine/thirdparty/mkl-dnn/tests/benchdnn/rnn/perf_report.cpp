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
#include "mkldnn_memory.hpp"

#include "rnn/rnn.hpp"

namespace rnn {

void perf_report(const rnn_prb_t *p, const res_t *r, const char *pstr) {
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
    DPRINT("time(ms):");
    DPRINT("min=%g,", t.ms(benchdnn_timer_t::min));
    DPRINT("max=%g,", t.ms(benchdnn_timer_t::max));
    DPRINT("avg=%g", t.ms(benchdnn_timer_t::avg));

#   undef DPRINT
    print(0, "%s\n", buffer);
}

} // namespace rnn
