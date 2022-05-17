/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include <stdio.h>
#include <stdlib.h>

#include <sstream>

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/parser.hpp"

#include "prelu/prelu.hpp"

namespace prelu {

void check_correctness(const settings_t &s) {
    for_(const auto &i_dir : s.dir)
    for_(const auto &i_sdt : s.sdt)
    for_(const auto &i_stag : s.stag)
    for (const auto &i_scratchpad_mode : s.scratchpad_mode) {
        if (s.sdims.size() != 2) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: input tensors were specified in wrong format. "
                    "Please use NxNxNxNxN:MxMxMxMxM as a problem description "
                    "format.");
            SAFE_V(FAIL);
        }
        if (i_sdt.size() != 2) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: input data types were specified in wrong format. "
                    "Please use --sdt=X:X format.");
            SAFE_V(FAIL);
        }
        if (i_stag.size() != 2) {
            BENCHDNN_PRINT(0, "%s\n",
                    "Error: input format tags were specified in wrong format. "
                    "Please use --stag=X:X format.");
            SAFE_V(FAIL);
        }

        attr_t attr;
        attr.insert(i_scratchpad_mode);

        const prb_t prb(s.sdims, i_dir, i_sdt, i_stag, attr);
        std::stringstream ss;
        ss << prb;
        const std::string cpp_pstr = ss.str();
        const char *pstr = cpp_pstr.c_str();
        BENCHDNN_PRINT(1, "run: %s\n", pstr);

        res_t res {};
        const int status = doit(&prb, &res);

        bool want_perf_report = false;
        parse_result(res, want_perf_report, status, pstr);

        if (want_perf_report && is_bench_mode(PERF)) {
            perf_report_t pr(&prb, s.perf_template);
            pr.report(&res, pstr);
        }

        benchdnn_stat.tests++;
    }
}

int bench(int argc, char **argv) {
    driver_name = "prelu";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(s.dir, def.dir, argv[0])
                || parse_multi_dt(s.sdt, def.sdt, argv[0])
                || parse_multi_tag(s.stag, def.stag, argv[0])
                || parse_attr_scratchpad_mode(
                        s.scratchpad_mode, def.scratchpad_mode, argv[0])
                || parse_perf_template(s.perf_template, s.perf_template_def,
                        s.perf_template_csv, argv[0])
                || parse_reset(s, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_multi_dims(s.sdims, argv[0]);
            check_correctness(s);
        }
    }

    return parse_last_argument();
}
} // namespace prelu
