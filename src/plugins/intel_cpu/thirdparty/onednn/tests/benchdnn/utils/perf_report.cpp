/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "dnn_types.hpp"
#include "dnnl_common.hpp"

#include "utils/perf_report.hpp"

void base_perf_report_t::report(const res_t *res, const char *prb_str) const {
    dump_perf_footer();

    std::stringstream ss;

    const char *pt = pt_;
    char c;
    while ((c = *pt++) != '\0') {
        if (c != '%') {
            ss << c;
            continue;
        }
        handle_option(ss, pt, res, prb_str);
    }

    std::string str = ss.str();
    BENCHDNN_PRINT(0, "%s\n", str.c_str());
};

void base_perf_report_t::handle_option(std::ostream &s, const char *&option,
        const res_t *res, const char *prb_str) const {
    const auto &t = res->timer;
    timer::timer_t::mode_t mode = timer::timer_t::min;
    (void)mode;
    double unit = 1e0;
    char c = *option;

    if (c == '-' || c == '0' || c == '+') {
        mode = modifier2mode(c);
        c = *(++option);
    }

    if (c == 'K' || c == 'M' || c == 'G') {
        unit = modifier2unit(c);
        c = *(++option);
    }

    auto get_flops = [&]() -> double {
        if (!t.sec(mode)) return 0;
        return ops() / t.sec(mode) / unit;
    };

    auto get_bw = [&]() -> double {
        if (!t.sec(mode)) return 0;
        return (res->ibytes + res->obytes) / t.sec(mode) / unit;
    };

    auto get_freq = [&]() -> double {
        if (!t.sec(mode)) return 0;
        return t.ticks(mode) / t.sec(mode) / unit;
    };

    // Please update doc/knobs_perf_report.md in case of any new options!

#define HANDLE(opt, ...) \
    if (!strncmp(opt "%", option, strlen(opt) + 1)) { \
        __VA_ARGS__; \
        option += strlen(opt) + 1; \
        return; \
    }

    // Options operating on driver specific types, e.g. alg_t.
    HANDLE("alg", dump_alg(s));
    HANDLE("cfg", dump_cfg(s));
    HANDLE("desc", dump_desc(s));
    HANDLE("DESC", dump_desc_csv(s));
    HANDLE("engine", dump_engine(s));
    HANDLE("flags", dump_flags(s));
    HANDLE("activation", dump_rnn_activation(s));
    HANDLE("direction", dump_rnn_direction(s));
    // Options operating on common types, e.g. attr_t.
    HANDLE("attr", if (attr() && !attr()->is_def()) s << *attr());
    HANDLE("axis", if (axis()) s << *axis());
    HANDLE("dir", if (dir()) s << *dir());
    HANDLE("dt", if (dt()) s << *dt());
    HANDLE("group", if (group()) s << *group());
    HANDLE("sdt", if (sdt()) s << *sdt());
    HANDLE("stag", if (stag()) s << *stag());
    HANDLE("mb", if (user_mb()) s << *user_mb());
    HANDLE("name", if (name()) s << name());
    HANDLE("ddt", if (ddt()) s << *ddt());
    HANDLE("dtag", if (dtag()) s << *dtag());
    HANDLE("prop", if (prop()) s << prop2str(*prop()));
    HANDLE("tag", if (tag()) s << *tag());
    HANDLE("stat_tag", if (stat_tag()) s << *stat_tag());
    HANDLE("wtag", if (wtag()) s << *wtag());
    // Options operating on driver independent objects, e.g. timer values.
    HANDLE("bw", s << get_bw());
    HANDLE("driver", s << driver_name);
    HANDLE("flops", s << get_flops());
    HANDLE("clocks", s << t.ticks(mode) / unit);
    HANDLE("prb", s << prb_str);
    HANDLE("freq", s << get_freq());
    HANDLE("ops", s << ops() / unit);
    HANDLE("time", s << t.ms(mode) / unit);
    HANDLE("impl", s << res->impl_name);
    HANDLE("ibytes", s << res->ibytes / unit);
    HANDLE("obytes", s << res->obytes / unit);
    HANDLE("iobytes", s << (res->ibytes + res->obytes) / unit);
    HANDLE("idx", s << benchdnn_stat.tests);

#undef HANDLE

    auto opt_name = std::string(option);
    opt_name.pop_back();
    BENCHDNN_PRINT(0, "Error: perf report option \"%s\" is not supported\n",
            opt_name.c_str());
    SAFE_V(FAIL);
}
