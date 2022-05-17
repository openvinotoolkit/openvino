/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef PERF_REPORT_HPP
#define PERF_REPORT_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "oneapi/dnnl/dnnl_types.h"

#include "common.hpp"
#include "utils/timer.hpp"

struct base_perf_report_t {
    base_perf_report_t(const char *perf_template) : pt_(perf_template) {}
    virtual ~base_perf_report_t() = default;

    void report(const res_t *res, const char *prb_str) const;

    /* truly common types */
    virtual double ops() const { return 0.; }
    virtual const attr_t *attr() const { return nullptr; }
    virtual const int *axis() const { return nullptr; }
    virtual const char *name() const { return nullptr; }
    virtual const int64_t *group() const { return nullptr; }
    virtual const dir_t *dir() const { return nullptr; }
    virtual const dnnl_data_type_t *dt() const { return nullptr; }
    virtual const std::vector<dnnl_data_type_t> *sdt() const { return nullptr; }
    virtual const dnnl_data_type_t *ddt() const { return nullptr; }
    virtual const std::string *tag() const { return nullptr; }
    virtual const std::string *stat_tag() const { return nullptr; }
    virtual const std::vector<std::string> *stag() const { return nullptr; }
    virtual const std::string *dtag() const { return nullptr; }
    virtual const std::string *wtag() const { return nullptr; }
    virtual const dnnl_prop_kind_t *prop() const { return nullptr; }
    virtual const int64_t *user_mb() const { return nullptr; }

    /* designed to be overloaded in reorder only to match verbose output */
    virtual void dump_engine(std::ostream &s) const { s << engine_tgt_kind; }

    /* primitive-specific properties (but with common interface) */
    virtual void dump_alg(std::ostream &) const { SAFE_V(FAIL); }
    virtual void dump_cfg(std::ostream &) const { SAFE_V(FAIL); }
    virtual void dump_desc(std::ostream &) const { SAFE_V(FAIL); }
    virtual void dump_desc_csv(std::ostream &) const { SAFE_V(FAIL); }
    virtual void dump_flags(std::ostream &) const { SAFE_V(FAIL); }
    virtual void dump_rnn_activation(std::ostream &) const { SAFE_V(FAIL); }
    virtual void dump_rnn_direction(std::ostream &) const { SAFE_V(FAIL); }

private:
    const char *pt_;

    void handle_option(std::ostream &s, const char *&option, const res_t *res,
            const char *prb_str) const;

    void dump_perf_footer() const {
        static bool footer_printed = false;
        if (!footer_printed) {
            BENCHDNN_PRINT(0, "Output template: %s\n", pt_);
            footer_printed = true;
        }
    }

    static timer::timer_t::mode_t modifier2mode(char c) {
        if (c == '-') return timer::timer_t::min;
        if (c == '0') return timer::timer_t::avg;
        if (c == '+') return timer::timer_t::max;
        return timer::timer_t::min;
    }

    static double modifier2unit(char c) {
        if (c == 'K') return 1e3;
        if (c == 'M') return 1e6;
        if (c == 'G') return 1e9;
        return 1e0;
    }
};

#endif
