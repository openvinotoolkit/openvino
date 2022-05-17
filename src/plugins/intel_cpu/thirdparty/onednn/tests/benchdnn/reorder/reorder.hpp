/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
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

#ifndef REORDER_HPP
#define REORDER_HPP

#include <iostream>

#include "oneapi/dnnl/dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/perf_report.hpp"

namespace reorder {

enum flag_bit_t {
    FLAG_NONE = 0x0U,
    FLAG_S8S8_COMP = 0x1U,
    FLAG_ZP_COMP = 0x2U,
};
using flag_t = std::pair<flag_bit_t, int>;
flag_t str2flag(const char *str);
std::string flag2str(flag_bit_t flag);

struct dt_conf_s {
    dnnl_data_type_t dt;
    int min;
    int max;
};
typedef const dt_conf_s *dt_conf_t;
dt_conf_t dt2cfg(dnnl_data_type_t dt);
dnnl_data_type_t cfg2dt(dt_conf_t cfg);

struct reorder_conf_t {
    dims_t dims;
    std::string tag_in, tag_out;
};

struct q10n_conf_t {
    dt_conf_t conf_in;
    dt_conf_t conf_out;
    /* TODO: add attrs */
    policy_t policy;
    float scale;
};

enum cross_engine_t { NONE, CPU2GPU, GPU2CPU };
cross_engine_t str2cross_engine(const char *str);
const char *cross_engine2str(cross_engine_t cross_engine);

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    dims_t dims;

    std::vector<dnnl_data_type_t> sdt {dnnl_f32}, ddt {dnnl_f32};
    std::vector<std::string> stag {tag::abx}, dtag {tag::abx};
    std::vector<float> def_scale {0.125, 0.25, 0.5, 1, 2, 4, 8};
    std::vector<std::vector<flag_t>> oflag {{{FLAG_NONE, 0}}};
    std::vector<unsigned> runtime_dim_mask {0};
    std::vector<cross_engine_t> cross_engine {NONE};
    std::vector<attr_t::scale_t> oscale {attr_t::scale_t()};
    std::vector<attr_t::zero_points_t> zero_points {attr_t::zero_points_t()};
    std::vector<attr_t::post_ops_t> post_ops {attr_t::post_ops_t()};
    std::vector<dnnl_scratchpad_mode_t> scratchpad_mode {
            dnnl_scratchpad_mode_library};
    attr_t attr = {};

    const char *perf_template_csv
            = "perf,%engine%,%impl%,%sdt%,%ddt%,%stag%,%dtag%,%flags%,%attr%,%"
              "DESC%,%-time%,%0time%";
    const char *perf_template_def
            = "perf,%engine%,%impl%,%prb%,%-time%,%0time%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t {
    prb_t(const reorder_conf_t &res, const dt_conf_t &conf_in,
            const dt_conf_t &conf_out, const attr_t &attr,
            const std::vector<flag_t> &oflag, cross_engine_t cross_engine,
            unsigned runtime_dim_mask, float scale)
        : reorder(res)
        , conf_in(conf_in)
        , conf_out(conf_out)
        , attr(attr)
        , oflag(oflag)
        , cross_engine(cross_engine)
        , runtime_dim_mask(runtime_dim_mask)
        , ndims((int)reorder.dims.size()) {
        this->attr.oscale.scale = scale;
        scales = generate_oscales();
        src_zp = generate_zero_points(DNNL_ARG_SRC);
        dst_zp = generate_zero_points(DNNL_ARG_DST);
    }
    ~prb_t() {
        if (scales) zfree(scales);
        if (src_zp) zfree(src_zp);
        if (dst_zp) zfree(dst_zp);
    }

    const reorder_conf_t reorder;
    dt_conf_t conf_in;
    dt_conf_t conf_out;
    attr_t attr;
    std::vector<flag_t> oflag;
    cross_engine_t cross_engine;
    unsigned runtime_dim_mask;
    int ndims;
    float *scales;
    int32_t *src_zp, *dst_zp;

    bool is_reorder_with_compensation() const {
        return !oflag.empty() && oflag[0].first != FLAG_NONE;
    }
    float *generate_oscales();
    int32_t *generate_zero_points(int arg) const;
};
std::ostream &operator<<(std::ostream &s, const prb_t &prb);
std::ostream &operator<<(std::ostream &s, const std::vector<flag_t> &oflag);

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template)
        , p_(prb)
        , sdt_({cfg2dt(p_->conf_in)})
        , ddt_(cfg2dt(p_->conf_out))
        , stag_({normalize_tag(p_->reorder.tag_in, p_->ndims)})
        , dtag_(normalize_tag(p_->reorder.tag_out, p_->ndims)) {}

    void dump_desc(std::ostream &s) const override { s << p_->reorder.dims; }

    void dump_desc_csv(std::ostream &s) const override {
        s << p_->reorder.dims;
    }

    void dump_engine(std::ostream &s) const override {
        if (p_->cross_engine == CPU2GPU)
            s << "cpu2gpu";
        else if (p_->cross_engine == GPU2CPU)
            s << "gpu2cpu";
        else
            base_perf_report_t::dump_engine(s);
    }

    void dump_flags(std::ostream &s) const override { s << p_->oflag; }

    const attr_t *attr() const override { return &p_->attr; }
    const std::vector<dnnl_data_type_t> *sdt() const override { return &sdt_; }
    const dnnl_data_type_t *ddt() const override { return &ddt_; }
    const std::vector<std::string> *stag() const override { return &stag_; }
    const std::string *dtag() const override { return &dtag_; }

private:
    const prb_t *p_;
    std::vector<dnnl_data_type_t> sdt_;
    dnnl_data_type_t ddt_;
    std::vector<std::string> stag_;
    std::string dtag_;
};

int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace reorder

#endif
