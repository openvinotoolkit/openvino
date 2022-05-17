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

#ifndef PRELU_HPP
#define PRELU_HPP

#include "dnnl.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"
#include "utils/perf_report.hpp"

namespace prelu {

struct settings_t {
    settings_t() = default;

    // ctor to save certain fields from resetting
    settings_t(const char *perf_template) : settings_t() {
        this->perf_template = perf_template;
    }

    std::vector<dims_t> sdims;

    std::vector<dir_t> dir {FWD_D};
    std::vector<std::vector<dnnl_data_type_t>> sdt {{dnnl_f32, dnnl_f32}};
    std::vector<std::vector<std::string>> stag {{tag::abx, tag::any}};
    std::vector<dnnl_scratchpad_mode_t> scratchpad_mode {
            dnnl_scratchpad_mode_library};

    const char *perf_template_csv
            = "perf,%engine%,%impl%,%dir%,%sdt%,%stag%,%DESC%,%-time%,%0time%";
    const char *perf_template_def
            = "perf,%engine%,%impl%,%prb%,%-time%,%0time%";
    const char *perf_template = perf_template_def;

    void reset() { *this = settings_t(perf_template); }
};

struct prb_t {
    prb_t(const std::vector<dims_t> &sdims, dir_t dir,
            const std::vector<dnnl_data_type_t> &sdt,
            const std::vector<std::string> &stag, const attr_t &attr)
        : sdims(sdims)
        , dir(dir)
        , sdt(sdt)
        , stag(stag)
        , attr(attr)
        , ndims((int)sdims[0].size()) {}
    ~prb_t() {}

    std::vector<dims_t> sdims;
    dir_t dir;
    std::vector<dnnl_data_type_t> sdt;
    std::vector<std::string> stag;
    attr_t attr;
    int ndims;

    int get_broadcast_mask() const {
        const dims_t &src = this->sdims[0];
        const dims_t &wei = this->sdims[1];

        int broadcast_mask = 0;
        for (int d = 0; d < ndims; ++d)
            broadcast_mask += src[d] == wei[d] ? (1 << d) : 0;
        return broadcast_mask;
    }
};

std::ostream &operator<<(std::ostream &s, const prb_t &prb);

struct perf_report_t : public base_perf_report_t {
    perf_report_t(const prb_t *prb, const char *perf_template)
        : base_perf_report_t(perf_template), prb_(prb), stag_({}) {
        for (size_t d = 0; d < prb_->stag.size(); d++)
            stag_.push_back(normalize_tag(prb_->stag[d], prb_->ndims));
    }

    void dump_desc(std::ostream &s) const override { s << prb_->sdims; }

    void dump_desc_csv(std::ostream &s) const override { s << prb_->sdims; }

    const dir_t *dir() const override { return &prb_->dir; }
    const std::vector<dnnl_data_type_t> *sdt() const override {
        return &prb_->sdt;
    }
    const std::vector<std::string> *stag() const override { return &stag_; }

private:
    const prb_t *prb_;
    std::vector<std::string> stag_;
};

int setup_prelu_po(const_dnnl_primitive_desc_t pd, std::vector<int> &args,
        std::vector<dnn_mem_t> &ref_mem, std::vector<dnn_mem_t> &prim_mem,
        const dnnl_engine_t &ref_engine = get_test_engine());
void compute_ref_fwd(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &weights, dnn_mem_t &dst);
void compute_ref_bwd(const prb_t *prb, const dnn_mem_t &src,
        const dnn_mem_t &weights, dnn_mem_t &diff_src,
        const dnn_mem_t &diff_dst, dnn_mem_t &diff_weights);
int doit(const prb_t *prb, res_t *res);
int bench(int argc, char **argv);

} // namespace prelu

#endif
