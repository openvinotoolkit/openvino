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

#include <algorithm>
#include <sstream>
#include <string>
#include <utility>

#include "reorder/reorder.hpp"
#include "utils/parser.hpp"

namespace reorder {

flag_t str2flag(const char *str) {
    std::string s(str);
    if (s.empty()) return std::make_pair(FLAG_NONE, 0);

    size_t start_pos = 0;
    // format of single entry is `flag_bit:mask`
    auto sub = parser::get_substr(s, start_pos, ':');
    std::transform(sub.begin(), sub.end(), sub.begin(), ::tolower);

    flag_bit_t flag = FLAG_NONE;
    if (sub.compare("s8s8_comp") == 0)
        flag = FLAG_S8S8_COMP;
    else if (sub.compare("zp_comp") == 0)
        flag = FLAG_ZP_COMP;
    else {
        assert(!"unknown flag");
        SAFE_V(FAIL);
    }

    int mask = std::stoi(parser::get_substr(s, start_pos, ':'));
    if (mask < 0) {
        fprintf(stderr,
                "ERROR: reorder driver: `mask` should be non-negative.\n"),
                fflush(stderr);
        SAFE_V(FAIL);
    }

    return std::make_pair(flag, mask);
}

std::string flag_name2str(flag_bit_t flag) {
    if (flag == FLAG_S8S8_COMP) return "s8s8_comp";
    if (flag == FLAG_ZP_COMP) return "zp_comp";
    assert(!"unsupported flag");
    return "";
}

std::ostream &operator<<(std::ostream &s, const std::vector<flag_t> &oflag) {
    const char *delim = "";
    for (const auto &i_oflag : oflag) {
        s << delim << flag_name2str(i_oflag.first) << ":" << i_oflag.second;
        delim = "+";
    }
    return s;
}

cross_engine_t str2cross_engine(const char *str) {
    if (!strcasecmp("none", str)) return NONE;
    if (!strcasecmp("cpu2gpu", str)) return CPU2GPU;
    if (!strcasecmp("gpu2cpu", str)) return GPU2CPU;
    assert(!"unknown cross engine");
    return NONE;
}

const char *cross_engine2str(cross_engine_t cross_engine) {
    switch (cross_engine) {
        case NONE: return "none";
        case CPU2GPU: return "cpu2gpu";
        case GPU2CPU: return "gpu2cpu";
        default: assert(!"unknown cross engine"); return "unknown cross engine";
    }
}

float *prb_t::generate_oscales() {
    const attr_t::scale_t &oscale = this->attr.oscale;
    const int mask = attr_t::get_default_mask(oscale.policy);

    int64_t uniq_scales = 1;
    for (int d = 0; d < this->ndims; ++d)
        if (mask & (1 << d)) uniq_scales *= this->reorder.dims[d];

    float *scales = (float *)zmalloc(sizeof(float) * uniq_scales, 64);
    SAFE_V(scales != nullptr ? OK : FAIL);
    for (int d = 0; d < uniq_scales; ++d)
        scales[d] = oscale.scale;
    if (uniq_scales > 1) scales[uniq_scales - 1] /= 2.f;
    return scales;
}

int32_t *prb_t::generate_zero_points(int arg) const {
    const attr_t::zero_points_t &zero_points = this->attr.zero_points;
    if (zero_points.is_def(arg)) return nullptr;

    const auto &e = zero_points.get(arg);
    assert(e.policy == policy_t::COMMON);

    int32_t *zp = (int32_t *)zmalloc(sizeof(int32_t), 4);
    SAFE_V(zp != nullptr ? OK : FAIL);
    zp[0] = e.value;
    return zp;
}

std::ostream &operator<<(std::ostream &s, const prb_t &prb) {
    dump_global_params(s);
    settings_t def;

    s << "--sdt=" << cfg2dt(prb.conf_in) << " ";
    s << "--ddt=" << cfg2dt(prb.conf_out) << " ";
    s << "--stag=" << prb.reorder.tag_in << " ";
    s << "--dtag=" << prb.reorder.tag_out << " ";

    if (canonical || (!prb.oflag.empty() && prb.oflag != def.oflag[0]))
        s << "--oflag=" << prb.oflag << " ";
    if (canonical || prb.cross_engine != def.cross_engine[0])
        s << "--cross-engine=" << cross_engine2str(prb.cross_engine) << " ";
    if (canonical || prb.runtime_dim_mask != def.runtime_dim_mask[0])
        s << "--runtime-dim-mask=" << prb.runtime_dim_mask << " ";

    s << prb.attr;
    s << prb.reorder.dims;

    return s;
}

} // namespace reorder
