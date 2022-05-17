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

#include <assert.h>
#include <cctype>
#include <cmath>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <string>

#include <algorithm>
#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#include "src/common/math_utils.hpp"

#include "common.hpp"
#include "dnn_types.hpp"
#include "dnnl_common.hpp"
#include "dnnl_debug.hpp"
#include "dnnl_memory.hpp"
#include "utils/parser.hpp"

#define BENCHDNN_DNNL_ARG_UNDEF 0

namespace tag {
const char *x {"x"};
const char *abx {"abx"};
const char *axb {"axb"};
const char *any {"any"};
const char *undef {"undef"};
} // namespace tag

// returns dims with current @p off values using actual values from @p dims
dims_t off2dims_idx(const dims_t &dims, int64_t off) {
    dims_t dims_idx;
    dims_idx.reserve(dims.size());

    for (int i = (int)dims.size() - 1; i >= 0; --i) {
        dims_idx.insert(dims_idx.begin(), off % dims[i]);
        off /= dims[i];
    }
    assert(off == 0);
    return dims_idx;
}

std::ostream &operator<<(std::ostream &s, const dims_t &dims) {
    if (!dims.empty()) s << dims[0];
    for (size_t d = 1; d < dims.size(); ++d)
        s << "x" << dims[d];
    return s;
}

std::ostream &operator<<(std::ostream &s, dir_t dir) {
#define CASE(x) \
    if (dir == (x)) return s << STRINGIFY(x)
    CASE(FWD_B);
    CASE(FWD_D);
    CASE(FWD_I);
    CASE(BWD_D);
    CASE(BWD_DW);
    CASE(BWD_W);
    CASE(BWD_WB);
#undef CASE
    SAFE_V(FAIL);
    return s;
}

std::ostream &operator<<(std::ostream &s, dnnl_data_type_t dt) {
    s << dt2str(dt);
    return s;
}

std::ostream &operator<<(std::ostream &s, dnnl_engine_kind_t ek) {
    s << engine_kind2str(ek);
    return s;
}

dir_t str2dir(const char *str) {
#define CASE(x) \
    if (!strcasecmp(STRINGIFY(x), str)) return x
    CASE(FWD_D);
    CASE(FWD_I);
    CASE(FWD_B);
    CASE(BWD_D);
    CASE(BWD_W);
    CASE(BWD_WB);
    CASE(BWD_DW);
#undef CASE
    assert(!"unknown dir");
    return DIR_UNDEF;
}

dnnl_prop_kind_t prop2prop_kind(const dir_t dir) {
    if (dir == FWD_D) return dnnl_forward_training;
    if (dir == FWD_I) return dnnl_forward_inference;
    if (dir == BWD_DW) return dnnl_backward;
    assert(!"unknown dir");
    return dnnl_prop_kind_undef;
}

const char *prop2str(dnnl_prop_kind_t prop) {
    if (prop == dnnl_forward_training) return "FWD_D";
    if (prop == dnnl_forward_inference) return "FWD_I";
    if (prop == dnnl_backward) return "BWD_DW";
    assert(!"unknown prop_kind");
    return "unknown prop_kind";
}

const char *data_kind2str(data_kind_t kind) {
    switch (kind) {
        case SRC: return "SRC";
        case WEI: return "WEI";
        case BIA: return "BIA";
        case DST: return "DST";
        case ACC: return "ACC";
        case DATA: return "DATA";
        case MEAN: return "MEAN";
        case VAR: return "VAR";
        case SS: return "SS";
        case SC: return "SC";
        case SH: return "SH";
        case GWEI: return "GWEI";
    }
    assert(!"incorrect data kind");
    return "incorrect data kind";
}

static const std::map<int, std::vector<const char *>> supported_args {
        {DNNL_ARG_SRC, {"src", "src0"}},
        {DNNL_ARG_SRC_1, {"src1"}},
        {DNNL_ARG_WEIGHTS, {"wei"}},
        {DNNL_ARG_DST, {"dst"}},
};

static int str2arg(const std::string &str) {
    for (const auto &arg : supported_args)
        for (const auto &s : arg.second)
            if (str.compare(s) == 0) return arg.first;
    return BENCHDNN_DNNL_ARG_UNDEF;
}

policy_t attr_t::str2policy(const std::string &str) {
    std::string s(str);
    // s.compare is lexicographical, case matters
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
#define CASE(_plc) \
    if (s.compare(STRINGIFY(_plc)) == 0) return _plc
    CASE(COMMON);
    CASE(PER_OC);
    CASE(PER_DIM_0);
    CASE(PER_DIM_1);
    CASE(PER_DIM_01);
    CASE(PER_MB_SPATIAL);
    CASE(PER_SPATIAL);
    CASE(PER_TENSOR);
#undef CASE
    assert(!"unknown attr_t::policy_t policy");
    return POLICY_TOTAL;
}

const char *attr_t::policy2str(policy_t policy) {
    if (policy == COMMON) return "common";
    if (policy == PER_OC) return "per_oc";
    if (policy == PER_DIM_0) return "per_dim_0";
    if (policy == PER_DIM_1) return "per_dim_1";
    if (policy == PER_DIM_01) return "per_dim_01";
    if (policy == PER_MB_SPATIAL) return "per_mb_spatial";
    if (policy == PER_SPATIAL) return "per_spatial";
    if (policy == PER_TENSOR) return "per_tensor";
    assert(!"unknown attr_t::policy_t policy");
    return "unknown attr_t::policy_t policy";
}

int attr_t::get_default_mask(policy_t policy) {
    switch (policy) {
        case PER_DIM_0: return (1 << 0);
        case PER_OC:
        case PER_DIM_1: return (1 << 1);
        case PER_DIM_01: return (1 << 0) + (1 << 1);
        case PER_MB_SPATIAL: return (1 << 0) + (1 << 2) + (1 << 3);
        case PER_SPATIAL: return (1 << 2) + (1 << 3);
        case PER_TENSOR: return (1 << DNNL_MAX_NDIMS) - 1;
        case COMMON: return 0;
        default: SAFE(FAIL, CRIT); return 0;
    }
}

// This function takes input string, extracts float value and runtime, if
// present, from the string. Updates @value and @runtime with extracted values.
int parse_value_and_runtime(float &value, bool &runtime, const std::string &s) {
    // process value
    size_t scale_pos = 0;
    value = std::stof(s, &scale_pos);
    runtime = false;
    if (scale_pos + 1 < s.size()) return FAIL;
    if (scale_pos == s.size()) return OK;
    if (s.back() != '*') return FAIL;
    runtime = true;
    return OK;
}

int attr_t::scale_t::from_str(const std::string &s) {
    *this = scale_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;
    // process policy
    this->policy = str2policy(parser::get_substr(s, start_pos, ':'));
    if (this->policy == POLICY_TOTAL) return FAIL;
    if (start_pos == std::string::npos) return OK;
    if (start_pos >= s.size()) return FAIL; // to catch dangling ':'

    SAFE(parse_value_and_runtime(this->scale, this->runtime,
                 parser::get_substr(s, start_pos, ':')),
            WARN);
    if (this->scale < 0) return FAIL;
    return OK;
}

int attr_t::zero_points_t::from_str(const std::string &s) {
    *this = zero_points_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto subs = parser::get_substr(s, start_pos, '+');
        size_t subs_pos = 0;

        auto arg = str2arg(parser::get_substr(subs, subs_pos, ':'));
        if (arg == BENCHDNN_DNNL_ARG_UNDEF || subs_pos == std::string::npos
                || subs_pos >= subs.size())
            return FAIL;

        auto policy = str2policy(parser::get_substr(subs, subs_pos, ':'));
        if (policy == POLICY_TOTAL || subs_pos == std::string::npos
                || subs_pos >= subs.size())
            return FAIL;

        float zp = 0;
        bool runtime = false;
        SAFE(parse_value_and_runtime(
                     zp, runtime, parser::get_substr(subs, subs_pos, '\0')),
                WARN);
        set(arg, policy, static_cast<int>(zp), runtime);
    }
    return OK;
}

int attr_t::arg_scales_t::from_str(const std::string &s) {
    *this = arg_scales_t();
    if (s.empty()) return OK;

    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto subs = parser::get_substr(s, start_pos, '+');
        // Special handling for really big float values
        if (subs.back() == 'e') {
            auto subs_add = parser::get_substr(s, start_pos, '+');
            subs += subs_add;
        }
        size_t subs_pos = 0;

        auto arg = str2arg(parser::get_substr(subs, subs_pos, ':'));
        if (arg == BENCHDNN_DNNL_ARG_UNDEF || subs_pos == std::string::npos
                || subs_pos >= s.size())
            return FAIL;

        scale_t arg_scale;
        SAFE(arg_scale.from_str(parser::get_substr(subs, subs_pos, '\0')),
                WARN);
        set(arg, arg_scale);
    }
    return OK;
}

using pk_t = attr_t::post_ops_t::kind_t;

struct po_table_entry_t {
    pk_t kind;
    std::vector<std::string> kind_names;
    dnnl_alg_kind_t dnnl_kind;
};

static po_table_entry_t kind_table[] = {
        // sum
        {pk_t::SUM, {"sum"}, dnnl_alg_kind_undef},
        // depthwise convolution
        {pk_t::DW_K3S1P1, {"dw_k3s1p1"}, dnnl_convolution_auto},
        {pk_t::DW_K3S2P1, {"dw_k3s2p1"}, dnnl_convolution_auto},
        // eltwise
        {pk_t::ELTWISE_START, {"eltwise_undef"}, dnnl_alg_kind_undef},
        {pk_t::ABS, {"abs", "eltwise_abs"}, dnnl_eltwise_abs},
        {pk_t::BRELU, {"bounded_relu", "eltwise_bounded_relu", "brelu"},
                dnnl_eltwise_bounded_relu},
        {pk_t::CLIP, {"clip", "eltwise_clip"}, dnnl_eltwise_clip},
        {pk_t::CLIP_V2, {"clip_v2", "eltwise_clip_v2"}, dnnl_eltwise_clip_v2},
        {pk_t::CLIP_V2_DST, {"clip_v2_dst", "eltwise_clip_v2_use_dst_for_bwd"},
                dnnl_eltwise_clip_v2_use_dst_for_bwd},
        {pk_t::ELU, {"elu", "eltwise_elu"}, dnnl_eltwise_elu},
        {pk_t::ELU_DST, {"elu_dst", "eltwise_elu_use_dst_for_bwd"},
                dnnl_eltwise_elu_use_dst_for_bwd},
        {pk_t::EXP, {"exp", "eltwise_exp"}, dnnl_eltwise_exp},
        {pk_t::EXP_DST, {"exp_dst", "eltwise_exp_use_dst_for_bwd"},
                dnnl_eltwise_exp_use_dst_for_bwd},
        {pk_t::GELU_ERF, {"gelu_erf", "eltwise_gelu_erf"},
                dnnl_eltwise_gelu_erf},
        {pk_t::GELU_TANH, {"gelu_tanh", "eltwise_gelu_tanh"},
                dnnl_eltwise_gelu_tanh},
        {pk_t::HARDSWISH, {"hardswish", "eltwise_hardswish"},
                dnnl_eltwise_hardswish},
        {pk_t::LINEAR, {"linear", "eltwise_linear"}, dnnl_eltwise_linear},
        {pk_t::LOG, {"log", "eltwise_log"}, dnnl_eltwise_log},
        {pk_t::LOGISTIC, {"logistic", "eltwise_logistic"},
                dnnl_eltwise_logistic},
        {pk_t::LOGISTIC_DST,
                {"logistic_dst", "eltwise_logistic_use_dst_for_bwd"},
                dnnl_eltwise_logistic_use_dst_for_bwd},
        {pk_t::LOGSIGMOID, {"logsigmoid", "eltwise_logsigmoid"},
                dnnl_eltwise_logsigmoid},
        {pk_t::MISH, {"mish", "eltwise_mish"}, dnnl_eltwise_mish},
        {pk_t::POW, {"pow", "eltwise_pow"}, dnnl_eltwise_pow},
        {pk_t::RELU, {"relu", "eltwise_relu"}, dnnl_eltwise_relu},
        {pk_t::RELU_DST, {"relu_dst", "eltwise_relu_use_dst_for_bwd"},
                dnnl_eltwise_relu_use_dst_for_bwd},
        {pk_t::ROUND, {"round", "eltwise_round"}, dnnl_eltwise_round},
        {pk_t::SQRT, {"sqrt", "eltwise_sqrt"}, dnnl_eltwise_sqrt},
        {pk_t::SQRT_DST, {"sqrt_dst", "eltwise_sqrt_use_dst_for_bwd"},
                dnnl_eltwise_sqrt_use_dst_for_bwd},
        {pk_t::SQUARE, {"square", "eltwise_square"}, dnnl_eltwise_square},
        {pk_t::SRELU, {"soft_relu", "eltwise_soft_relu", "srelu"},
                dnnl_eltwise_soft_relu},
        {pk_t::SWISH, {"swish", "eltwise_swish"}, dnnl_eltwise_swish},
        {pk_t::TANH, {"tanh", "eltwise_tanh"}, dnnl_eltwise_tanh},
        {pk_t::TANH_DST, {"tanh_dst", "eltwise_tanh_use_dst_for_bwd"},
                dnnl_eltwise_tanh_use_dst_for_bwd},
        {pk_t::ELTWISE_END, {"eltwise_undef"}, dnnl_alg_kind_undef},
        // binary
        {pk_t::BINARY_START, {"binary_undef"}, dnnl_alg_kind_undef},
        {pk_t::ADD, {"add", "binary_add"}, dnnl_binary_add},
        {pk_t::DIV, {"div", "binary_div"}, dnnl_binary_div},
        {pk_t::EQ, {"eq", "binary_eq"}, dnnl_binary_eq},
        {pk_t::GE, {"ge", "binary_ge"}, dnnl_binary_ge},
        {pk_t::GT, {"gt", "binary_gt"}, dnnl_binary_gt},
        {pk_t::LE, {"le", "binary_le"}, dnnl_binary_le},
        {pk_t::LT, {"lt", "binary_lt"}, dnnl_binary_lt},
        {pk_t::MAX, {"max", "binary_max"}, dnnl_binary_max},
        {pk_t::MIN, {"min", "binary_min"}, dnnl_binary_min},
        {pk_t::MUL, {"mul", "binary_mul"}, dnnl_binary_mul},
        {pk_t::NE, {"ne", "binary_ne"}, dnnl_binary_ne},
        {pk_t::SUB, {"sub", "binary_sub"}, dnnl_binary_sub},
        {pk_t::BINARY_END, {"binary_undef"}, dnnl_alg_kind_undef},
        // prelu
        {pk_t::PRELU, {"prelu"}, dnnl_alg_kind_undef},
        // guard entry
        {pk_t::KIND_TOTAL, {"kind_undef"}, dnnl_alg_kind_undef}};

pk_t attr_t::post_ops_t::str2kind(const std::string &str) {
    std::string s(str);
    // string::operator== is lexicographical, case matters
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    for (const auto &e : kind_table) {
        for (const auto &name : e.kind_names) {
            if (s == name) return e.kind;
        }
    }
    assert(!"unknown attr_t::post_ops_t::kind_t kind");
    const auto table_size = sizeof(kind_table) / sizeof(*kind_table);
    return kind_table[table_size - 1].kind;
}

const char *attr_t::post_ops_t::kind2str(pk_t kind) {
    for (const auto &e : kind_table) {
        if (e.kind == kind) return e.kind_names[0].c_str();
    }
    assert(!"unknown attr::post_ops::kind");
    const auto table_size = sizeof(kind_table) / sizeof(*kind_table);
    return kind_table[table_size - 1].kind_names[0].c_str();
}

dnnl_alg_kind_t attr_t::post_ops_t::kind2dnnl_kind(pk_t kind) {
    for (const auto &e : kind_table) {
        if (e.kind == kind) return e.dnnl_kind;
    }
    assert(!"unknown attr::post_ops::kind");
    const auto table_size = sizeof(kind_table) / sizeof(*kind_table);
    return kind_table[table_size - 1].dnnl_kind;
}

std::vector<std::pair<int, int>> attr_t::post_ops_t::get_po_masks() const {
    std::vector<std::pair<int, int>> v_masks;
    for (int idx = 0; idx < len(); ++idx) {
        const auto &e = this->entry[idx];
        policy_t policy = policy_t::COMMON;
        int arg = BENCHDNN_DNNL_ARG_UNDEF;
        if (e.is_binary_kind()) {
            policy = e.binary.policy;
            arg = DNNL_ARG_SRC_1;
        } else if (e.is_prelu_kind()) {
            policy = e.prelu.policy;
            arg = DNNL_ARG_WEIGHTS;
        } else
            continue;

        const auto mask = attr_t::get_default_mask(policy);
        v_masks.emplace_back(std::make_pair(
                DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | arg, mask));
    }
    return v_masks;
}

std::vector<int> attr_t::post_ops_t::get_binary_po_masks() const {
    std::vector<int> v_masks;
    for (int idx = 0; idx < len(); ++idx) {
        const auto &e = this->entry[idx];
        if (!e.is_binary_kind()) continue;

        const auto policy = e.binary.policy;
        const auto mask = attr_t::get_default_mask(policy);
        v_masks.push_back(mask);
    }
    return v_masks;
}

int attr_t::post_ops_t::from_str(const std::string &s) {
    *this = post_ops_t();
    if (s.empty()) return OK;

    // TODO: remove me after a while
    if (s.front() == '\'' || s.back() == '\'') {
        BENCHDNN_PRINT(0, "%s\n",
                "ERROR: `--attr-post-ops` no longer requires opening and "
                "closing `'` (and `\"` for CLI) quotes. Please discard them to "
                "proceed with successful parsing.");
        return FAIL;
    } else if (s.find_first_of(";", 0) != std::string::npos) {
        BENCHDNN_PRINT(0, "%s\n",
                "ERROR: `--attr-post-ops` no longer accepts `;` as post-ops "
                "delimiter. Please use `+` as a delimiter between several "
                "post-ops, i.e. `--attr-post-ops=sum+relu`.");
        return FAIL;
    }

    size_t start_pos = 0;
    while (start_pos != std::string::npos) {
        auto subs = parser::get_substr(s, start_pos, '+');
        size_t subs_pos = 0;

        auto kind = str2kind(parser::get_substr(subs, subs_pos, ':'));
        if (kind == KIND_TOTAL) return FAIL;

        entry.emplace_back(kind);
        if (subs_pos == std::string::npos) continue;
        if (subs_pos >= subs.size()) return FAIL; // to catch dangling ':'

        auto &e = entry.back();
        if (e.is_sum_kind()) {
            e.sum.scale = std::stof(parser::get_substr(subs, subs_pos, ':'));
            if (subs_pos == std::string::npos) continue;
            if (subs_pos >= subs.size()) return FAIL; // to catch dangling ':'

            auto zp_str = parser::get_substr(subs, subs_pos, ':');
            e.sum.zero_point = std::stoi(zp_str);
            if (subs_pos == std::string::npos) continue;
            if (subs_pos >= subs.size()) return FAIL; // to catch dangling ':'
            if (std::to_string(e.sum.zero_point) != zp_str) return FAIL;

            e.sum.dt = str2dt(parser::get_substr(subs, subs_pos, ':').c_str());
            // sum dt, if specified, should be defined
            if (e.sum.dt == dnnl_data_type_undef) return FAIL;
        } else if (e.is_convolution_kind()) {
            e.convolution.dst_dt
                    = str2dt(parser::get_substr(subs, subs_pos, ':').c_str());
            if (e.convolution.dst_dt == dnnl_data_type_undef) return FAIL;
            if (subs_pos == std::string::npos) continue;
            if (subs_pos >= subs.size()) return FAIL; // to catch dangling ':'

            auto scale_str = parser::get_substr(subs, subs_pos, '+');
            SAFE(e.convolution.oscale.from_str(scale_str), WARN);
        } else if (e.is_eltwise_kind()) {
            e.eltwise.alpha
                    = std::stof(parser::get_substr(subs, subs_pos, ':'));
            if (subs_pos == std::string::npos) continue;
            if (subs_pos >= subs.size()) return FAIL; // to catch dangling ':'

            e.eltwise.beta = std::stof(parser::get_substr(subs, subs_pos, ':'));
            if (subs_pos == std::string::npos) continue;
            if (subs_pos >= subs.size()) return FAIL; // to catch dangling ':'

            e.eltwise.scale
                    = std::stof(parser::get_substr(subs, subs_pos, ':'));
            if (e.eltwise.scale <= 0) return FAIL;
        } else if (e.is_binary_kind()) {
            e.binary.src1_dt
                    = str2dt(parser::get_substr(subs, subs_pos, ':').c_str());
            if (e.binary.src1_dt == dnnl_data_type_undef) return FAIL;
            if (subs_pos == std::string::npos) continue;
            if (subs_pos >= subs.size()) return FAIL; // to catch dangling ':'

            e.binary.policy
                    = str2policy(parser::get_substr(subs, subs_pos, ':'));
            if (e.binary.policy == POLICY_TOTAL) return FAIL;
            if (subs_pos == std::string::npos) continue;
            if (subs_pos >= subs.size()) return FAIL; // to catch dangling ':'

            e.binary.tag = parser::get_substr(subs, subs_pos, ':');
            SAFE(check_tag(e.binary.tag), WARN);
        } else if (e.is_prelu_kind()) {
            e.prelu.policy
                    = str2policy(parser::get_substr(subs, subs_pos, ':'));
            if (e.prelu.policy == POLICY_TOTAL) return FAIL;
            if (subs_pos == std::string::npos) continue;
            if (subs_pos >= subs.size()) return FAIL; // to catch dangling ':'
        }
        if (subs_pos == std::string::npos) continue;
        if (subs_pos >= subs.size()) return FAIL; // to catch dangling ':'
    }
    return OK;
}

bool attr_t::is_def() const {
    return oscale.is_def() && scales.is_def() && zero_points.is_def()
            && post_ops.is_def()
            && scratchpad_mode == dnnl_scratchpad_mode_library;
}

int attr_t::post_ops_t::find(pk_t kind, int start, int stop) const {
    if (stop == -1) stop = len();
    stop = MIN2(stop, len());
    for (int idx = start; idx < stop; ++idx)
        if (entry[idx].kind == kind) return idx;
    return -1;
}

bool attr_t::post_ops_t::entry_t::is_sum_kind() const {
    return kind == SUM;
}
bool attr_t::post_ops_t::entry_t::is_convolution_kind() const {
    return kind == DW_K3S1P1 || kind == DW_K3S2P1;
}
bool attr_t::post_ops_t::entry_t::is_eltwise_kind() const {
    return kind > ELTWISE_START && kind < ELTWISE_END;
}
bool attr_t::post_ops_t::entry_t::is_binary_kind() const {
    return kind > pk_t::BINARY_START && kind < pk_t::BINARY_END;
}
bool attr_t::post_ops_t::entry_t::is_prelu_kind() const {
    return kind == PRELU;
}

int attr_t::post_ops_t::convolution_index() const {
    for (int i = 0; i < len(); ++i) {
        if (entry[i].is_convolution_kind()) return i;
    }
    return -1;
}

int attr_t::post_ops_t::eltwise_index() const {
    for (int i = 0; i < len(); ++i) {
        if (entry[i].is_eltwise_kind()) return i;
    }
    return -1;
}

int attr_t::post_ops_t::binary_index() const {
    for (int i = 0; i < len(); ++i) {
        if (entry[i].is_binary_kind()) return i;
    }
    return -1;
}

int attr_t::post_ops_t::prelu_index() const {
    for (int i = 0; i < len(); ++i) {
        if (entry[i].is_prelu_kind()) return i;
    }
    return -1;
}

int str2attr(attr_t *attr, const char *str) {
    if (attr == nullptr || str == nullptr) return FAIL;

    *attr = attr_t();
    std::string s(str), entry_name;
    size_t start_pos = 0, end_pos = start_pos;

    entry_name = "oscale=";
    start_pos = s.find(entry_name);
    if (start_pos != std::string::npos) {
        start_pos += entry_name.size();
        // ';' is an attribute delimeter
        auto status
                = attr->oscale.from_str(parser::get_substr(s, start_pos, ';'));
        if (status != OK) return status;
    }

    entry_name = "zero_points=";
    start_pos = s.find(entry_name);
    if (start_pos != std::string::npos) {
        start_pos += entry_name.size();
        auto status = attr->zero_points.from_str(
                parser::get_substr(s, start_pos, ';'));
        if (status != OK) return status;
    }

    entry_name = "scales=";
    start_pos = s.find(entry_name);
    if (start_pos != std::string::npos) {
        start_pos += entry_name.size();
        auto status
                = attr->scales.from_str(parser::get_substr(s, start_pos, ';'));
        if (status != OK) return status;
    }

    entry_name = "post_ops=";
    start_pos = s.find(entry_name);
    if (start_pos != std::string::npos) {
        start_pos += entry_name.size();
        // "\'\'" separates post-ops from everything else
        end_pos = s.find_first_of('\'', s.find_first_of('\'', start_pos) + 1);
        // +1 to include '\'' as an essential part of post-ops
        auto parse_str = s.substr(start_pos, end_pos - start_pos + 1);
        auto status = attr->post_ops.from_str(parse_str);
        if (status != OK) return status;
    }

    return OK;
}

void handle_legacy_attr(attr_t &attr, const attr_t &legacy_attr) {
    if (legacy_attr.is_def()) return;

    if (!attr.is_def()) {
        BENCHDNN_PRINT(0, "%s\n",
                "ERROR: both `--attr=` and one of `--attr-post-ops=`, "
                "`--attr-scales=`, `--attr-zero-points=` or `--attr-oscale=` "
                "options are specified, please use latter options.");
        SAFE_V(FAIL);
    }

    attr = legacy_attr;
}

std::ostream &operator<<(std::ostream &s, const policy_t &policy) {
    s << attr_t::policy2str(policy);
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::scale_t &scale) {
    s << scale.policy << ":" << scale.scale;
    if (scale.runtime) s << '*';
    return s;
}

std::ostream &operator<<(
        std::ostream &s, const attr_t::zero_points_t &zero_points) {
    const char *delim = "";
    for (const auto &point : zero_points.points) {
        s << delim;
        s << supported_args.at(point.first)[0] << ":" << point.second.policy
          << ":" << point.second.value;
        if (point.second.runtime) s << '*';
        delim = "+";
    }

    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::arg_scales_t &scales) {
    const char *delim = "";
    for (const auto &v : scales.scales) {
        if (!v.second.is_def()) {
            s << delim;
            s << supported_args.at(v.first)[0] << ":" << v.second;
            delim = "+";
        }
    }
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t::kind_t &k) {
    s << attr_t::post_ops_t::kind2str(k);
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t::post_ops_t &post_ops) {
    for (int idx = 0; idx < post_ops.len(); ++idx) {
        if (idx > 0) s << "+";

        const auto &e = post_ops.entry[idx];
        s << e.kind;

        if (e.is_sum_kind()) {
            if (e.sum.scale != 1.0f || e.sum.zero_point != 0
                    || e.sum.dt != dnnl_data_type_undef)
                s << ":" << e.sum.scale;
            if (e.sum.zero_point != 0 || e.sum.dt != dnnl_data_type_undef)
                s << ":" << e.sum.zero_point;
            if (e.sum.dt != dnnl_data_type_undef) s << ":" << e.sum.dt;
        } else if (e.is_convolution_kind()) {
            const auto &co = e.convolution.oscale;
            if (e.convolution.dst_dt != dnnl_f32 || !co.is_def())
                s << ":" << e.convolution.dst_dt;
            if (!co.is_def()) s << ":" << co;
        } else if (e.is_eltwise_kind()) {
            if (e.eltwise.scale != 1.f)
                s << ":" << e.eltwise.alpha << ":" << e.eltwise.beta << ":"
                  << e.eltwise.scale;
            else if (e.eltwise.beta != 0.f)
                s << ":" << e.eltwise.alpha << ":" << e.eltwise.beta;
            else if (e.eltwise.alpha != 0.f)
                s << ":" << e.eltwise.alpha;
        } else if (e.is_binary_kind()) {
            s << ":" << e.binary.src1_dt;
            if (e.binary.policy != policy_t::COMMON) {
                s << ":" << e.binary.policy;
                if (attr_t::get_default_mask(e.binary.policy) >= 4)
                    s << ":" << e.binary.tag;
            }
        } else if (e.is_prelu_kind()) {
            if (e.prelu.policy != policy_t::COMMON) {
                s << ":" << e.prelu.policy;
            }
        } else {
            assert(!"unknown kind");
            s << "unknown_kind";
        }
    }

    return s;
}

std::ostream &operator<<(std::ostream &s, dnnl_scratchpad_mode_t sm) {
    s << scratchpad_mode2str(sm);
    return s;
}

std::ostream &operator<<(std::ostream &s, const attr_t &attr) {
    if (!attr.is_def()) {
        if (!attr.oscale.is_def()) s << "--attr-oscale=" << attr.oscale << " ";
        if (!attr.scales.is_def()) s << "--attr-scales=" << attr.scales << " ";
        if (!attr.zero_points.is_def())
            s << "--attr-zero-points=" << attr.zero_points << " ";
        if (!attr.post_ops.is_def())
            s << "--attr-post-ops=" << attr.post_ops << " ";
        if (attr.scratchpad_mode != dnnl_scratchpad_mode_library)
            s << "--attr-scratchpad=" << attr.scratchpad_mode << " ";
    }
    return s;
}

std::ostream &operator<<(std::ostream &s, bench_mode_t mode) {
    if (is_bench_mode(RUN) && !(is_bench_mode(CORR) || is_bench_mode(PERF)))
        s << "R";
    if (is_bench_mode(CORR)) s << "C";
    if (is_bench_mode(PERF)) s << "P";
    if (is_bench_mode(LIST)) s << "L";
    return s;
}

std::ostream &dump_global_params(std::ostream &s) {
    s << "--" << driver_name << " ";
    if (canonical) s << "--canonical=" << bool2str(canonical) << " ";
    if (canonical || engine_tgt_kind != dnnl_cpu) {
        s << "--engine=" << engine_tgt_kind;
        if (engine_index != 0) s << ":" << engine_index;
        s << " ";
    }
    if (canonical || fast_ref_gpu != true)
        s << "--fast-ref-gpu=" << bool2str(fast_ref_gpu) << " ";
    if (!skip_impl.empty()) s << "--skip-impl=" << skip_impl << " ";
    if (canonical || mem_check != true)
        s << "--mem-check=" << bool2str(mem_check) << " ";
    if (canonical || allow_enum_tags_only != true)
        s << "--allow-enum-tags-only=" << bool2str(allow_enum_tags_only) << " ";
    if (canonical || hints.get() != isa_hints_t::none)
        s << "--cpu-isa-hints=" << isa_hints_t::hints2str(hints) << " ";
    if (canonical || bench_mode != CORR) s << "--mode=" << bench_mode << " ";
    if (canonical || attr_same_pd_check != false)
        s << "--attr-same-pd-check=" << bool2str(attr_same_pd_check) << " ";

    return s;
}

dnnl_engine_kind_t str2engine_kind(const char *str) {
    const char *param = "cpu";
    if (!strncasecmp(param, str, strlen(param))) return dnnl_cpu;

    param = "gpu";
    if (!strncasecmp(param, str, strlen(param))) return dnnl_gpu;

    assert(!"not expected");
    return dnnl_cpu;
}

dnnl_scratchpad_mode_t str2scratchpad_mode(const char *str) {
    const char *param = "library";
    if (!strncasecmp(param, str, strlen(param)))
        return dnnl_scratchpad_mode_library;

    param = "user";
    if (!strncasecmp(param, str, strlen(param)))
        return dnnl_scratchpad_mode_user;

    assert(!"not expected");
    return dnnl_scratchpad_mode_library;
}

void attr_args_t::prepare_output_scales(
        const attr_t &attr, const void *vals, int64_t count, int mask) {
    insert(DNNL_ARG_ATTR_OUTPUT_SCALES, vals, count, mask, attr.oscale.runtime);
}

struct post_ops_rhs_tensor_entry_t {
    dnnl_data_type_t dt;
    policy_t policy;
    std::string tag;
    int arg_attr_mask;
};

namespace {

post_ops_rhs_tensor_entry_t get_po_rhs_tensor_entry(
        const attr_t::post_ops_t::entry_t &entry) {
    if (entry.is_prelu_kind()) {
        const auto &prelu = entry.prelu;
        return {dnnl_f32, prelu.policy, tag::axb, DNNL_ARG_WEIGHTS};
    } else if (entry.is_binary_kind()) {
        const auto &binary = entry.binary;
        return {binary.src1_dt, binary.policy, binary.tag, DNNL_ARG_SRC_1};
    }

    return post_ops_rhs_tensor_entry_t {};
}

} // namespace

int attr_args_t::prepare_post_ops_mds(
        const attr_t &attr, int ndims, const dnnl_dims_t dims) {
    const auto &po = attr.post_ops;
    // iterate over all post ops and prepare md for each binary
    for (int idx = 0; idx < po.len(); ++idx) {
        const auto &e = po.entry[idx];
        if (e.is_binary_kind() || e.is_prelu_kind()) {

            const auto po_rhs_tensor_entry = get_po_rhs_tensor_entry(e);
            const int mask
                    = attr_t::get_default_mask(po_rhs_tensor_entry.policy);

            // deduce binary, prelu dims based on input policy
            dnnl_dims_t rhs_tensor_dims = {};
            for (auto d = 0; d < ndims; ++d)
                rhs_tensor_dims[d] = (!(mask & (1 << d))) ? 1 : dims[d];

            dnnl_memory_desc_t rhs_tensor_desc;
            SAFE(init_md(&rhs_tensor_desc, ndims, rhs_tensor_dims,
                         po_rhs_tensor_entry.dt, po_rhs_tensor_entry.tag),
                    WARN);
            mds.emplace((DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)
                                | po_rhs_tensor_entry.arg_attr_mask),
                    rhs_tensor_desc);
        }
    }

    return OK;
}

void attr_args_t::prepare_dw_post_op(const attr_t &attr,
        dnnl_data_type_t wei_dt, dnnl_data_type_t bia_dt, const void *vals,
        int64_t count, int mask) {
    const int dw_idx = attr.post_ops.convolution_index();
    if (dw_idx == -1) return;

    const auto &dw = attr.post_ops.entry[dw_idx].convolution;
    // insert output scale which applies in fused convolution
    insert(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_ATTR_OUTPUT_SCALES, vals, count,
            mask, dw.oscale.runtime);

    dw_entry.wei_dt = wei_dt;
    dw_entry.bia_dt = bia_dt;
}

dnnl_primitive_attr_t create_dnnl_attr(
        const attr_t &attr, const attr_args_t &attr_args) {
    dnnl_primitive_attr_t dnnl_attr = nullptr;
    DNN_SAFE_V(dnnl_primitive_attr_create(&dnnl_attr));

    if (!attr.oscale.is_def()) {
        const auto &os_args = attr_args.get(DNNL_ARG_ATTR_OUTPUT_SCALES);
        const auto &policy = attr.oscale.policy;

        const auto count = os_args.get_count(policy);
        const auto mask = os_args.get_mask(policy);
        const auto scales = os_args.get_float_ptr();

        DNN_SAFE_V(dnnl_primitive_attr_set_output_scales(
                dnnl_attr, count, mask, scales));
    } else if (!attr.scales.is_def()) {
        for (const auto &arg : supported_args) {
            const auto arg_name = arg.first;
            const auto &as = attr.scales;
            if (as.is_def(arg_name)) continue;

            const auto &e = as.get(arg_name);
            // Only common policy is supported in the library at this point
            int64_t count = 1;
            int mask = attr_t::get_default_mask(e.policy);
            const float *scales = e.runtime ? &DNNL_RUNTIME_F32_VAL : &e.scale;

            DNN_SAFE_V(dnnl_primitive_attr_set_scales(
                    dnnl_attr, arg_name, count, mask, scales));
        }
    }

    if (!attr.zero_points.is_def()) {
        for (const auto &arg : supported_args) {
            const auto arg_name = arg.first;
            const auto &zp = attr.zero_points;
            if (zp.is_def(arg_name)) continue;

            const auto &e = zp.get(arg_name);
            // Only common policy/single RT value are supported in the library
            // at this point
            int64_t count = 1;
            int mask = attr_t::get_default_mask(e.policy);
            const auto values = e.runtime ? &DNNL_RUNTIME_S32_VAL : &e.value;

            DNN_SAFE_V(dnnl_primitive_attr_set_zero_points(
                    dnnl_attr, arg_name, count, mask, values));
        }
    }

    if (!attr.post_ops.is_def()) {
        dnnl_post_ops_t ops;
        DNN_SAFE_V(dnnl_post_ops_create(&ops));

        const auto &po = attr.post_ops;
        for (int idx = 0; idx < po.len(); ++idx) {
            const auto &e = po.entry[idx];
            if (e.is_sum_kind()) {
                DNN_SAFE_V(dnnl_post_ops_append_sum_v3(
                        ops, e.sum.scale, e.sum.zero_point, e.sum.dt));
            } else if (e.is_convolution_kind()) {
                const auto wei_dt = attr_args.get_dw_arg(DNNL_ARG_WEIGHTS);
                const auto bia_dt = attr_args.get_dw_arg(DNNL_ARG_BIAS);

                const auto &os_args = attr_args.get(
                        DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_ATTR_OUTPUT_SCALES);
                const auto scales = os_args.get_float_ptr();
                const auto &policy = e.convolution.oscale.policy;
                // API expects count=0 if output_scale was not set
                const auto count = scales ? os_args.get_count(policy) : 0;
                const auto mask = os_args.get_mask(policy);

                const auto dnnl_post_ops_append_dw = e.convolution.stride == 1
                        ? dnnl_post_ops_append_dw_k3s1p1
                        : dnnl_post_ops_append_dw_k3s2p1;
                DNN_SAFE_V(dnnl_post_ops_append_dw(ops, wei_dt, bia_dt,
                        e.convolution.dst_dt, count, mask, scales));
            } else if (e.is_eltwise_kind()) {
                DNN_SAFE_V(dnnl_post_ops_append_eltwise(ops, e.eltwise.scale,
                        e.eltwise.alg, e.eltwise.alpha, e.eltwise.beta));
            } else if (e.is_binary_kind()) {
                const auto &src1_md = attr_args.get_md(
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx) | DNNL_ARG_SRC_1));
                assert(src1_md.ndims != 0);
                DNN_SAFE_V(dnnl_post_ops_append_binary(
                        ops, e.binary.alg, &src1_md));
            } else if (e.is_prelu_kind()) {
                const auto &policy = e.prelu.policy;
                const auto mask = attr_t::get_default_mask(policy);
                DNN_SAFE_V(dnnl_post_ops_append_prelu(ops, mask));
            } else {
                assert(!"unknown attr::post_ops::kind");
            }
        }
        DNN_SAFE_V(dnnl_primitive_attr_set_post_ops(dnnl_attr, ops));

        const_dnnl_post_ops_t c_ops;
        DNN_SAFE_V(dnnl_primitive_attr_get_post_ops(dnnl_attr, &c_ops));
        SAFE_V(dnnl_post_ops_len(c_ops) == po.len() ? OK : FAIL);

        DNN_SAFE_V(dnnl_post_ops_destroy(ops));
    }

    DNN_SAFE_V(dnnl_primitive_attr_set_scratchpad_mode(
            dnnl_attr, attr.scratchpad_mode));

    return dnnl_attr;
}

// Exception free version of std::stoi, sets idx to 0 and returns 0 in case of
// error.
static int stoi_safe(const std::string &s, size_t *idx) {
    if (s.empty() || !std::isdigit(s[0])) {
        *idx = 0;
        return 0;
    }
    return std::stoi(s, idx);
}

static bool is_abc_tag(const std::string &tag) {
    if (tag == tag::undef || tag == tag::any) return true;

    bool mask[DNNL_MAX_NDIMS] = {};
    for (auto &c : tag) {
        if (!std::isalpha(c)) continue;
        int idx = std::tolower(c) - 'a';
        if (idx < 0 || idx >= DNNL_MAX_NDIMS) return false;
        mask[idx] = true;
    }
    // Check there are no gaps, e.g. [1 1 1 1 0 0 ...].
    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        if (mask[i]) continue;
        for (int j = i + 1; j < DNNL_MAX_NDIMS; j++)
            if (mask[j]) return false;
        break;
    }
    return true;
}

int check_abc_tag(const std::string &tag_, bool check_enum_tags_only) {
    if (tag_.empty()) return FAIL;
    if (!is_abc_tag(tag_)) return FAIL;
    if (check_enum_tags_only) {
        if (str2fmt_tag(tag_.c_str()) == dnnl_format_tag_last) return FAIL;
        return OK;
    }

    enum class dim_state_t { undef = 0, upper, lower, lower_with_block };
    dim_state_t dim_states[DNNL_MAX_NDIMS] = {};
    bool in_inner_block = false;
    auto tag = tag_;
    while (!tag.empty()) {
        // Parse block size if presented.
        size_t idx;
        int block = stoi_safe(tag, &idx);
        if (block == 0 && idx != 0) return FAIL;
        if (idx == 0) block = 0;
        if (block > 0) in_inner_block = true;

        // Move to the first position after the block.
        tag = tag.substr(idx);
        if (tag.empty()) return FAIL;

        char c = tag[0];
        bool is_lower = ('a' <= c && c <= 'a' + DNNL_MAX_NDIMS - 1);
        bool is_upper = ('A' <= c && c <= 'A' + DNNL_MAX_NDIMS - 1);
        if (!is_lower && !is_upper) return FAIL;

        // Uppercase cannot be with block.
        if (is_upper && block != 0) return FAIL;
        // Block sizes are required within inner block.
        if (block == 0 && in_inner_block) return FAIL;

        // Check rules related to lowercase/uppercase/block order.
        int dim_idx = std::tolower(c) - 'a';
        dim_state_t prev_state = dim_states[dim_idx];
        dim_state_t cur_state = is_upper
                ? dim_state_t::upper
                : block != 0 ? dim_state_t::lower_with_block
                             : dim_state_t::lower;

        switch (cur_state) {
            case dim_state_t::upper:
            case dim_state_t::lower:
                // Letter without block must be the first.
                if (prev_state != dim_state_t::undef) return FAIL;
                break;
            case dim_state_t::lower_with_block:
                // Letter with block must be after uppercase or after a letter
                // with block.
                if (prev_state != dim_state_t::upper
                        && prev_state != dim_state_t::lower_with_block)
                    return FAIL;
                break;
            default: assert(!"not expected");
        }

        // Update state, move to the next position.
        dim_states[dim_idx] = cur_state;
        tag = tag.substr(1);
    }

    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        // Uppercase letter must be followed by lowercase.
        if (dim_states[i] == dim_state_t::upper) return FAIL;

        // Ensure there are no gaps (e.g. acd).
        if (dim_states[i] == dim_state_t::undef) {
            for (int j = i + 1; j < DNNL_MAX_NDIMS; j++)
                if (dim_states[j] != dim_state_t::undef) return FAIL;
            break;
        }
    }

    return OK;
}

static std::string trim_letter(const std::string &tag_, char c) {
    auto tag = tag_;
    auto pos = tag.find(c);
    if (pos == std::string::npos) return tag;

    tag.replace(pos, 1, "");
    if (pos == 0) return tag;

    pos--;
    while (std::isdigit(tag[pos])) {
        tag.replace(pos, 1, "");
        if (pos == 0) break;
        pos--;
    }
    return tag;
}

// Removes extra dimensions from a tag according to ndims.
static std::string trim_tag(const std::string &tag, int ndims) {
    std::string trimmed_tag = tag;
    for (char c = 'a' + ndims; c <= 'a' + (char)(DNNL_MAX_NDIMS - 1); c++) {
        trimmed_tag = trim_letter(trimmed_tag, c);
        trimmed_tag = trim_letter(trimmed_tag, std::toupper(c));
    }
    return trimmed_tag;
}

// Tries to map a tag to an abc-tag according to a logical tag. For example:
// nchw -> abcd.
static std::string try_map_tag(
        const std::string &logical_tag, const std::string &tag, int *nmatched) {
    // Check if all the required letters are presented.
    for (auto &c : logical_tag) {
        if (std::toupper(c) == c
                && tag.find(std::tolower(c)) == std::string::npos)
            return {};
    }

    // Check that all letters are known and assign indices to letters.
    int logical_indices[DNNL_MAX_NDIMS] = {};
    for (auto &c : tag) {
        if (!std::isalpha(c)) continue;

        auto lower_pos = logical_tag.find(std::tolower(c));
        auto upper_pos = logical_tag.find(std::toupper(c));
        auto pos = (lower_pos == std::string::npos ? upper_pos : lower_pos);
        if (pos == std::string::npos) return {};

        logical_indices[pos] = 1;
    }

    for (int i = 0, idx = 0; i < (int)logical_tag.size(); i++) {
        if (logical_indices[i] == 0) continue;
        logical_indices[i] = idx++;
    }

    (*nmatched)++;
    std::string mapped_tag = tag;
    for (int i = 0; i < (int)tag.size(); i++) {
        char c = tag[i];
        if (!std::isalpha(tag[i])) continue;
        auto pos = logical_tag.find(std::tolower(c));
        if (pos == std::string::npos) pos = logical_tag.find(std::toupper(c));

        mapped_tag[i]
                = (char)(tag[i] - std::tolower(c) + 'a' + logical_indices[pos]);
    }
    return mapped_tag;
}

// Maps a tag to an abc-tag.
static std::string map_tag_letters(const std::string &tag) {
    int nmatched = 0;

    // Mapping rules:
    // - Uppercase letters are mandatory
    // - Lowercase letters are optional
    auto tag_goidhw = try_map_tag("GOIdhw", tag, &nmatched);
    auto tag_oidhw = try_map_tag("OIdhw", tag, &nmatched);
    auto tag_ncdhw = try_map_tag("NCdhw", tag, &nmatched);
    auto tag_tnc = try_map_tag("TNc", tag, &nmatched);
    auto tag_ldnc = try_map_tag("LDNC", tag, &nmatched);
    auto tag_ldigo = try_map_tag("LDigO", tag, &nmatched);

    if (nmatched == 0) return tag;
    if (nmatched > 1) assert(!"Not expected: ambiguous tag.");

    if (!tag_goidhw.empty()) return tag_goidhw;
    if (!tag_oidhw.empty()) return tag_oidhw;
    if (!tag_ncdhw.empty()) return tag_ncdhw;
    if (!tag_tnc.empty()) return tag_tnc;
    if (!tag_ldnc.empty()) return tag_ldnc;
    if (!tag_ldigo.empty()) return tag_ldigo;

    return tag;
}

std::string normalize_tag(const std::string &tag_, int ndims) {
    std::string tag = tag_;
    if (tag == tag::undef || tag == tag::any || ndims == 0) return tag;
    if (tag == tag::x) {
        if (ndims >= 0) assert(ndims == 1);
        return "a";
    }

    // Handle meta-tags (abx, axb, etc).
    auto pos = tag.find("x");
    if (pos != std::string::npos) {
        // Non-grouped tags will start `x` from `c`, but grouped will most of
        // times start `x` from `d`.
        char start_x = 'c';
        for (char c = 'a' + DNNL_MAX_NDIMS - 1; c >= 'b'; c--) {
            if (tag.find(c) != std::string::npos) {
                start_x = c + 1;
                break;
            }
        }
        // Adjust ndims if they are not specified.
        int meta_ndims = (ndims == -1 ? (start_x - 'a' + 1) : ndims);
        std::string tail;
        for (int i = 0; i < meta_ndims - (start_x - 'a'); i++)
            tail += (start_x + i);
        return trim_tag(tag.replace(pos, 1, tail), meta_ndims);
    }

    return map_tag_letters(tag);
}

int check_tag(const std::string &tag_, bool check_enum_tags_only) {
    auto tag = normalize_tag(tag_);
    if (tag == tag::undef || tag == tag::any) return OK;
    return check_abc_tag(tag, check_enum_tags_only);
}

void maybe_oscale(const attr_t &attr, float &d, float *scales, int64_t oc) {
    if (!attr.oscale.is_def()) {
        int64_t idx = attr.oscale.policy == policy_t::COMMON ? 0 : oc;
        d *= scales[idx];
    }
}

void maybe_zero_point(const attr_t &attr, float &d, const int32_t *zero_points,
        int64_t c, int arg, bool opposite_zero_point) {
    if (attr.zero_points.is_def()) return;

    const auto &e = attr.zero_points.get(arg);
    if (!e.is_def()) {
        const int idx = e.policy == policy_t::COMMON ? 0 : c;
        const int zp_sign = opposite_zero_point ? -1 : 1;
        d -= zp_sign * zero_points[idx];
    }
}

float compute_eltwise_fwd(
        pk_t kind, float src, float scale, float alpha, float beta) {
    // don't compute on nan, propagate it
    if (std::isnan(src)) return NAN;

    using namespace dnnl::impl::math;

    switch (kind) {
        case pk_t::RELU: return scale * relu_fwd(src, alpha);
        case pk_t::TANH: return scale * tanh_fwd(src);
        case pk_t::ELU: return scale * elu_fwd(src, alpha);
        case pk_t::SQUARE: return scale * square_fwd(src);
        case pk_t::ABS: return scale * abs_fwd(src);
        case pk_t::SQRT: return scale * sqrt_fwd(src);
        case pk_t::LINEAR: return scale * linear_fwd(src, alpha, beta);
        case pk_t::BRELU: return scale * bounded_relu_fwd(src, alpha);
        case pk_t::SRELU: return scale * soft_relu_fwd(src);
        case pk_t::LOGSIGMOID: return scale * logsigmoid_fwd(src);
        case pk_t::MISH: return scale * mish_fwd(src);
        case pk_t::LOGISTIC: return scale * logistic_fwd(src);
        case pk_t::EXP: return scale * exp_fwd(src);
        case pk_t::GELU_TANH: return scale * gelu_tanh_fwd(src);
        case pk_t::SWISH: return scale * swish_fwd(src, alpha);
        case pk_t::LOG: return scale * log_fwd(src);
        case pk_t::CLIP: return scale * clip_fwd(src, alpha, beta);
        case pk_t::CLIP_V2: return scale * clip_v2_fwd(src, alpha, beta);
        case pk_t::POW: return scale * pow_fwd(src, alpha, beta);
        case pk_t::GELU_ERF: return scale * gelu_erf_fwd(src);
        case pk_t::ROUND: return scale * round_fwd(src);
        case pk_t::HARDSWISH: return scale * hardswish_fwd(src);
        case pk_t::RELU_DST: return scale * relu_fwd(src, alpha);
        case pk_t::TANH_DST: return scale * tanh_fwd(src);
        case pk_t::ELU_DST: return scale * elu_fwd(src, alpha);
        case pk_t::SQRT_DST: return scale * sqrt_fwd(src);
        case pk_t::LOGISTIC_DST: return scale * logistic_fwd(src);
        case pk_t::EXP_DST: return scale * exp_fwd(src);
        case pk_t::CLIP_V2_DST: return scale * clip_v2_fwd(src, alpha, beta);

        default: assert(!"unknown attr::post_ops::kind");
    };
    return NAN;
}

float compute_eltwise_bwd(
        pk_t kind, float d_dst, float src, float alpha, float beta) {
    using namespace dnnl::impl::math;

    switch (kind) {
        case pk_t::RELU: return relu_bwd(d_dst, src, alpha);
        case pk_t::TANH: return tanh_bwd(d_dst, src);
        case pk_t::ELU: return elu_bwd(d_dst, src, alpha);
        case pk_t::SQUARE: return square_bwd(d_dst, src);
        case pk_t::ABS: return abs_bwd(d_dst, src);
        case pk_t::SQRT: return sqrt_bwd(d_dst, src);
        case pk_t::LINEAR: return linear_bwd(d_dst, src, alpha, beta);
        case pk_t::BRELU: return bounded_relu_bwd(d_dst, src, alpha);
        case pk_t::SRELU: return soft_relu_bwd(d_dst, src);
        case pk_t::LOGSIGMOID: return logsigmoid_bwd(d_dst, src);
        case pk_t::MISH: return mish_bwd(d_dst, src);
        case pk_t::LOGISTIC: return logistic_bwd(d_dst, src);
        case pk_t::EXP: return exp_bwd(d_dst, src);
        case pk_t::GELU_TANH: return gelu_tanh_bwd(d_dst, src);
        case pk_t::SWISH: return swish_bwd(d_dst, src, alpha);
        case pk_t::LOG: return log_bwd(d_dst, src);
        case pk_t::CLIP: return clip_bwd(d_dst, src, alpha, beta);
        case pk_t::CLIP_V2: return clip_v2_bwd(d_dst, src, alpha, beta);
        case pk_t::POW: return pow_bwd(d_dst, src, alpha, beta);
        case pk_t::GELU_ERF: return gelu_erf_bwd(d_dst, src);
        case pk_t::HARDSWISH: return hardswish_bwd(d_dst, src);

        case pk_t::RELU_DST: return relu_bwd_use_dst(d_dst, src, alpha);
        case pk_t::TANH_DST: return tanh_bwd_use_dst(d_dst, src);
        case pk_t::ELU_DST: return elu_bwd_use_dst(d_dst, src, alpha);
        case pk_t::SQRT_DST: return sqrt_bwd_use_dst(d_dst, src);
        case pk_t::LOGISTIC_DST: return logistic_bwd_use_dst(d_dst, src);
        case pk_t::EXP_DST: return exp_bwd_use_dst(d_dst, src);
        case pk_t::CLIP_V2_DST:
            return clip_v2_bwd_use_dst(d_dst, src, alpha, beta);

        default: assert(!"unknown attr::post_ops::kind");
    }
    return NAN;
}

float compute_binary(pk_t kind, float src0, float src1) {
    // don't compute on nan, propagate it
    if (std::isnan(src0) || std::isnan(src1)) return NAN;

    if (kind == pk_t::ADD) {
        return src0 + src1;
    } else if (kind == pk_t::MUL) {
        return src0 * src1;
    } else if (kind == pk_t::MAX) {
        return MAX2(src0, src1);
    } else if (kind == pk_t::MIN) {
        return MIN2(src0, src1);
    } else if (kind == pk_t::DIV) {
        return src0 / src1;
    } else if (kind == pk_t::SUB) {
        return src0 - src1;
    } else if (kind == pk_t::GE) {
        return src0 >= src1;
    } else if (kind == pk_t::GT) {
        return src0 > src1;
    } else if (kind == pk_t::LE) {
        return src0 <= src1;
    } else if (kind == pk_t::LT) {
        return src0 < src1;
    } else if (kind == pk_t::EQ) {
        return src0 == src1;
    } else if (kind == pk_t::NE) {
        return src0 != src1;
    } else {
        assert(!"operation not supported!");
    }
    return NAN;
}

void maybe_post_ops(const attr_t &attr, float &val, float sum_val,
        const std::vector<float> &v_po_vals) {
    using namespace dnnl::impl::math;

    auto it_po = v_po_vals.begin();
    const auto &po = attr.post_ops;
    for (int idx = 0; idx < po.len(); ++idx) {
        const auto &e = po.entry[idx];

        if (e.is_sum_kind()) {
            val += e.sum.scale * (sum_val - e.sum.zero_point);
        } else if (e.is_convolution_kind()) {
            continue;
        } else if (e.is_eltwise_kind()) {
            const auto &s = e.eltwise.scale;
            const auto &a = e.eltwise.alpha;
            const auto &b = e.eltwise.beta;
            val = compute_eltwise_fwd(e.kind, val, s, a, b);
        } else if (e.is_binary_kind()) {
            val = compute_binary(e.kind, val, *it_po);
            it_po++;
        } else if (e.is_prelu_kind()) {
            val = val > 0 ? val : val * (*it_po);
            it_po++;
        }
    }
}

void update_cpu_ref_attrs(attr_t &attr, dnnl_data_type_t new_dt) {
    auto &os = attr.oscale;
    os.runtime = false;

    auto &po = attr.post_ops;
    for (int idx = 0; idx < po.len(); ++idx) {
        auto &e = po.entry[idx];
        if (!e.is_binary_kind()) continue;

        e.binary.src1_dt = new_dt;
    }
}

#undef BENCHDNN_DNNL_ARG_UNDEF
