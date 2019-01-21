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

#include <assert.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>

#include "mkldnn.h"

#include "common.hpp"
#include "dnn_types.hpp"
#include "mkldnn_common.hpp"
#include "mkldnn_debug.hpp"

dir_t str2dir(const char *str) {
#define CASE(x) if (!strcasecmp(STRINGIFY(x), str)) return x
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

const char *dir2str(dir_t dir) {
#define CASE(x) if (dir == x) return STRINGIFY(x)
    CASE(FWD_D);
    CASE(FWD_I);
    CASE(FWD_B);
    CASE(BWD_D);
    CASE(BWD_DW);
    CASE(BWD_W);
    CASE(BWD_WB);
#undef CASE
    assert(!"unknown dir");
    return "DIR_UNDEF";
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
    case GWEI: return "GWEI";
    }
    assert(!"incorrect data kind");
    return "incorrect data kind";
}

data_kind_t fmt2data_kind(mkldnn_memory_format_t fmt) {
    switch (fmt) {
    case mkldnn_x:
    case mkldnn_nc:
    case mkldnn_tnc:
    case mkldnn_ntc:

    case mkldnn_ncw:
    case mkldnn_nwc:
    case mkldnn_nCw16c:

    case mkldnn_nchw:
    case mkldnn_nhwc:
    case mkldnn_chwn:
    case mkldnn_nChw8c:
    case mkldnn_nChw16c:

    case mkldnn_ncdhw:
    case mkldnn_ndhwc:
    case mkldnn_nCdhw16c:
        return DATA;

    case mkldnn_goiw:
    case mkldnn_gOIw16i16o:
    case mkldnn_gOIw16o16i:
    case mkldnn_gOiw16o:
    case mkldnn_gOwi16o:
    case mkldnn_gOIw8i16o2i:
    case mkldnn_goihw:
    case mkldnn_hwigo:
    case mkldnn_hwigo_s8s8:
    case mkldnn_gOIhw8i8o:
    case mkldnn_gOIhw16i16o:
    case mkldnn_gOIhw4i16o4i:
    case mkldnn_gOIhw4i16o4i_s8s8:
    case mkldnn_gOIhw8i16o2i:
    case mkldnn_gOIdhw8i16o2i:
    case mkldnn_gOIhw8o16i2o:
    case mkldnn_gOIhw8o8i:
    case mkldnn_gOIhw16o16i:
    case mkldnn_gIOhw16o16i:
    case mkldnn_gOihw8o:
    case mkldnn_gOihw16o:
    case mkldnn_gOhwi8o:
    case mkldnn_gOhwi16o:
    case mkldnn_Goihw8g:
    case mkldnn_Goihw16g:
    case mkldnn_gOhIw16o4i:
    case mkldnn_goidhw:
    case mkldnn_gOIdhw16i16o:
    case mkldnn_gOIdhw16o16i:
    case mkldnn_gOidhw16o:
    case mkldnn_gOdhwi16o:
        return GWEI;

    default: return WEI;
    }
}

attr_t::scale_t::policy_t attr_t::scale_t::str2policy(const char *str) {
#define CASE(_plc) if (!strcasecmp(STRINGIFY(_plc), str)) return _plc
    CASE(NONE);
    CASE(COMMON);
    CASE(PER_OC);
#undef CASE
    assert(!"unknown attr::scale::policy");
    return NONE;
}

const char *attr_t::scale_t::policy2str(attr_t::scale_t::policy_t policy) {
    if (policy == NONE) return "none";
    if (policy == COMMON) return "common";
    if (policy == PER_OC) return "per_oc";
    assert(!"unknown attr::scale::policy");
    return "unknown attr::scale::policy";
}

int attr_t::scale_t::str2scale(const char *str, const char **end_s) {
    *this = attr_t::scale_t();

    if (str == NULL) return FAIL;

    const char *s_;
    const char * &s = end_s ? *end_s : s_;
    s = str;

    for (policy_t p = NONE; true; p = (policy_t)((int)p + 1)) {
        if (p == POLICY_TOTAL) return FAIL;

        const char *ps = policy2str(p);
        if (!strncasecmp(ps, s, strlen(ps))) {
            this->policy = p;
            s += strlen(ps);
            break;
        }
    }

    if (*s != ':') return OK;
    s++;

    char *end;
    this->scale = strtof(s, &end);
    if (this->scale < 0 || end == s) return FAIL;

    s = end;
    assert(*s == '\0' || *s == ';');

    return OK;
}

void attr_t::scale_t::scale2str(char *buffer, char **end_b) const {
    assert(buffer);
    buffer += sprintf(buffer, "%s:%g", policy2str(this->policy), this->scale);
    if (end_b) *end_b = buffer;
}

attr_t::post_ops_t::kind_t attr_t::post_ops_t::str2kind(const char *str) {
#define CASE(_knd) if (!strcasecmp(STRINGIFY(_knd), str)) return _knd
    CASE(SUM);
    CASE(RELU);
#undef CASE
    assert(!"unknown attr::post_ops::kind");
    return KIND_TOTAL;
}

const char *attr_t::post_ops_t::kind2str(attr_t::post_ops_t::kind_t kind) {
    if (kind == SUM) return "sum";
    if (kind == RELU) return "relu";
    assert(!"unknown attr::post_ops::kind");
    return "unknown attr::post_ops::kind";
}

int attr_t::post_ops_t::from_str(const char *str, const char **end_s) {
    *this = post_ops_t();

    if (str == NULL || *str != '\'') return FAIL;

    const char *s_;
    const char * &s = end_s ? *end_s : s_;
    s = str;

    ++s;
    for (;;) {
        if (*s == '\'') { ++s; return OK; }
        if (len == capacity) return FAIL;

        for (kind_t k = SUM; true; k = (kind_t)((int)k + 1)) {
            if (k == KIND_TOTAL) return FAIL;

            const char *ks = kind2str(k);
            if (!strncasecmp(ks, s, strlen(ks))) {
                auto &e = entry[len];

                e.kind = k;
                s += strlen(ks);
                if (k == SUM) {
                    if (*s == ':') {
                        char *end;
                        e.sum.scale = strtof(++s, &end);
                        if (e.sum.scale <= 0 || end == s) return FAIL;
                        s = end;
                    } else {
                        e.sum.scale = 1.f;
                    }
                } else if (k == RELU) {
                    e.eltwise.scale = 1.f;
                    e.eltwise.alpha = e.eltwise.beta = 0.f;
                }

                break;
            }
        }
        ++len;

        if (*s == ';') ++s;
    }

    return FAIL; /* unreachable */
}

void attr_t::post_ops_t::to_str(char *buffer, char **end_b) const {
    assert(buffer);

    buffer += sprintf(buffer, "'");
    for (int idx = 0; idx < len; ++idx) {
        buffer += sprintf(buffer, "%s", idx > 0 ? ";" : "");
        const auto &e = entry[idx];

        switch (e.kind) {
        case SUM:
            buffer += sprintf(buffer, "%s:%g", kind2str(e.kind), e.sum.scale);
            break;
        case RELU:
            buffer += sprintf(buffer, "%s", kind2str(e.kind));
            break;
        default:
            assert(!"unknown kind");
            buffer += sprintf(buffer, "unknown_kind");
        }
    }
    buffer += sprintf(buffer, "'");
    if (end_b) *end_b = buffer;
}

bool attr_t::is_def() const {
    return true
        && irmode == round_mode_t::NEAREST
        && oscale.is_def()
        && post_ops.is_def();
}

int str2attr(attr_t *attr, const char *str) {
    if (attr == NULL || str == NULL) return FAIL;
    *attr = attr_t();

    const char *s = str;

    while (*s != '\0') {
        int rc = FAIL;
        const char *param;

        param = "irmode=";
        if (!strncasecmp(param, s, strlen(param))) {
            s += strlen(param);
            attr->irmode = (attr_t::round_mode_t)str2rmode(s);
            s += strlen(rmode2str((mkldnn_round_mode_t)attr->irmode));
            rc = OK;
        }

        param = "oscale=";
        if (!strncasecmp(param, s, strlen(param))) {
            s += strlen(param);
            rc = attr->oscale.str2scale(s, &s);
            if (rc != OK) return rc;
        }

        param = "post_ops=";
        if (!strncasecmp(param, s, strlen(param))) {
            s += strlen(param);
            rc = attr->post_ops.from_str(s, &s);
            if (rc != OK) return rc;
        }

        if (rc != OK) return FAIL;
        if (*s == ';') ++s;
    }

    return OK;
}

void attr2str(const attr_t *attr, char *buffer) {
    buffer += sprintf(buffer, "irmode=%s",
            rmode2str((mkldnn_round_mode_t)attr->irmode));
    buffer += sprintf(buffer, ";oscale=");
    attr->oscale.scale2str(buffer, &buffer);
    buffer += sprintf(buffer, ";post_ops=");
    attr->post_ops.to_str(buffer, &buffer);
}

mkldnn_primitive_attr_t create_mkldnn_attr(const attr_t &attr, int scale_cnt,
        int scale_mask, const float *scales) {
    mkldnn_primitive_attr_t mkldnn_attr = NULL;
    DNN_SAFE_V(mkldnn_primitive_attr_create(&mkldnn_attr));

    if (attr.irmode != attr_t::round_mode_t::NEAREST)
        DNN_SAFE_V(mkldnn_primitive_attr_set_int_output_round_mode(mkldnn_attr,
                    (mkldnn_round_mode_t)attr.irmode));

    if (!attr.oscale.is_def()) {
        using P = attr_t::scale_t::policy_t;
        int count = attr.oscale.policy == P::COMMON ? 1 : scale_cnt;
        if (scale_mask == -1)
            scale_mask = attr.oscale.policy == P::PER_OC ? 1 << 1 : 0;

        float *gen_scs = NULL;
        if (scales == NULL) {
            gen_scs = (float *)zmalloc(count * sizeof(float), 64);
            SAFE_V(gen_scs != NULL ? OK : FAIL);
            for (int i = 0; i < count; ++i)
                gen_scs[i] = attr.oscale.scale;
            scales = gen_scs;
        }

        DNN_SAFE_V(mkldnn_primitive_attr_set_output_scales(mkldnn_attr, count,
                    scale_mask, scales));

        if (gen_scs)
            zfree(gen_scs);
    }

    if (!attr.post_ops.is_def()) {
        mkldnn_post_ops_t ops;
        DNN_SAFE_V(mkldnn_post_ops_create(&ops));
        for (int idx = 0; idx < attr.post_ops.len; ++idx) {
            const auto &e = attr.post_ops.entry[idx];
            switch (attr.post_ops.entry[idx].kind) {
            case attr_t::post_ops_t::SUM:
                DNN_SAFE_V(mkldnn_post_ops_append_sum(ops, e.sum.scale));
                break;
            case attr_t::post_ops_t::RELU:
                DNN_SAFE_V(mkldnn_post_ops_append_eltwise(ops, e.eltwise.scale,
                            mkldnn_eltwise_relu, e.eltwise.alpha,
                            e.eltwise.beta));
                break;
            default:
                assert(!"unknown attr::post_ops::kind");
            }
        }
        DNN_SAFE_V(mkldnn_primitive_attr_set_post_ops(mkldnn_attr, ops));

        const_mkldnn_post_ops_t c_ops;
        DNN_SAFE_V(mkldnn_primitive_attr_get_post_ops(mkldnn_attr, &c_ops));
        SAFE_V(mkldnn_post_ops_len(c_ops) == attr.post_ops.len ? OK : FAIL);

        DNN_SAFE_V(mkldnn_post_ops_destroy(ops));
    }

    return mkldnn_attr;
}

mkldnn_memory_format_t get_default_format(int ndims, data_kind_t kind) {
    switch(kind) {
    case DATA: return (ndims == 5)
        ? mkldnn_ncdhw
        : (ndims == 4)
        ? mkldnn_nchw
        : mkldnn_ncw;
    case GWEI: return (ndims == 6)
        ? mkldnn_goidhw
        : (ndims == 5)
        ? mkldnn_goihw
        : mkldnn_goiw;
    case WEI: return (ndims == 5)
        ? mkldnn_oidhw
        : (ndims == 4)
        ? mkldnn_oihw
        : mkldnn_oiw;
    default:
        assert(!"unknown kind");
    }
    return mkldnn_format_undef;
}
