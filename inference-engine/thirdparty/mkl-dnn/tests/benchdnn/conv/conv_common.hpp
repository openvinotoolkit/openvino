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

#ifndef _CONV_COMMON_HPP
#define _CONV_COMMON_HPP

#include <stdint.h>
#include <limits.h>
#include <assert.h>

#include "common.hpp"
#include "dnn_types.hpp"
#include "mkldnn_common.hpp"
#include "mkldnn_memory.hpp"

namespace conv {

enum alg_t { DIRECT, WINO };
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);

enum merge_t { NONE, RELU, };
merge_t str2merge(const char *str);
const char *merge2str(merge_t merge);

struct desc_t {
    int g, mb;
    int ic, id, ih, iw;
    int oc, od, oh, ow;
    int kd, kh, kw;
    int sd, sh, sw;
    int pd, ph, pw;
    int dd, dh, dw;

    const char *name;
};
const size_t max_desc_len = 196;
int str2desc(desc_t *desc, const char *str, bool is_deconv);
void desc2str(const desc_t *d, char *buffer, bool canonical = false);

/** configuration structure, that controls initial data filling + error check
 *
 * dt defines convolution precision
 *
 * for each type (SRC, WEI, BIA, and DST) the values are filled as follows:
 * if (rand() > f_sparsity) then:
 *     v <-- f_base // it is guaranteed each kernel window
 *                  // has at least one non-zero element
 * else:
 *     v <-- f_min + rand() * f_step % (f_max - f_min)
 *
 *
 * on final check the resulting values should be in [min .. max] range, the
 * relative difference should not exceed eps
 */
typedef struct dt_conf_t {
    mkldnn_data_type_t dt;
    double min, max; /* representative */
    int f_min, f_max; /* fill range */
    int f_base; /* fill base, use 0 */
    int f_step; /* fill step, use 1 */
    double f_sparsity; /* amount of non-zeros, default 0.25 */
    double eps; /* acceptable error */
} _dt_conf_t[DAT_TOTAL];

extern const _dt_conf_t conf_f32;
extern const _dt_conf_t conf_f32_full;
extern const _dt_conf_t conf_f32_wino;
extern const _dt_conf_t conf_s16s16s32s32;
extern const _dt_conf_t conf_s32s16s16s32;
extern const _dt_conf_t conf_s16s32s16s32;
extern const _dt_conf_t conf_u8s8s32s32;
extern const _dt_conf_t conf_u8s8s8s32;
extern const _dt_conf_t conf_u8s8u8s32;
extern const _dt_conf_t conf_s8s8s32s32;
extern const _dt_conf_t conf_s8s8s8s32;
extern const _dt_conf_t conf_s8s8u8s32;
extern const _dt_conf_t conf_u8s8f32s32_wino;
extern const _dt_conf_t conf_u8s8s32s32_wino;
extern const _dt_conf_t conf_u8s8s8s32_wino;
extern const _dt_conf_t conf_u8s8u8s32_wino;

const dt_conf_t *str2cfg(const char *str);
const char *cfg2str(const dt_conf_t *cfg);

struct prb_t: public desc_t {
    prb_t(const desc_t &desc, dir_t dir, const dt_conf_t *cfg, alg_t alg,
            merge_t merge, const attr_t &attr, int mb = 0)
        : desc_t(desc), dir(dir), cfg(cfg), alg(alg), merge(merge), attr(attr)
        , ops(0), scales(NULL) {
        if (mb) this->mb = mb;
        count_ops();
        generate_oscales();
    }
    ~prb_t() { if (scales) zfree(scales); }

    dir_t dir;
    const dt_conf_t *cfg;
    alg_t alg;
    merge_t merge;
    attr_t attr;

    double ops;
    float *scales;

    void count_ops();
    void generate_oscales();

private:
    prb_t(const prb_t &) = delete;
    prb_t &operator=(const prb_t &) = delete;
};
const size_t max_prb_len = max_attr_len + max_desc_len + 196;
void prb2str(const prb_t *p, char *buffer, bool canonical = false);

/* some extra control parameters which shouldn't be placed in prb_t */
extern const char *skip_impl; /* NULL or "" means do not skip anything */
extern bool allow_unimpl; /* true means do not treat unimplemented as error */
extern const char *perf_template; /* performance output template */

inline size_t src_off_f(const prb_t *p, int mb, int g, int ic,
                        int id, int ih, int iw)
{
    return ((((size_t)mb * p->ic + g * p->ic/p->g + ic)
        * p->id + id) * p->ih + ih) * p->iw + iw;
}

inline void inv_src_off_f(const prb_t *p, size_t off, int &mb, int &g, int &ic,
        int &id, int &ih, int &iw) {
    iw = off % p->iw; off /= p->iw;
    ih = off % p->ih; off /= p->ih;
    id = off % p->id; off /= p->id;
    ic = off % (p->ic / p->g); off /= (p->ic / p->g);
    g = off % p->g; off /= p->g;
    mb = off % p->mb; off /= p->mb;
    assert(off == 0);
}

inline size_t wei_off_f(const prb_t *p, int g, int oc, int ic,
                        int kd, int kh, int kw)
{
    return (((((size_t)g * p->oc / p->g + oc) * p->ic / p->g + ic)
        * p->kd + kd) * p->kh + kh) * p->kw + kw;
}

inline void inv_wei_off_f(const prb_t *p, size_t off, int &g, int &oc, int &ic,
        int &kd, int &kh, int &kw) {
    kw = off % p->kw; off /= p->kw;
    kh = off % p->kh; off /= p->kh;
    kd = off % p->kd; off /= p->kd;
    ic = off % (p->ic / p->g); off /= (p->ic / p->g);
    oc = off % (p->oc / p->g); off /= (p->oc / p->g);
    g = off % p->g; off /= p->g;
    assert(off == 0);
}

inline size_t bia_off_f(const prb_t *p, int g, int oc) {
    return (size_t)g * p->oc / p->g + oc;
}

inline void inv_bia_off_f(const prb_t *p, size_t off, int &g, int &oc) {
    oc = off % (p->oc / p->g); off /= (p->oc / p->g);
    g = off % p->g; off /= p->g;
    assert(off == 0);
}

inline size_t dst_off_f(const prb_t *p, int mb, int g, int oc,
                        int od, int oh, int ow)
{
    return ((((size_t)mb * p->oc + g * p->oc/p->g + oc) * p->od + od)
        * p->oh + oh) * p->ow + ow;
}

inline void inv_dst_off_f(const prb_t *p, size_t off, int &mb, int &g, int &oc,
        int &od, int &oh, int &ow) {
    ow = off % p->ow; off /= p->ow;
    oh = off % p->oh; off /= p->oh;
    od = off % p->od; off /= p->od;
    oc = off % (p->oc / p->g); off /= (p->oc / p->g);
    g = off % p->g; off /= p->g;
    mb = off % p->mb; off /= p->mb;
    assert(off == 0);
}

float oscale(const prb_t *p, int oc);

void compute_ref_fwd(const prb_t *p, dnn_mem_t &src_m, dnn_mem_t &wei_m,
        dnn_mem_t &bia_m, dnn_mem_t &dst_m);
void compute_ref_bwd_d(const prb_t *p, dnn_mem_t &diff_src_m, dnn_mem_t &wei_m,
        dnn_mem_t &bia_m, dnn_mem_t &diff_dst_m);
void compute_ref_bwd_w(const prb_t *p, dnn_mem_t &src_m, dnn_mem_t &diff_wei_m,
        dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m);

void compute_ref_direct_fwd(const prb_t *p, dnn_mem_t &src_m, dnn_mem_t &wei_m,
        dnn_mem_t &bia_m, dnn_mem_t &dst_m);
void compute_ref_direct_bwd_d(const prb_t *p, dnn_mem_t &diff_src_m, dnn_mem_t &wei_m,
        dnn_mem_t &bia_m, dnn_mem_t &diff_dst_m);
void compute_ref_direct_bwd_w(const prb_t *p, dnn_mem_t &src_m, dnn_mem_t &diff_wei_m,
        dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m);

void compute_wino_ref_fwd(const prb_t *p, dnn_mem_t &src_m, dnn_mem_t &wei_m,
        dnn_mem_t &bia_m, dnn_mem_t &dst_m);
void compute_wino_ref_bwd_d(const prb_t *p, dnn_mem_t &idiff_src_m,
        dnn_mem_t &wei_m, dnn_mem_t &bia_m, dnn_mem_t &diff_dst_m);
void compute_wino_ref_bwd_w(const prb_t *p, dnn_mem_t &src_m,
        dnn_mem_t &diff_wei_m, dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m);

void perf_report(const prb_t *p, const res_t *r, const char *pstr);

int compare_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false);
int compare_wei(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false);
int compare_bia(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false);
int compare_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false);
int fill_src(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r);
int fill_wei(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
    res_t *r);
int fill_bia(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r);
int fill_dst(const prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r);
double get_trust_nz_level(const prb_t *p, data_kind_t kind, bool final_compare);

void compute_ref_bwd_bias(const prb_t *p, dnn_mem_t &diff_bia_m, dnn_mem_t &diff_dst_m);
void compute_bias_fwd(const prb_t *p, dnn_mem_t &bia_m, dnn_mem_t &dst_m);
void compute_ref_bwd_weights(const prb_t *p, dnn_mem_t &src_m,dnn_mem_t &diff_wei_m, dnn_mem_t &diff_dst_m);

}

#endif
