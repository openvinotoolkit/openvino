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

#include "rnn/rnn_aux.hpp"
#include "norm.hpp"

#define DPRINT(...)                                     \
    do {                                                \
        int l = snprintf(buffer, rem_len, __VA_ARGS__); \
        buffer += l;                                    \
        rem_len -= l;                                   \
    } while (0)

namespace rnn {

alg_t str2alg(const char *str) {
#define CASE(_alg)                         \
    if (!strcasecmp(STRINGIFY(_alg), str)) \
    return _alg
    CASE(VANILLA_RNN);
    CASE(VANILLA_LSTM);
    CASE(VANILLA_GRU);
    CASE(GRU_LINEAR_BEFORE_RESET);
#undef CASE
    assert(!"unknown algorithm");
    return VANILLA_RNN;
}

const char *alg2str(alg_t alg) {
    if (alg == VANILLA_RNN)
        return "VANILLA_RNN";
    if (alg == VANILLA_LSTM)
        return "VANILLA_LSTM";
    if (alg == VANILLA_GRU)
        return "VANILLA_GRU";
    if (alg == GRU_LINEAR_BEFORE_RESET)
        return "GRU_LINEAR_BEFORE_RESET";
    assert(!"unknown algorithm");
    return "unknown algorithm";
}

mkldnn_alg_kind_t alg2kind(alg_t alg) {
    if (alg == VANILLA_RNN)
        return mkldnn_vanilla_rnn;
    if (alg == VANILLA_LSTM)
        return mkldnn_vanilla_lstm;
    if (alg == VANILLA_GRU)
        return mkldnn_vanilla_gru;
    if (alg == GRU_LINEAR_BEFORE_RESET)
        return mkldnn_gru_linear_before_reset;
    assert(!"unknown algorithm");
    return mkldnn_alg_kind_undef;
}

activation_t str2activation(const char *str) {
#define CASE(_act)                         \
    if (!strcasecmp(STRINGIFY(_act), str)) \
    return _act
    CASE(RELU);
    CASE(LOGISTIC);
    CASE(TANH);
#undef CASE
    assert(!"unknown activation");
    return TANH;
}

const char *activation2str(activation_t act) {
    const char *str = "unknown activation";
    switch (act) {
    case RELU: str = "RELU"; break;
    case LOGISTIC: str = "LOGISTIC"; break;
    case TANH: str = "TANH"; break;
    default: assert(!"unknown activation");
    }
    return str;
}

mkldnn_alg_kind_t activation2kind(activation_t act) {
    mkldnn_alg_kind_t alg_kind = mkldnn_alg_kind_undef;
    switch (act) {
    case RELU: alg_kind = mkldnn_eltwise_relu; break;
    case LOGISTIC: alg_kind = mkldnn_eltwise_logistic; break;
    case TANH: alg_kind = mkldnn_eltwise_tanh; break;
    default: assert(!"unknown activation");
    }
    return alg_kind;
}

const char *direction2str(mkldnn_rnn_direction_t direction) {

    // typedef enum {
    //     mkldnn_unidirectional_left2right,
    //     mkldnn_unidirectional_right2left,
    //     mkldnn_unidirectional = mkldnn_unidirectional_left2right,
    //     mkldnn_bidirectional_concat,
    //     mkldnn_bidirectional_sum,
    // } mkldnn_rnn_direction_t;

    if (direction == mkldnn_unidirectional_left2right)
        return "left2right";
    if (direction == mkldnn_unidirectional_right2left)
        return "right2left";
    if (direction == mkldnn_bidirectional_concat)
        return "concat";
    if (direction == mkldnn_bidirectional_sum)
        return "sum";
    assert(!"unknown direction");
    return "unknown direction";
}

void prb2str(const rnn_prb_t *p, const res_t *res, char *buffer) {
    int rem_len = max_prb_len;

    DPRINT("%s(%s,%s)", alg2str(p->alg), activation2str(p->activation),
            direction2str(p->direction));
    DPRINT("l%d", p->n_layer);
    DPRINT("t%d", p->n_iter);
    DPRINT("m%d", p->mb);
    DPRINT("sic%d", p->sic);
    DPRINT("slc%d", p->slc);
    DPRINT("dic%d", p->dic);
    DPRINT("dlc%d", p->dlc);
    DPRINT("n\"%s\"", p->name);
}

void init_buffer(float *buf, int size, float value) {
    for (int i = 0; i < size; i++)
        buf[i] = value;
}

float logistic(float x) {
    return 1.0f / (1.0f + expf(-x));
}
float dlogistic(float x) {
    float tmp = logistic(x);
    return tmp * (1 - tmp);
}
float relu(float x) {
    return x > 0 ? x : 0;
}
float drelu(float x) {
    return float(x > 0);
}
float dtanhf(float x) {
    return (1 - (tanhf(x) * tanhf(x)));
}

int compare_dat(const rnn_prb_t *p, rnn_data_kind_t kind, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare = false) {
    size_t nelems = mem_dt.nelems();

    const char *skind = rnn_data_kind2str(kind);

    int in = 0, below = 0, above = 0;
    int in_ok = 0, below_ok = 0, above_ok = 0;
    int non_zero = 0;

    diff_norm_t diff_norm;

    r->errors = 0;
    r->total = nelems;

    for (size_t i = 0; i < nelems; ++i) {
        const float dt = ((float *)mem_dt)[i];
        const float fp0 = ((float *)mem_fp)[i];

        float fp = fp0;

        const float diff = fabsf(fp - dt);
        const float rel_diff = diff / (fabsf(fp) > FLT_MIN ? fabsf(fp) : 1);

        bool ok = true;
        if (fp < p->cfg_[kind].min) {
            diff_norm.update(p->cfg_[kind].min, dt);
            ok = dt == p->cfg_[kind].min;
            below += 1;
            below_ok += ok;
        } else if (fp > p->cfg_[kind].max) {
            diff_norm.update(p->cfg_[kind].max, dt);
            ok = dt == p->cfg_[kind].max;
            above += 1;
            above_ok += ok;
        } else {
            diff_norm.update(fp, dt);
            ok = (fabs(fp) > 1e-5 ? rel_diff : diff) <= p->cfg_[kind].eps;
            in += 1;
            in_ok += ok;
        }
        if (!ok) {
            r->errors++;
            if (r->errors < 10 || verbose >= 10) {
                int n = 0, t = 0, c = 0, s = 0, l = 0, d = 0, w = 0, ic = 0,
                    oc = 0, b = 0;
                switch (kind) {
                case input:
                    inv_ntc_off_f(p, i, n, t, c);
                    print(0, "%lu, %s, [%s][%d,%d,%d] "
                             "fp:%8g fp0:%8g dt:%8g diff:%8g rdiff:%8g\n",
                            (unsigned long)i,
                            final_compare == false ? "REORDER " : "", skind, n,
                            t, c, fp, fp0, dt, diff, rel_diff);
                    break;
                case states:
                    inv_ldsnc_off_f(p, i, l, d, s, n, c);
                    print(0, "%lu, %s, [%s][%d,%d,%d,%d,%d] "
                             "fp:%8g fp0:%8g dt:%8g diff:%8g rdiff:%8g\n",
                            (unsigned long)i,
                            final_compare == false ? "REORDER " : "", skind, l,
                            d, s, n, c, fp, fp0, dt, diff, rel_diff);
                    break;
                case weights_input:
                    inv_ldigo_off_f(p, i, l, d, w, ic, oc);
                    print(0, "%lu, %s, [%s][%d,%d,%d,%d,%d] "
                             "fp:%8g fp0:%8g dt:%8g diff:%8g rdiff:%8g\n",
                            (unsigned long)i,
                            final_compare == false ? "REORDER " : "", skind, l,
                            d, w, ic, oc, fp, fp0, dt, diff, rel_diff);
                    break;
                case weights_states:
                    inv_ldigo_off_f(p, i, l, d, w, ic, oc);
                    print(0, "%lu, %s, [%s][%d,%d,%d,%d,%d] "
                             "fp:%8g fp0:%8g dt:%8g diff:%8g rdiff:%8g\n",
                            (unsigned long)i,
                            final_compare == false ? "REORDER " : "", skind, l,
                            d, w, ic, oc, fp, fp0, dt, diff, rel_diff);
                    break;
                case bias:
                    inv_ldgo_off_f(p, i, l, d, b, c);
                    print(0, "%lu, %s, [%s][%d,%d,%d,%d] "
                             "fp:%8g fp0:%8g dt:%8g diff:%8g rdiff:%8g\n",
                            (unsigned long)i,
                            final_compare == false ? "REORDER " : "", skind, l,
                            d, b, c, fp, fp0, dt, diff, rel_diff);
                    break;
                case dst_last_layer:
                    inv_tnc_off_f(p, i, s, t, n, c);
                    print(0, "%lu, %s, [%s][%d,%d,%d,%d] "
                             "fp:%8g fp0:%8g dt:%8g diff:%8g rdiff:%8g\n",
                            (unsigned long)i,
                            final_compare == false ? "REORDER " : "", skind, s,
                            t, n, c, fp, fp0, dt, diff, rel_diff);
                    break;
                case dst_last_iteration:
                    inv_ldsnc_off_f(p, i, l, d, s, n, c);
                    print(0, "%lu, %s, [%s][%d,%d,%d,%d,%d "
                             "fp:%8g fp0:%8g dt:%8g diff:%8g rdiff:%8g\n",
                            (unsigned long)i,
                            final_compare == false ? "REORDER " : "", skind, l,
                            d, s, n, c, fp, fp0, dt, diff, rel_diff);
                    break;
                default: assert("unknown data kind"); return FAIL;
                }
            }
        }

#if 1
        /* for debug purposes only: dump the output */
        if (final_compare && verbose >= 50) {
            int n = 0, t = 0, c = 0, s = 0, l = 0, d = 0, w = 0, ic = 0, oc = 0,
                b = 0;

            switch (kind) {
            case input:
                inv_ntc_off_f(p, i, n, t, c);
                print(0, "[%4lu][%s][%d,%d,%d] fp:%8g fp0:%8g dt:%8g\n",
                        (unsigned long)i, skind, n, t, c, fp, fp0, dt);
                break;
            case states:
                inv_ldsnc_off_f(p, i, l, d, s, n, c);
                print(0, "[%4lu][%s][%d,%d,%d,%d,%d] fp:%8g fp0:%8g dt:%8g\n",
                        (unsigned long)i, skind, l, d, s, n, c, fp, fp0, dt);
                break;
            case weights_input:
                inv_ldigo_off_f(p, i, l, d, w, ic, oc);
                print(0, "[%4lu][%s][%d,%d,%d,%d,%d] fp:%8g fp0:%8g dt:%8g\n",
                        (unsigned long)i, skind, l, d, w, ic, oc, fp, fp0, dt);
                break;
            case weights_states:
                inv_ldigo_off_f(p, i, l, d, w, ic, oc);
                break;
                print(0, "[%4lu][%s][%d,%d,%d,%d,%d] fp:%8g fp0:%8g dt:%8g\n",
                        (unsigned long)i, skind, l, d, w, ic, oc, fp, fp0, dt);
            case bias:
                inv_ldgo_off_f(p, i, l, d, b, c);
                break;
                print(0, "[%4lu][%s][%d,%d,%d,%d] fp:%8g fp0:%8g dt:%8g\n",
                        (unsigned long)i, skind, l, d, b, c, fp, fp0, dt);
            case dst_last_layer:
                inv_tnc_off_f(p, i, s, t, n, c);
                print(0, "[%4lu][%s][%d,%d,%d] fp:%8g fp0:%8g dt:%8g\n",
                        (unsigned long)i, skind, n, t, c, fp, fp0, dt);
                break;
            case dst_last_iteration:
                inv_ldsnc_off_f(p, i, l, d, s, n, c);
                print(0, "[%4lu][%s][%d,%d,%d,%d,%d] fp:%8g fp0:%8g dt:%8g\n",
                        (unsigned long)i, skind, l, d, s, n, c, fp, fp0, dt);
                break;
            default:
                print(0, "[%4lu][unknown] fp:%8g fp0:%8g dt:%8g\n",
                        (unsigned long)i, fp, fp0, dt);
                break;
            }
        }
#endif

        non_zero += fp != 0;
    }

    diff_norm.done();

    if (final_compare || r->errors) {
        const int vl = r->errors ? 0 : 2;
        print(vl,
                "@@@ [%s] %sdiff: l0(``%g``) "
                "l1:(%g,%g,%g,``%g``) "
                "l2:(%g,%g,%g,``%g``) "
                "l8:(%g,%g,%g,``%g``)\n",
                skind, final_compare ? "final: " : "",
                diff_norm.rel_diff(norm_t::L0), diff_norm.a_[norm_t::L1],
                diff_norm.b_[norm_t::L1], diff_norm.diff_[norm_t::L1],
                diff_norm.rel_diff(norm_t::L1), diff_norm.a_[norm_t::L2],
                diff_norm.b_[norm_t::L2], diff_norm.diff_[norm_t::L2],
                diff_norm.rel_diff(norm_t::L2), diff_norm.a_[norm_t::L8],
                diff_norm.b_[norm_t::L8], diff_norm.diff_[norm_t::L8],
                diff_norm.rel_diff(norm_t::L8));
    }

    // const double trust_rg_level = 0.3;
    //??        const double trust_nz_level = get_trust_nz_level(p, kind,
    // final_compare);

    // const double trust_rg = (double)in / r->total;
    // const double trust_nz = (double)non_zero / r->total;

    // const bool no_trust = true /* ...in the test ...at all */
    // && final_compare
    //??            && (trust_rg < trust_rg_level || trust_nz <
    // trust_nz_level)
    //;

    // const bool dump = verbose >= 20
    // || (verbose >= 10 && (trust_rg < 1. || trust_nz < 1.));
    /*??
    if (dump) {
        print(0, "@@@ [%s] %strust range:%.2f nz:%.2f "
            "(level range:%.2f nz:%.2f). "
            "in:%d (ok:%d) below:%d (ok:%d) above:%d (ok:%d) nz:%d "
            "total:%lu\n", skind, final_compare ? "final: " : "",
            trust_rg, trust_nz, trust_rg_level, trust_nz_level, in, in_ok,
            below, below_ok, above, above_ok, non_zero,
            (unsigned long)r->total);
    }
    */

    /*??
    if (no_trust) {
        r->state = MISTRUSTED;
        print(0, "@@@ [%s] test-bug: trust is too low. "
            "range:%.2f (?<%.2f) nz:%.2f (?<%.2f) (nz: %d total: %lu)\n",
            skind, trust_rg, trust_rg_level, trust_nz, trust_nz_level,
            non_zero, (unsigned long)r->total);
    }*/

    if (r->errors)
        r->state = FAILED;

    if (final_compare && r->state == UNTESTED)
        r->state = PASSED; /* optimism */

    return r->state == FAILED ? FAIL : OK;
}

int compare_input(const rnn_prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false) {
    return compare_dat(p, input, mem_dt, mem_fp, r, final_compare);
}
int compare_states(const rnn_prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false) {
    return compare_dat(p, states, mem_dt, mem_fp, r, final_compare);
}
int compare_weights_input(const rnn_prb_t *p, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare = false) {
    return compare_dat(p, weights_input, mem_dt, mem_fp, r, final_compare);
}
int compare_weights_states(const rnn_prb_t *p, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare = false) {
    return compare_dat(p, weights_states, mem_dt, mem_fp, r, final_compare);
}
int compare_bias(const rnn_prb_t *p, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        res_t *r, bool final_compare = false) {
    return compare_dat(p, bias, mem_dt, mem_fp, r, final_compare);
}
int compare_dst_last_layer(const rnn_prb_t *p, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare = false) {
    return compare_dat(p, dst_last_layer, mem_dt, mem_fp, r, final_compare);
}
int compare_dst_last_iteration(const rnn_prb_t *p, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *r, bool final_compare = false) {
    return compare_dat(p, dst_last_iteration, mem_dt, mem_fp, r, final_compare);
}

} // namespace rnn
