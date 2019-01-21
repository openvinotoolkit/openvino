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

#ifndef _LSTM_HPP
#define _LSTM_HPP

#include <assert.h>
#include <limits.h>
#include <stdint.h>

#include "common.hpp"
#include "dnn_types.hpp"
#include "mkldnn_common.hpp"
#include "mkldnn_debug.hpp"
#include "mkldnn_memory.hpp"

namespace rnn {

enum alg_t { VANILLA_RNN, VANILLA_LSTM, VANILLA_GRU, LBR_GRU };
alg_t str2alg(const char *str);
const char *alg2str(alg_t alg);
mkldnn_alg_kind_t alg2kind(alg_t alg);

enum activation_t { RELU, LOGISTIC, TANH };
activation_t str2activation(const char *str);
const char *activation2str(activation_t alg);
mkldnn_alg_kind_t activation2kind(activation_t alg);

mkldnn_rnn_direction_t str2direction(const char *str);
const char *direction2str(mkldnn_rnn_direction_t direction);

const int H = 0;
const int C = 1;

template <typename Telem>
struct array_offset_calculator {
    template <typename... Targs>
    array_offset_calculator(Telem *base, Targs... Fargs)
        : _size(sizeof...(Fargs)) {
        const int init_list[] = { Fargs... };
        _dims = new int[_size];
        for (int i = 0; i < _size; ++i)
            _dims[i] = init_list[i];

        _base_ptr = base;
    }
    ~array_offset_calculator() { delete[] _dims; }
    template <typename... Targs>
    inline Telem &operator()(Targs... Fargs) {
        return *(_base_ptr + _offset(1, Fargs...));
    }

private:
    template <typename... Targs>
    inline int _offset(int const dimension, int element) {
        return element;
    }

    template <typename... Targs>
    inline int _offset(int const dimension, int theta, int element) {
        return element + (_dims[dimension] * theta);
    }

    template <typename... Targs>
    inline int _offset(
            int const dimension, int theta, int element, Targs... Fargs) {
        int t_prime = element + (_dims[dimension] * theta);
        return _offset(dimension + 1, t_prime, Fargs...);
    }

    Telem *_base_ptr;
    int _size;
    int *_dims;
};

struct rnn_desc_t {
    int sic;
    int slc;
    int dic;
    int dlc;
    int mb;
    int n_layer;
    int n_iter;
    const char *name;
};
int str2desc(rnn_desc_t *desc, const char *str);

enum rnn_data_kind_t {
    input,
    states,
    weights_input,
    weights_states,
    bias,
    dst_last_layer,
    dst_last_iteration,
    dst_diff_input,
    dst_diff_states,
    dst_diff_weights_input,
    dst_diff_weights_states,
    dst_diff_bias,
    diff_last_layer,
    diff_last_iteration,
    data_kind_total // should be last to provide the total number of data kinds
};

inline const char *rnn_data_kind2str(rnn_data_kind_t kind) {
    switch (kind) {
    case input: return "INPUT";
    case states: return "STATES";
    case weights_input: return "WEIGHTS_INPUT";
    case weights_states: return "WEIGHTS_STATES";
    case bias: return "BIAS";
    case dst_last_layer: return "DST_LAST_LAYER";
    case dst_last_iteration: return "DST_LAST_ITERATION";
    default:
        assert(!"incorrect rnn data kind");
        return "incorrect rnn data kind";
    }
}

/** configuration structure, that controls initial data filling + error check
*
* dt defines precision
*
* for each lst data kind the values are filled as follows:
* if (rand() > f_sparsity) then:
*     v <-- f_base
* else:
*     v <-- f_min + rand() * f_step % (f_max - f_min)
*
* on final check the resulting values should be in [min .. max] range, the
* relative difference should not exceed eps
*/

typedef struct dt_conf_t {
    mkldnn_data_type_t dt;
    int min, max; /* representative */
    int f_min, f_max; /* fill range */
    int f_base; /* fill base, use 0 */
    int f_step; /* fill step, use 1 */
    double f_sparsity; /* amount of non-zeros, default 0.25 */
    double eps; /* acceptable error */
} _dt_conf_t[data_kind_total];

extern const _dt_conf_t conf_f32;

struct rnn_prb_t : public rnn_desc_t {
    rnn_prb_t(const rnn_desc_t desc, const dt_conf_t *cfg,
            mkldnn_prop_kind_t prop, alg_t alg,
            mkldnn_rnn_direction_t direction, activation_t activation)
        : rnn_desc_t(desc), cfg(cfg), prop(prop), alg(alg),
        direction(direction), activation(activation){
    }

    int n_directions() const {
        return (direction == mkldnn_bidirectional_concat
                       || direction == mkldnn_bidirectional_sum) ?
                2 :
                1;
    }
    int n_weights() const { return 1; }
    int n_states() const { return alg == VANILLA_LSTM ? 2 : 1; }
    int n_gates() const {
        return alg == VANILLA_LSTM ?
                4 :
                (alg == VANILLA_GRU || alg == LBR_GRU ? 3 : 1);
    }

    const dt_conf_t *cfg;
    mkldnn_prop_kind_t prop;
    alg_t alg;
    mkldnn_rnn_direction_t direction;
    activation_t activation;

private:
    rnn_prb_t(const rnn_prb_t &) = delete;
    rnn_prb_t &operator=(const rnn_prb_t &) = delete;
};

const size_t max_prb_len = 392;
void prb2str(const rnn_prb_t *p, const res_t *res, char *buffer);

void compute_ref_fwd(const rnn_prb_t *p, dnn_mem_t &input_m,
        dnn_mem_t &states_m, dnn_mem_t &weights_input_m,
        dnn_mem_t &weights_states_m, dnn_mem_t &bias_m,
        dnn_mem_t &dst_last_layer_m, dnn_mem_t &dst_last_iteration_m,
        mkldnn_rnn_direction_t direction);

void compute_ref_bwd(const rnn_prb_t *p, dnn_mem_t &input_m,
        dnn_mem_t &states_m, dnn_mem_t &diff_last_layer_m,
        dnn_mem_t &diff_last_iteration_m, dnn_mem_t &weights_input_m,
        dnn_mem_t &weights_states_m, dnn_mem_t &bias_m,
        dnn_mem_t &dst_last_layer_m, dnn_mem_t &dst_last_iteration_m,
        dnn_mem_t &dst_diff_input_m, dnn_mem_t &dst_diff_states_m,
        dnn_mem_t &dst_diff_weights_input_m,
        dnn_mem_t &dst_diff_weights_states_m, dnn_mem_t &dst_diff_bias_m,
        mkldnn_rnn_direction_t direction);

// mkldnn_ntc
inline size_t ntc_off_f(const rnn_prb_t *p, int n, int t, int c) {
    return ((size_t)n * p->n_iter + t) * p->slc + c;
}

inline void inv_ntc_off_f(
        const rnn_prb_t *p, size_t off, int &n, int &t, int &c) {
    c = off % p->slc;
    off /= p->slc;
    t = off % p->n_iter;
    off /= p->n_iter;
    n = off % p->mb;
    off /= p->mb;
    assert(off == 0);
}

// mkldnn_ldsnc
inline size_t ldsnc_off_f(
        const rnn_prb_t *p, int l, int d, int s, int n, int c) {
    return ((((size_t)l * p->n_directions() + d) * p->n_states() + s) * p->mb
                   + n)
            * p->sic
            + c;
}

inline void inv_ldsnc_off_f(const rnn_prb_t *p, size_t off, int &l, int &d,
        int &s, int &n, int &c) {
    c = off % p->sic;
    off /= p->sic;
    n = off % p->mb;
    off /= p->mb;
    s = off % p->n_states();
    off /= p->n_states();
    d = off % p->n_directions();
    off /= p->n_directions();
    l = off % p->n_layer;
    off /= p->n_layer;
    assert(off == 0);
}

// mkldnn_ldigo
inline size_t ldigo_off_f(
        const rnn_prb_t *p, int l, int d, int w, int ic, int oc) {
    return ((((size_t)l * p->n_directions() + d) * p->n_weights() + w)
                           * (4 * p->slc)
                   + ic)
            * p->sic
            + oc;
}

inline void inv_ldigo_off_f(const rnn_prb_t *p, size_t off, int &l, int &d,
        int &w, int &ic, int &oc) {
    oc = off % p->sic;
    off /= p->sic;
    ic = off % (4 * p->slc);
    off /= (4 * p->slc);
    w = off % p->n_weights();
    off /= p->n_weights();
    d = off % p->n_directions();
    off /= p->n_directions();
    l = off % p->n_layer;
    off /= p->n_layer;
    assert(off == 0);
}

// mkldnn_ldwOcIc
inline size_t ldwOcIc_off_f(
        const rnn_prb_t *p, int l, int d, int w, int oc, int ic) {
    return ((((size_t)l * p->n_directions() + d) * p->n_weights() + w)
                           * (4 * p->sic)
                   + oc)
            * p->slc
            + ic;
}

inline void inv_ldwOcIc_off_f(const rnn_prb_t *p, size_t off, int &l, int &d,
        int &w, int &oc, int &ic) {
    ic = off % p->slc;
    off /= p->slc;
    oc = off % (4 * p->sic);
    off /= (4 * p->sic);
    w = off % p->n_weights();
    off /= p->n_weights();
    d = off % p->n_directions();
    off /= p->n_directions();
    l = off % p->n_layer;
    off /= p->n_layer;
    assert(off == 0);
}

// bias: mkldnn_ldgo
inline size_t ldgo_off_f(const rnn_prb_t *p, int l, int d, int b, int c) {
    return (((size_t)l * p->n_directions() + d) * p->n_gates() + b) * p->sic
            + c;
}

inline void inv_ldgo_off_f(
        const rnn_prb_t *p, size_t off, int &l, int &d, int &b, int &c) {
    c = off % p->sic;
    off /= p->sic;
    b = off % p->n_gates();
    off /= p->n_gates();
    d = off % p->n_directions();
    off /= p->n_directions();
    l = off % p->n_layer;
    off /= p->n_layer;
    assert(off == 0);
}

// dst_last_layer: mkldnn_tnc
inline size_t tnc_off_f(const rnn_prb_t *p, int s, int t, int n, int c) {
    return (((size_t)s * p->n_iter + t) * p->mb + n) * p->sic + c;
}

inline void inv_tnc_off_f(
        const rnn_prb_t *p, size_t off, int &s, int &t, int &n, int &c) {
    c = off % p->sic;
    off /= p->sic;
    n = off % p->mb;
    off /= p->mb;
    t = off % p->n_iter;
    off /= p->n_iter;
    s = off % p->n_states();
    off /= p->n_states();
    assert(off == 0);
}

void perf_report(const rnn_prb_t *p, const res_t *r, const char *pstr);

int doit(const rnn_prb_t *p, res_t *res);
void check(rnn_desc_t *p);
int bench(int argc, char **argv, bool main_bench = true);
} // namespace rnn

#endif
