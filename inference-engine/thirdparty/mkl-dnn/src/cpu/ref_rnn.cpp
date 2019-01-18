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

/*
  General architecture

  for diff states, we have n_states + 1 as we have n_states diff
  to propagate to the previous iteration and 1 states to propagate
  to the previous layer
  index 0 is dh for cell(t-1, l) to consume
  index 1 is dc for cell(t-1, l) to consume
  index 2 is dh for cell(t, l-1) to consume
  this indexing enables to have the same indexing for states in elemwise
  function
  only the cell execution function should be impacted

 */

#include "c_types_map.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "mkldnn_traits.hpp"
#include "type_helpers.hpp"
#include "gemm/gemm.hpp"

#include "ref_rnn.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::math;
using namespace prop_kind;
using namespace alg_kind;

#define AOC array_offset_calculator

inline float one_m_square(float x) {
    return (1.0f - x) * (1.0f + x);
}
inline float x_m_square(float x) {
    return (1.0f - x) * x;
}

template <>
float activation<alg_kind::eltwise_relu, prop_kind::forward>(
        float dd, float s, float alpha, float cliping) {
    return relu_fwd<float>(s, alpha);
}

template <>
float activation<alg_kind::eltwise_relu, prop_kind::backward>(
        float dd, float s, float alpha, float cliping) {
    return relu_bwd<float>(dd, s, alpha);
}

template <>
float activation<alg_kind::eltwise_tanh, prop_kind::forward>(
        float dd, float s, float alpha, float cliping) {
    return tanh_fwd<float>(s);
}

template <>
float activation<alg_kind::eltwise_tanh, prop_kind::backward>(
        float dd, float s, float alpha, float cliping) {
    return dd * one_m_square(s);
}

template <>
float activation<alg_kind::eltwise_logistic, prop_kind::forward>(
        float dd, float s, float alpha, float cliping) {
    return logistic_fwd<float>(s);
}

template <>
float activation<alg_kind::eltwise_logistic, prop_kind::backward>(
        float dd, float s, float alpha, float cliping) {
    return dd * x_m_square(s);
}

//************************* Cell execution *************************//
/// @todo shall this be templated on activation function to enable svml calls
/// particularly?
template <>
elemwise_sig(_ref_rnn_common_t<prop_kind::forward>::rnn_elemwise) {
    AOC<float, 3> ws_gates(ws_gates_, batch, conf_.GC());
    AOC<const float, 2> bias(bias_, n_gates, dic);
    AOC<float, 4> states_t_l(states_t_l_, n_states, iter_stride, batch, wic);
    parallel_nd(batch, [&](int i) {
        for (int j = 0; j < dic; j++) {
            const float h =
                activation_func(0, ws_gates(i, j) + bias(0, j), 0, 0);
            ws_gates(i, j) = states_t_l(0, 0, i, j) = h;
        }
    });
}

template <>
elemwise_sig(_ref_rnn_common_t<prop_kind::backward>::rnn_elemwise) {
    AOC<float, 3> ws_gates(ws_gates_, batch, conf_.GC());
    AOC<float, 4> diff_states_tp1_l(
            diff_states_tp1_l_, n_states + 1, iter_stride, batch, wic);
    AOC<float, 4> diff_states_t_lp1(
            diff_states_t_lp1_, n_states + 1, iter_stride, batch, wic);
    parallel_nd(batch, [&](int i) {
        for (int j = 0; j < dic; ++j) {
            const float dH = diff_states_t_lp1(n_states, 0, i, j)
                + diff_states_tp1_l(0, 0, i, j);
            auto g = ws_gates(i, j);
            ws_gates(i, j) = activation_func(dH, g, 0, 0);
        }
    });
}

template <>
elemwise_sig(_ref_rnn_common_t<prop_kind::forward>::lstm_elemwise) {
    AOC<float, 3> ws_gates(ws_gates_, batch, conf_.GC());
    AOC<const float, 2> bias(bias_, n_gates, dic);
    AOC<float, 4> states_t_l(states_t_l_, n_states, iter_stride, batch, wic);
    AOC<float, 4> states_tm1_l(states_tm1_l_, n_states, iter_stride, batch, wic);

    parallel_nd(batch, [&](int i) {
// WA. Loss of correctnes in case of simd loop unrolling with icc 18
#if !defined(__INTEL_COMPILER)
        PRAGMA_OMP_SIMD()
#endif
        for (int j = 0; j < dic; j++) {
            ws_gates(i, 0 * dic + j) = logistic_fwd(ws_gates(i, 0 * dic + j) + bias(0, j));
            ws_gates(i, 1 * dic + j) = logistic_fwd(ws_gates(i, 1 * dic + j) + bias(1, j));
            ws_gates(i, 2 * dic + j) = tanh_fwd(ws_gates(i, 2 * dic + j) + bias(2, j));
            ws_gates(i, 3 * dic + j) = logistic_fwd(ws_gates(i, 3 * dic + j) + bias(3, j));

            float tmp = ws_gates(i, 1 * dic + j) * states_tm1_l(1, 0, i, j)
                    + ws_gates(i, 0 * dic + j) * ws_gates(i, 2 * dic + j);
            states_t_l(0, 0, i, j) = ws_gates(i, 3 * dic + j) * tanh_fwd(tmp);
            states_t_l(1, 0, i, j) = tmp;
        }
    });
}

template <>
elemwise_sig(_ref_rnn_common_t<prop_kind::backward>::lstm_elemwise) {
    AOC<float, 3> ws_gates(ws_gates_, batch, conf_.GC());
    AOC<const float, 2> bias(bias_, n_gates, dic);
    AOC<float, 4> states_t_l(states_t_l_, n_states, iter_stride, batch, wic);
    AOC<float, 4> states_tm1_l(states_tm1_l_, n_states, iter_stride, batch, wic);
    AOC<float, 4> diff_states_t_l(diff_states_t_l_, n_states + 1, iter_stride, batch, wic);
    AOC<float, 4> diff_states_tp1_l(
        diff_states_tp1_l_, n_states + 1, iter_stride, batch, wic);
    AOC<float, 4> diff_states_t_lp1(
        diff_states_t_lp1_, n_states + 1, iter_stride, batch, wic);

    parallel_nd(batch, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < dic; j++) {
            float Ct = states_t_l(1, 0, i, j);
            /// @todo save it in the workspace in fwd pass or recompute it to
            /// save bw
            float tanhCt = tanh_fwd(Ct);
            // we have 2 incoming diffs on Ht
            float dHt = diff_states_tp1_l(0, 0, i, j)
            + diff_states_t_lp1(n_states, 0, i, j);
            float dCt = diff_states_tp1_l(1, 0, i, j)
                    + one_m_square(tanhCt) * ws_gates(i, 3 * dic + j) * dHt;

            float dG1 = states_tm1_l(1, 0, i, j) * dCt
                    * x_m_square(ws_gates(i, 1 * dic + j));
            float dG0 = ws_gates(i, 2 * dic + j) * dCt
                    * x_m_square(ws_gates(i, 0 * dic + j));
            float dG3 = tanhCt * dHt * x_m_square(ws_gates(i, 3 * dic + j));
            float dG2 = ws_gates(i, 0 * dic + j) * dCt
                    * one_m_square(ws_gates(i, 2 * dic + j));

            diff_states_t_l(1, 0, i, j) = dCt * ws_gates(i, 1 * dic + j);

            ws_gates(i, 0 * dic + j) = dG0;
            ws_gates(i, 1 * dic + j) = dG1;
            ws_gates(i, 2 * dic + j) = dG2;
            ws_gates(i, 3 * dic + j) = dG3;
        }
    });
}

template <prop_kind_t aprop>
gemm_sig(_ref_rnn_common_t<aprop>::packed_gemm) {
#if (USE_MKL_PACKED_GEMM)
    cblas_sgemm_compute(CblasColMajor, CblasPacked,
            is_B_trans ? CblasTrans : CblasNoTrans, m, n, k, a_, strideA_m, b_,
            is_B_trans ? strideB_n : strideB_k, beta, c_, strideC_m);
#else
    UNUSED(m);
    UNUSED(n);
    UNUSED(k);
    UNUSED(a_);
    UNUSED(b_);
    UNUSED(c_);
    UNUSED(is_B_trans);
    UNUSED(beta);
    assert(!"packed gemm is disabled");
#endif
}

template <prop_kind_t aprop>
gemm_sig(_ref_rnn_common_t<aprop>::gemm) {
    float alpha = 1.f;
    extended_sgemm("N", is_B_trans ? "T" : "N", &m, &n, &k, &alpha,
            a_, &strideA_m, b_, is_B_trans ? &strideB_n : &strideB_k, &beta,
            c_, &strideC_m, nullptr, use_jit_sgemm_);
}

template <prop_kind_t aprop>
void _ref_rnn_common_t<aprop>::gates_reduction(int n_gates, int dic, int wic, int batch,
        const float *ws_gates_, float *diff_bias_) {
    auto body = [&](int i, int k) {
        for (int j = 0; j < batch; j++)
            diff_bias_[i * dic + k]
                    += ws_gates_[j * conf_.GC() + i * dic + k];
    };

    // @todo block k on simd-width
#if MKLDNN_THR == MKLDNN_THR_OMP && _OPENMP >= 201307 \
    /* icc 17.0 has a problem with simd collapse */ \
    && !((defined __INTEL_COMPILER) && (__INTEL_COMPILER == 1700))
#pragma omp parallel for simd collapse(2)
    for (int i = 0; i < n_gates; i++)
        for (int k = 0; k < dic; k++)
            body(i, k);
#else
    parallel_nd(n_gates, dic, body);
#endif
}
/// @todo template this function on fwd or bwd, if the overhead
///  to pass argument for empty function is too big
template <>
cell_execution_sig(_ref_rnn_common_t<prop_kind::forward>::cell_execution) {
    if (!merge_gemm_layer) {
        (this->*gemm_input_func)(n_gates * dic, batch, slc, conf_.WL_GLD(), slc,
                batch, wic, conf_.GC(), batch, w_input_[0], states_t_lm1_,
                ws_gates_, false, 0.0f);
    }
    (this->*gemm_state_func)(n_gates * dic, batch, sic, conf_.WI_GLD(), sic,
            batch, wic, conf_.GC(), batch, w_state_[0], states_tm1_l_,
            ws_gates_, false, 1.0f);
    (this->*elemwise_func)(dic, wic, batch, n_states, iter_stride, n_gates, ws_gates_,
            states_t_l_, states_t_lm1_, states_tm1_l_, diff_states_t_l_,
            diff_states_t_lp1_, diff_states_tp1_l_, bias_, ws_grid_, ws_cell_);
}

template <>
cell_execution_sig(_ref_rnn_common_t<prop_kind::backward>::cell_execution) {
    (this->*elemwise_func)(dic, wic, batch, n_states, iter_stride, n_gates, ws_gates_,
            states_t_l_, states_t_lm1_, states_tm1_l_, diff_states_t_l_,
            diff_states_t_lp1_, diff_states_tp1_l_, bias_, ws_grid_, ws_cell_);

    /// bwd by data on the cell
    (this->*gemm_state_func)(sic, batch, n_gates * dic, conf_.WI_GLD(),
            n_gates * dic, batch, conf_.GC(), wic, batch, w_state_[0],
            ws_gates_, diff_states_t_l_, false, 0.0f);

    if (!merge_gemm_layer) {
        (this->*gemm_input_func)(slc, batch, n_gates * dic, conf_.WL_GLD(),
                n_gates * dic, batch, conf_.GC(), wic, batch, w_input_[0],
                ws_gates_,
                diff_states_t_l_ + n_states * iter_stride * (batch * wic),
                false, 0.0f);

        /// bwd by weights on the cell
        gemm(n_gates * dic, slc, batch, conf_.GC(), batch, wic, batch,
                conf_.DWL_GLD(), slc, ws_gates_, states_t_lm1_, diff_w_input_,
                true, 1.0f);
    }

    if (!merge_gemm_iter)
        gemm(n_gates * dic, sic, batch, conf_.GC(), batch, wic, batch,
                conf_.DWI_GLD(), sic, ws_gates_, states_tm1_l_, diff_w_state_,
                true, 1.0f);
    /// bwd by bias we just accumulate diffs from the gates
    gates_reduction(n_gates, dic, wic, batch, ws_gates_, diff_bias_);
}

template <>
cell_execution_sig(_ref_rnn_common_t<prop_kind::forward>::cell_execution_gru) {
    AOC<float, 3> ws_gates(ws_gates_, batch, conf_.GC());
    AOC<const float, 2> bias(bias_, n_gates, dic);
    AOC<float, 2> states_t_l(states_t_l_, batch, wic);
    AOC<float, 2> states_tm1_l(states_tm1_l_, batch, wic);

    // 1. gemm Wx[0-2],x
    if (!merge_gemm_layer) {
        (this->*gemm_input_func)(n_gates * dic, batch, slc, conf_.WL_GLD(), slc,
                batch, wic, conf_.GC(), batch, w_input_[0], states_t_lm1_,
                ws_gates_, false, 0.0f);
    }

    // 2. gemm Wh[0-1],h
    (this->*gemm_state_func)((n_gates - 1) * dic, batch, sic, conf_.WI_GLD(),
            sic, batch, wic, conf_.GC(), batch, w_state_[0], states_tm1_l_,
            ws_gates_, false, 1.0f);

    // 3. activation zt and rt + elemwise multiplication rt,ht-1
    parallel_nd(batch, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < dic; j++) {
            ws_gates(i, 0 * dic + j) = logistic_fwd(ws_gates(i, 0 * dic + j) + bias(0, j));
            ws_gates(i, 1 * dic + j) = logistic_fwd(ws_gates(i, 1 * dic + j) + bias(1, j));
            states_t_l(i, j) = states_tm1_l(i, j) * ws_gates(i, 1 * dic + j);
        }
    });

    // 4. gemm Wh[2],h~t
    (this->*gemm_state_func)(dic, batch, sic, conf_.WI_GLD(), sic, batch, wic,
            conf_.GC(), batch, w_state_[1], states_t_l_,
            &(ws_gates(0, 2 * dic)), false, 1.0f);

    // 5. activation h~t + calculate ht
    parallel_nd(batch, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < dic; j++) {
            ws_gates(i, 2 * dic + j) = tanh_fwd(ws_gates(i, 2 * dic + j) + bias(2, j));
            states_t_l(i, j) = states_tm1_l(i, j) * ws_gates(i, 0 * dic + j) +
                (1.0f - ws_gates(i, 0 * dic +  j)) * ws_gates(i, 2 * dic + j);
        }
    });
}

template <>
elemwise_sig(_ref_rnn_common_t<prop_kind::forward>::gru_lbr_elemwise) {
    bool is_training = conf_.is_training();
    AOC<float, 3> ws_gates(ws_gates_, batch, conf_.GC());
    AOC<float, 2> ws_Wh_b(ws_grid_, batch, dic);
    AOC<const float, 2> bias(bias_, n_gates + 1, dic);
    AOC<float, 2> states_t_l(states_t_l_, batch, wic);
    AOC<float, 2> states_tm1_l(states_tm1_l_, batch, wic);
    AOC<float, 3> ws_gemm_state(ws_cell_, batch, conf_.GC());
    parallel_nd(batch, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < dic; j++) {
            float Wh_b = ws_gemm_state(i, 2 * dic + j) + bias(3, j);
            ws_gates(i, 0 * dic + j) = logistic_fwd(ws_gates(i, 0 * dic + j) +
                ws_gemm_state(i, j) + bias(0, j));
            ws_gates(i, 1 * dic + j) = logistic_fwd(ws_gates(i, 1 * dic + j) +
                ws_gemm_state(i, dic + j) + bias(1, j));
            ws_gates(i, 2 * dic + j) = tanh_fwd(ws_gates(i, 2 * dic + j) +
                ws_gates(i, 1 * dic + j) * Wh_b + bias(2, j));
            states_t_l(i, j) = states_tm1_l(i, j) * ws_gates(i, 0 * dic + j) +
                (1.0f - ws_gates(i, 0 * dic + j)) * ws_gates(i, 2 * dic + j);
            if (is_training) ws_Wh_b(i, j) = Wh_b;
        }
    });
}

template <>
cell_execution_sig(_ref_rnn_common_t<prop_kind::forward>::cell_execution_gru_lbr) {
    if (!merge_gemm_layer) {
        (this->*gemm_input_func)(n_gates * dic, batch, slc, conf_.WL_GLD(), slc,
                batch, wic, conf_.GC(), batch, w_input_[0], states_t_lm1_,
                ws_gates_, false, 0.0f);
    }
    (this->*gemm_state_func)(n_gates * dic, batch, sic, conf_.WI_GLD(), sic,
            batch, wic, conf_.GC(), batch, w_state_[0], states_tm1_l_, ws_cell_,
            false, 0.0f);
    (this->*elemwise_func)(dic, wic, batch, n_states, iter_stride, n_gates, ws_gates_,
            states_t_l_, states_t_lm1_, states_tm1_l_, diff_states_t_l_,
            diff_states_t_lp1_, diff_states_tp1_l_, bias_, ws_grid_, ws_cell_);
}

template <>
elemwise_sig(_ref_rnn_common_t<prop_kind::backward>::gru_lbr_elemwise) {
    AOC<float, 3> ws_gates(ws_gates_, batch, conf_.GC());
    AOC<const float, 2> states_tm1_l(states_tm1_l_, batch, wic);
    AOC<float, 4> diff_states_t_l(diff_states_t_l_, n_states + 1, iter_stride, batch, wic);//dht-1 dxt
    AOC<float, 4> diff_states_tp1_l(
        diff_states_tp1_l_, n_states + 1, iter_stride, batch, wic);
    AOC<float, 4> diff_states_t_lp1(
        diff_states_t_lp1_, n_states + 1, iter_stride, batch, wic);
    AOC<float, 3> ws_gates_r(ws_cell_, batch, conf_.GC());
    AOC<float, 2> ws_Wh_b(ws_grid_, batch, dic);

    // 1. calculate dG1 dG2 dG3
    // dG0 = (dht - G2) * dht * (1 - G0) * G0
    // dG1 = (W*h + b) * dG2 * (1 - G1) * G1
    // dG2 = (1 - G0) * dht * (1 - G2*G2)
    parallel_nd(batch, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < dic; j++) {
            float h = states_tm1_l(i, j);
            float dHt = diff_states_tp1_l(0, 0, i, j)
                    + diff_states_t_lp1(n_states, 0, i, j);
            float dG0 = (h - ws_gates(i, 2 * dic + j)) * dHt
                    * x_m_square(ws_gates(i, 0 * dic + j));
            float dG2 = (1.0f - ws_gates(i, 0 * dic + j))
                    * one_m_square(ws_gates(i, 2 * dic + j)) * dHt;
            float dG1 = ws_Wh_b(i, j) * dG2
                    * x_m_square(ws_gates(i, 1 * dic + j));

            diff_states_t_l(0, 0, i, j) = dHt * ws_gates(i, 0 * dic + j);
            ws_gates(i, 2 * dic + j) = dG2;
            ws_gates_r(i, 2 * dic + j) = dG2 * ws_gates(i, 1 * dic + j);
            ws_gates(i, 0 * dic + j) = ws_gates_r(i, 0 * dic + j) = dG0;
            ws_gates(i, 1 * dic + j) = ws_gates_r(i, 1 * dic + j) = dG1;
        }
    });
}

template <>
cell_execution_sig(_ref_rnn_common_t<prop_kind::backward>::cell_execution_gru_lbr) {
    AOC<float, 2> diff_bias(diff_bias_, n_gates + 1, dic);
    AOC<float, 3> ws_gates_r(ws_cell_, batch, conf_.GC());

    (this->*elemwise_func)(dic, wic, batch, n_states, iter_stride, n_gates, ws_gates_,
            states_t_l_, states_t_lm1_, states_tm1_l_, diff_states_t_l_,
            diff_states_t_lp1_, diff_states_tp1_l_, bias_, ws_grid_, ws_cell_);

    if (!merge_gemm_layer) {
         //  dx = dG * Wx^t
         (this->*gemm_input_func)(slc, batch, n_gates * dic, conf_.WL_GLD(),
                 n_gates * dic, batch, conf_.GC(), wic, batch, w_input_[0],
                 ws_gates_,
                 diff_states_t_l_ + n_states * iter_stride * (batch * wic),
                 false, 0.0f);
         // dWx +=  dG^t * x
         gemm(n_gates * dic, slc, batch, conf_.GC(), batch, wic, batch,
                 conf_.DWL_GLD(), slc, ws_gates_, states_t_lm1_, diff_w_input_,
                 true, 1.0f);
    }
    // dh +=  dGr * Wh^t
    (this->*gemm_state_func)(sic, batch, n_gates * dic, conf_.WI_GLD(),
            n_gates * dic, batch, conf_.GC(), wic, batch, w_state_[0], ws_cell_,
            diff_states_t_l_, false, 1.0f);

    // dWh += dGr^t * h
    gemm(n_gates * dic, sic, batch, conf_.GC(), batch, wic, batch,
            conf_.DWL_GLD(), sic, ws_cell_, states_tm1_l_, diff_w_state_, true,
            1.0f);

    // db1-3 += e * dG
    // db4 += e * (r * dG2)
    gates_reduction(n_gates, dic, wic, batch, ws_gates_, diff_bias_);

    parallel_nd(dic, [&](int j) {
        for (int i = 0; i < batch; i++) {
            diff_bias_[3 * dic + j] += ws_gates_r(i, 2 *dic + j);
        }
    });
}

template <>
cell_execution_sig(_ref_rnn_common_t<prop_kind::backward>::cell_execution_gru) {
    AOC<float, 2> ws_gates(ws_gates_, batch, conf_.GC());
    AOC<const float, 2> states_tm1_l(states_tm1_l_, batch, wic);
    AOC<float, 4> diff_states_t_l(diff_states_t_l_, n_states + 1, iter_stride, batch, wic);//dht-1 dxt
    AOC<float, 3> diff_w_state(diff_w_state_, sic, conf_.GC());
    AOC<float, 4> diff_states_tp1_l(
        diff_states_tp1_l_, n_states + 1, iter_stride, batch, wic);
    AOC<float, 4> diff_states_t_lp1(
        diff_states_t_lp1_, n_states + 1, iter_stride, batch, wic);
    //use state memory for intermediate computations
    float *dhG1_ = &(diff_states_t_l(n_states, 0, 0, 0));
    float *hG1_ = dhG1_;
    AOC<float, 2> dhG1(dhG1_, batch, wic);
    AOC<float, 2> hG1(hG1_, batch, wic);

    // 1. calculate dG2, dG1, and part of dht-1
    // dG2^ = dh * (1 - G0) * (1 - G2^2)
    // dG0^ = dh * (ht-1 - G2) * u * (1 - G0)
    // dht-1 (part) = dh * G0
    parallel_nd(batch, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < dic; j++) {
            float h = states_tm1_l(i, j);
            float dHt = diff_states_tp1_l(0, 0, i, j)
                    + diff_states_t_lp1(n_states, 0, i, j);
            float dG2 = (1.0f - ws_gates(i, 0 * dic + j)) * dHt
                    * one_m_square(ws_gates(i, 2 * dic + j));
            float dG0 = (h - ws_gates(i, 2 * dic + j)) * dHt
                    * x_m_square(ws_gates(i, 0 * dic + j));

            diff_states_t_l(0, 0, i, j) = dHt * ws_gates(i, 0 * dic + j);
            ws_gates(i, 0 * dic + j) = dG0;
            ws_gates(i, 2 * dic + j) = dG2;
        }
    });

    //2. calculate intermediate d(hG1)
    //d(hG1) = dG2 * W2h^t
    (this->*gemm_state_func)(sic, batch, dic, conf_.WI_GLD(), n_gates * dic,
            batch, conf_.GC(), wic, batch, w_state_[1], &(ws_gates(0, 2 * dic)),
            dhG1_, false, 0.0f);

    //3. calculate dG1^ and part of dht-1
    //dG1^ = d(hG1) * h * G1 * (1 - G1)
    //dht-1 (part) += d(hG1) * G1
    //h * G1 (required for dWh)
    parallel_nd(batch, [&](int i) {
        PRAGMA_OMP_SIMD()
        for (int j = 0; j < dic; j++) {
            float h = states_tm1_l(i, j);
            float G1 =  ws_gates(i, 1 * dic + j);
            diff_states_t_l(0, 0, i, j) += dhG1(i, j) * G1;
            ws_gates(i, 1 * dic + j) = dhG1(i, j) * h * x_m_square(G1);
            hG1(i, j) = G1 * h;
        }
    });

    //4. calculate diff weights
    //dWh1 += dG1 * h, dWh2 += dG2 * h, dWh3 += dG3 * (G1(*)h)
    gemm((n_gates - 1) * dic, sic, batch, conf_.GC(), batch, wic, batch,
            conf_.DWI_GLD(), sic, ws_gates_, states_tm1_l_, diff_w_state_, true,
            1.0f);
    gemm(dic, sic, batch, conf_.GC(), batch, wic, batch, conf_.DWI_GLD(), sic,
            &(ws_gates(0, 2 * dic)), hG1_, &(diff_w_state(0, 2 * dic)), true,
            1.0f);

    //5. calculate diff states
    //dht-1 += dG1 * W1h + dG0 * W0h
    (this->*gemm_state_func)(sic, batch, (n_gates - 1) * dic, conf_.WI_GLD(),
            n_gates * dic, batch, conf_.GC(), wic, batch, w_state_[0],
            ws_gates_, diff_states_t_l_, false, 1.0f);

    if (!merge_gemm_layer) {
        //dWx += [dG0 dG1 dG2] * [x]
        gemm(n_gates * dic, slc, batch, conf_.GC(), batch, wic, batch,
                conf_.DWL_GLD(), slc, ws_gates_, states_t_lm1_, diff_w_input_,
                true, 1.0f);
        //dx = dG2 * W2x + dG1 * W1x + dG0 * W0x
        (this->*gemm_input_func)(slc, batch, n_gates * dic, conf_.WL_GLD(),
                n_gates * dic, batch, conf_.GC(), wic, batch, w_input_[0],
                ws_gates_, &(diff_states_t_l(n_states, 0, 0, 0)), false, 0.0f);
    }

    //6. calculate diff bias
    gates_reduction(n_gates, dic, wic, batch, ws_gates_, diff_bias_);
}

//*************** Grid computations strategy: linear ***************//
template <prop_kind_t aprop>
grid_execution_sig(_ref_rnn_common_t<aprop>::linear_execution) {
    AOC<float, 5> ws_states(ws_states_, n_layer + 1, n_direction, n_states, n_iter + 1,
            batch * wic);
    AOC<float, 5> ws_diff_states(ws_diff_states_, n_layer + 1, n_direction, (n_states + 1),
            n_iter + 1, batch * wic);
    AOC<float, 4> ws_gates(
            ws_gates_, n_layer, n_direction, n_iter, batch * conf_.GC());
    AOC<float *, 3> weights_input(weights_input_, n_layer, n_direction,
            n_parts_wei_i);
    AOC<float *, 3> weights_states(weights_states_, n_layer, n_direction,
            n_parts_wei_st);
    AOC<const float, 3> bias(bias_, n_layer, n_direction, n_bias * dic);
    AOC<float, 3> diff_weights_layer(
            diff_weights_layer_, n_layer, n_direction, slc * conf_.DWL_GLD());
    AOC<float, 3> diff_weights_iter(
            diff_weights_iter_, n_layer, n_direction, sic * conf_.DWI_GLD());
    AOC<float, 3> diff_bias(diff_bias_, n_layer, n_direction, n_bias * dic);
    AOC<float, 4> ws_grid(ws_grid_, n_layer, n_direction, n_iter, ws_per_cell);

    // We run the grid of computation
    for (int dir = 0; dir < n_direction; dir++) {
        for (int j = 0; j < n_layer; j++) {
            int lay = (aprop == prop_kind::forward) ? j : n_layer - j - 1;
            if ((aprop == prop_kind::forward) && merge_gemm_layer) {
                /* Assumption: merge_gemm_layer happens only on forward */
                (this->*gemm_input_func)(n_gates * dic, batch * n_iter, slc,
                        conf_.WL_GLD(), slc, batch * n_iter, wic, conf_.GC(),
                        batch * n_iter, weights_input(lay, dir, 0),
                        &(ws_states(lay, dir, 0, 1, 0)),
                        &(ws_gates(lay, dir, 0, 0)), false, 0.0f);
            }
            for (int i = 0; i < n_iter; i++) {
                int iter = (aprop == prop_kind::forward) ? i : n_iter - i - 1;
                (this->*cell_func)(dic, slc, sic, wic, batch, n_gates, n_states, n_iter + 1,
                        &(ws_states(lay + 1, dir, 0, iter + 1, 0)),
                        &(ws_diff_states(lay, dir, 0, iter, 0)),
                        &(weights_input(lay, dir, 0)),
                        &(weights_states(lay, dir, 0)),
                        &(bias(lay, dir, 0)),
                        &(ws_states(lay, dir, 0, iter + 1, 0)),
                        &(ws_states(lay + 1, dir, 0, iter, 0)),
                        &(ws_diff_states(lay + 1, dir, 0, iter, 0)),
                        &(ws_diff_states(lay, dir, 0, iter + 1, 0)),
                        &(diff_weights_layer(lay, dir, 0)),
                        &(diff_weights_iter(lay, dir, 0)),
                        &(diff_bias(lay, dir, 0)),
                        &(ws_gates(lay, dir, iter, 0)),
                        &(ws_grid(lay, dir, iter, 0)),
                        ws_cell_);
            }
            if ((aprop == prop_kind::backward) && merge_gemm_layer) {
                (this->*gemm_input_func)(slc, batch * n_iter, n_gates * dic,
                        conf_.WL_GLD(), n_gates * dic, batch * n_iter,
                        conf_.GC(), wic, batch * n_iter,
                        weights_input(lay, dir, 0), &(ws_gates(lay, dir, 0, 0)),
                        &(ws_diff_states(lay, dir, n_states, 0, 0)), false,
                        0.0f);
                gemm(n_gates * dic, slc, batch * n_iter, conf_.GC(),
                        batch * n_iter, wic, batch * n_iter, conf_.DWL_GLD(),
                        slc, &(ws_gates(lay, dir, 0, 0)),
                        &(ws_states(lay, dir, 0, 1, 0)),
                        &(diff_weights_layer(lay, dir, 0)), true, 1.0f);
            }
            if ((aprop == prop_kind::backward) && merge_gemm_iter) {
                gemm(n_gates * dic, sic, batch * n_iter, conf_.GC(),
                        batch * n_iter, wic, batch * n_iter, conf_.DWI_GLD(),
                        sic, &(ws_gates(lay, dir, 0, 0)),
                        &(ws_states(lay + 1, dir, 0, 0, 0)),
                        &(diff_weights_iter(lay, dir, 0)), true, 1.0f);
            }
        }
    }
}

//********* GRID computations strategy: utility functions **********//

template <>
void _ref_rnn_common_t<prop_kind::forward>::copy_init_layer(bool lr, bool rl,
        int n_layer, int n_direction, int n_iter, int batch, int slc, int dic,
        int dlc, int wic, int n_states, float *ws_states_,
        float *ws_diff_states_, const float *xt_,
        const float *diff_dst_layer_) {
    AOC<float, 5> ws_states(
            ws_states_, n_direction, n_states, n_iter + 1, batch, wic);
    auto xt_d = memory_desc_wrapper(conf_.src_pd(0));

    parallel_nd(n_iter, [&](int it) {
        auto xxt = xt_ + xt_d.blk_off(it);
        if (lr)
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < slc; c++)
                    ws_states(0, 0, it + 1, b, c) = *(xxt + b * slc + c);
        if (rl)
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < slc; c++)
                    ws_states(n_direction - 1, 0, n_iter - it, b, c)
                            = *(xxt + b * slc + c);
    });
}

template <>
void _ref_rnn_common_t<prop_kind::backward>::copy_init_layer(bool lr, bool rl,
        int n_layer, int n_direction, int n_iter, int batch, int slc, int dic,
        int dlc, int wic, int n_states, float *ws_states_,
        float *ws_diff_states_, const float *xt_,
        const float *diff_dst_layer_) {
    AOC<float, 6> ws_diff_states(ws_diff_states_, n_layer + 1, n_direction,
            (n_states + 1), n_iter + 1, batch, wic);
    auto diff_dst_layer_d = memory_desc_wrapper(conf_.diff_dst_pd(0));

    switch (conf_.direction()) {
    case mkldnn_bidirectional_concat:
        parallel_nd(n_iter, batch, [&](int it, int b) {
            auto diff_dst_layer_x
            = diff_dst_layer_ + diff_dst_layer_d.blk_off(it, b);
            for (int s = 0; s < dic; s++) {
                ws_diff_states(n_layer, 0, n_states, it, b, s)
                    = diff_dst_layer_x[s];
                ws_diff_states(n_layer, 1, n_states, n_iter - it - 1, b, s)
                    = diff_dst_layer_x[dic + s];
            }
        });
        break;
    case mkldnn_bidirectional_sum:
        parallel_nd(n_iter, batch, [&](int it, int b) {
            auto diff_dst_layer_x
            = diff_dst_layer_ + diff_dst_layer_d.blk_off(it, b);
            for (int s = 0; s < dic; s++) {
                ws_diff_states(n_layer, 0, n_states, it, b, s)
                    = diff_dst_layer_x[s];
                ws_diff_states(n_layer, 1, n_states, n_iter - it - 1, b, s)
                    = diff_dst_layer_x[s];
            }
        });
        break;
    case mkldnn_unidirectional_left2right:
        parallel_nd(n_iter, batch, [&](int it, int b) {
            auto diff_dst_layer_x
                    = diff_dst_layer_ + diff_dst_layer_d.blk_off(it, b);
            for (int s = 0; s < dic; s++) {
                ws_diff_states(n_layer, 0, n_states, it, b, s)
                        = diff_dst_layer_x[s];
            }
        });
        break;
    case mkldnn_unidirectional_right2left:
        parallel_nd(n_iter, batch, [&](int it, int b) {
            auto diff_dst_layer_x
                    = diff_dst_layer_ + diff_dst_layer_d.blk_off(n_iter - it - 1, b);
            for (int s = 0; s < dic; s++) {
                ws_diff_states(n_layer, 0, n_states, it, b, s)
                        = diff_dst_layer_x[s];
            }
        });
        break;
    default:
        assert(!"Unsupported direction");
        break;
    }
}

template <>
void _ref_rnn_common_t<prop_kind::forward>::copy_init_iter(int n_layer,
        int n_direction, int n_states, int batch, int sic, int dic, int wic,
        int n_iter, float *ws_states_, float *ws_diff_states_,
        const float *firstit_states_, const float *diff_dst_iter_) {
    AOC<float, 6> ws_states(ws_states_, n_layer + 1, n_direction, n_states,
            n_iter + 1, batch, wic);
    auto firstit_states_d = memory_desc_wrapper(conf_.src_pd(1));
    if (firstit_states_) {
        parallel_nd(n_layer, n_direction, [&](int lay, int dir) {
            for (int state = 0; state < n_states; state++)
                for (int b = 0; b < batch; ++b) {
                    array_copy(&(ws_states(lay + 1, dir, state, 0, b, 0)),
                        firstit_states_ + firstit_states_d.blk_off(
                        lay, dir, state, b), sic);
                }
        });
    } else {
        parallel_nd(n_layer, n_direction, [&](int lay, int dir) {
            for (int state = 0; state < n_states; state++)
                for (int i = 0; i < batch; i++)
                    for (int j = 0; j < sic; j++)
                        ws_states(lay + 1, dir, state, 0, i, j) = 0.0f;
        });
    }
}

template <>
void _ref_rnn_common_t<prop_kind::backward>::copy_init_iter(int n_layer,
        int n_direction, int n_states, int batch, int sic, int dic, int wic,
        int n_iter, float *ws_states_, float *ws_diff_states_,
        const float *firstit_states_, const float *diff_dst_iter_) {
    AOC<float, 6> ws_diff_states(ws_diff_states_, n_layer + 1, n_direction,
            n_states + 1, n_iter + 1, batch, wic);
    auto diff_dst_iter_d = memory_desc_wrapper(conf_.diff_dst_pd(1));
    if (diff_dst_iter_) {
        parallel_nd(n_layer, n_direction, n_states, batch,
            [&](int lay, int dir, int state, int b) {
            array_copy(&(ws_diff_states(lay, dir, state, n_iter, b, 0)),
                diff_dst_iter_ + diff_dst_iter_d.blk_off(lay, dir, state, b),
                dic);
        });
    } else {
        parallel_nd(n_layer, n_direction, n_states, batch,
            [&](int lay, int dir, int state, int i) {
            for (int j = 0; j < dic; j++)
                ws_diff_states(lay, dir, state, n_iter, i, j) = 0.0f;
        });
    }
}

template <>
void _ref_rnn_common_t<prop_kind::forward>::copy_res_layer(bool lr, bool rl,
        int n_layer, int n_direction, int n_iter, int batch,
        int n_output_features, int slc, int dic, int wic, int n_states,
        mkldnn_rnn_direction_t direction, float *dst_layer_,
        float *diff_src_layer, const float *ws_states_,
        const float *ws_diff_states_) {
    auto dst_layer_d = memory_desc_wrapper(conf_.dst_pd(0));
    AOC<const float, 6> ws_states(ws_states_, n_layer + 1, n_direction,
            n_states, n_iter + 1, batch, wic);

    parallel_nd(n_iter, batch, [&](int it, int b) {
        int dir = 0;
        if (lr) {
            for (int s = 0; s < dic; s++)
                dst_layer_[dst_layer_d.blk_off(it, b, dir * dic + s)]
                        = ws_states(n_layer, dir, 0, it + 1, b, s);
            dir = 1;
        }
        if (rl) {
            for (int s = 0; s < dic; s++)
                switch (direction) {
                case mkldnn_bidirectional_sum:
                    dst_layer_[dst_layer_d.blk_off(it, b, s)] += ws_states(
                            n_layer, dir, 0, n_iter - it, b, s);
                    break;
                default:
                    dst_layer_[dst_layer_d.blk_off(it, b, dir * dic + s)]
                            = ws_states(n_layer, dir, 0, n_iter - it, b, s);
                }
        }
    });
}

template <>
void _ref_rnn_common_t<prop_kind::backward>::copy_res_layer(bool lr, bool rl,
        int n_layer, int n_direction, int n_iter, int batch,
        int n_output_features, int slc, int dic, int wic, int n_states,
        mkldnn_rnn_direction_t direction, float *dst_layer_,
        float *diff_src_layer_, const float *ws_states_,
        const float *ws_diff_states_) {
    auto diff_src_layer_d = memory_desc_wrapper(conf_.diff_src_pd(0));
    AOC<const float, 6> ws_diff_states(ws_diff_states_, n_layer + 1,
            n_direction, n_states + 1, n_iter + 1, batch, wic);

    parallel_nd(n_iter, batch, [&](int it, int b) {
        int dir = 0;
        for (int s = 0; s < slc; s++) {
            float *dst_addr = diff_src_layer_
                    + diff_src_layer_d.blk_off(
                              (direction
                                      == mkldnn_unidirectional_right2left) ?
                                      n_iter - 1 - it :
                                      it,
                              b, dir * slc + s);
            float res = ws_diff_states(0, 0, n_states, it, b, s);
            if (n_direction - 1)
                res += ws_diff_states(
                        0, 1, n_states, n_iter - 1 - it, b, s);
            dst_addr[0] = res;
        }
    });
}

template <>
void _ref_rnn_common_t<prop_kind::forward>::copy_res_iter(int n_layer,
        int n_direction, int n_states, int batch, int sic, int dic, int wic,
        int n_iter, float *dst_iter_, float *diff_src_iter_,
        const float *ws_states_, const float *ws_diff_states_) {
    auto dst_iter_d = memory_desc_wrapper(conf_.dst_pd(1));
    AOC<const float, 6> ws_states(ws_states_, n_layer + 1, n_direction,
            n_states, n_iter + 1, batch, wic);
    if (dst_iter_) {
        parallel_nd(n_layer, n_direction, n_states, batch,
            [&](int lay, int dir, int state, int b) {
            for (int s = 0; s < dic; s++) {
                dst_iter_[dst_iter_d.blk_off(lay, dir, state, b, s)]
                        = ws_states(lay + 1, dir, state, n_iter, b, s);
            }
        });
    }
}

template <>
void _ref_rnn_common_t<prop_kind::backward>::copy_res_iter(int n_layer,
        int n_direction, int n_states, int batch, int sic, int dic, int wic,
        int n_iter, float *dst_iter_, float *diff_src_iter_,
        const float *ws_states_, const float *ws_diff_states_) {
    auto diff_src_iter_d = memory_desc_wrapper(conf_.diff_src_pd(1));
    AOC<const float, 6> ws_diff_states(ws_diff_states_, n_layer + 1,
            n_direction, n_states + 1, n_iter + 1, batch, wic);
    if (diff_src_iter_) {
        parallel_nd(n_layer, n_direction, n_states, batch,
            [&](int lay, int dir, int state, int b) {
            for (int s = 0; s < sic; s++) {
                diff_src_iter_[diff_src_iter_d.blk_off(
                        lay, dir, state, b, s)]
                        = ws_diff_states(lay, dir, state, 0, b, s);
            }
        });
    }
}

template <prop_kind_t aprop>
packing_sig(_ref_rnn_common_t<aprop>::pack_weights) {
#if (USE_MKL_PACKED_GEMM)
    AOC<const float, 5> w(
            w_, n_layer, n_direction, IC_size, n_gates, OC_size);
    AOC<float *, 3> weights(weights_, n_layer, n_direction, n_parts);
    int m = 0, n = 0, k = 0;
    auto transA = CblasNoTrans;
    bool is_fwd = aprop == prop_kind::forward;
    if (is_fwd) {
        m = n_gates * OC_size;
        n = batch;
        k = IC_size;
        //todo: do a transposition if ldgoi
        transA = CblasNoTrans;
    } else {
        m = IC_size;
        n = batch;
        k = n_gates * OC_size;
        //TODO: do a transposition if ldigo
        transA = CblasNoTrans;
    }
    for (int i = 0; i < n_layer; i++) {
        for (int d = 0; d < n_direction; d++) {
            for (int p = 0; p < n_parts; p++) {
                int m_p = is_fwd ? (gates_per_part[p] * OC_size) : m;
                int k_p = is_fwd ? k : (gates_per_part[p] * OC_size);
                int g = (p > 0) ? gates_per_part[p - 1] : 0;
                weights(i, d, p) = cblas_sgemm_alloc(CblasAMatrix, m_p, n, k_p);
                cblas_sgemm_pack(CblasColMajor, CblasAMatrix, transA, m_p, n,
                        k_p, 1.0f, &(w(i, d, 0, g, 0)), m, weights(i, d, p));
            }
        }
    }
#else
    UNUSED(n_layer);
    UNUSED(n_direction);
    UNUSED(n_weights);
    UNUSED(n_gates);
    UNUSED(n_parts);
    UNUSED(gates_per_part);
    UNUSED(batch);
    UNUSED(OC_size);
    UNUSED(IC_size);
    UNUSED(weights_);
    UNUSED(w_);
    assert(!"packed gemm is disabled");
#endif
}

template <prop_kind_t aprop>
packing_sig(_ref_rnn_common_t<aprop>::no_pack_weights) {
    AOC<const float, 3> w(
            w_, n_layer, n_direction, IC_size * n_gates * OC_size);
    AOC<float *, 3> weights(weights_, n_layer, n_direction, n_parts);
    int m = 0, n = 0, ldA = 0;

    bool is_fwd = aprop == prop_kind::forward;
    if (is_fwd) {
        m = n_gates * OC_size;
        n = IC_size;
        ldA = conf_.GC();
    } else {
        m = IC_size;
        n = n_gates * OC_size;
        ldA = conf_.WIC();
    }

    if (!do_copy) {
        for (int i=0; i < n_layer; i++)
            for (int d = 0; d < n_direction; d++) {
                weights(i, d, 0) = (float *) &(w(i, d, 0));
                for (int p = 1; p < n_parts; p++) {
                    size_t offset = is_fwd
                        ? gates_per_part[p - 1] * OC_size
                        : gates_per_part[p - 1] * OC_size * IC_size;
                    weights(i, d, p) = (float *) &w(i, d, offset);
                }
            }
        return;
    }

    /* We always assume
       - column major
       - alpha = 1.0f
    */
    auto copy_matrix = [](char trans, int nrows, int ncols,
            const float *src, const int ld_src, float *dst, const int ld_dst){
        for (int i = 0; i < ncols; i++)
            for (int j = 0; j < nrows; j++)
                dst[i * ld_dst + j] = src[i * ld_src + j];
    };

    AOC<float, 3> tmp(scratch_mem, n_layer, n_direction, ldA * n);
    mkldnn::impl::parallel_nd(n_layer, n_direction, [&](int i, int d) {
            auto src_mat = &(w(i, d, 0));
            auto dst_mat = &(tmp(i, d, 0));
            copy_matrix('N', m, n, src_mat, m, dst_mat, ldA);
            weights(i, d, 0) = &tmp(i, d, 0);
            for (int p = 1; p < n_parts; p++) {
                size_t offset = is_fwd
                    ? gates_per_part[p - 1] * OC_size
                    : gates_per_part[p - 1] * OC_size * conf_.WIC();
                weights(i, d, p) = &tmp(i, d, offset);
            }
        });
}


template <prop_kind_t aprop>
free_packed_sig(_ref_rnn_common_t<aprop>::free_packed_weights) {
#if (USE_MKL_PACKED_GEMM)
    AOC<float *, 3> weights(weights_, n_layer, n_direction, n_parts);
    for (int i = 0; i < n_layer; i++)
        for (int j = 0; j < n_direction; j++)
            for (int k = 0; k < n_parts; k++)
                cblas_sgemm_free(weights(i, j, k));
#else
    UNUSED(n_layer);
    UNUSED(n_direction);
    UNUSED(n_parts);
    UNUSED(weights_);
    assert(!"packed gemm is disabled");
#endif
}

template <prop_kind_t aprop>
free_packed_sig(_ref_rnn_common_t<aprop>::free_no_packed_weights) {
    // IN this case, only scratchpad is used, so no free necessary
}

//********************* Execution function *********************//
template <prop_kind_t aprop>
void _ref_rnn_common_t<aprop>::execute_() {
    int n_layer = conf_.L();
    int n_direction = conf_.D();
    int n_iter = conf_.T();
    int n_gates = conf_.G();
    int n_bias = n_gates + conf_.is_lbr();
    int n_states = conf_.S();
    int n_weights_input = conf_.SLC();
    int n_weights_state = conf_.SIC();
    int batch = conf_.MB();
    int slc = conf_.SLC();
    int sic = conf_.SIC();
    int dic = conf_.DIC();
    int dlc = conf_.DLC();
    int wic = conf_.WIC();

    bool is_orig_gru = conf_.cell_kind()
        == alg_kind::vanilla_gru;
    int n_parts_wei_st = is_orig_gru ? 2 : 1, n_parts_wei_i = 1;
    int parts_wei_st = n_gates, parts_wei_i = n_gates,
        parts_wei_st_gru[2] = {2, 1};
    bool is_fwd = aprop == prop_kind::forward;
    int ws_per_cell = conf_.ws_per_cell();

    int input_idx = 0;
    int output_idx = 0;
    auto input
            = reinterpret_cast<const float *>(this->input_memory(input_idx++));
    auto states = conf_.with_src_iter() ?
            reinterpret_cast<const float *>(this->input_memory(input_idx++)) :
            nullptr;
    auto w_input
            = reinterpret_cast<const float *>(this->input_memory(input_idx++));
    auto w_state
            = reinterpret_cast<const float *>(this->input_memory(input_idx++));
    auto bias = conf_.with_bias() ?
            reinterpret_cast<const float *>(this->input_memory(input_idx++)) :
            nullptr;

    auto dst_last_layer = is_fwd ?
            reinterpret_cast<float *>(this->memory(output_idx++)) :
            const_cast<float *>(reinterpret_cast<const float *>(
                    this->input_memory(input_idx++)));
    auto dst_last_iter = conf_.with_dst_iter() ?
            (is_fwd ? reinterpret_cast<float *>(this->memory(output_idx++)) :
                      const_cast<float *>(reinterpret_cast<const float *>(
                              this->input_memory(input_idx++)))) :
            nullptr;

    auto diff_dst_layer = is_fwd ?
            nullptr :
            reinterpret_cast<const float *>(this->input_memory(input_idx++));
    auto diff_dst_iter = is_fwd || !conf_.with_dst_iter() ?
            nullptr :
            reinterpret_cast<const float *>(this->input_memory(input_idx++));

    // fetchihg buffers from the workspace
    // if no workspace was provided we use the scratchpad
    float *scratch_ptr = ((float *)scratchpad_->get());
    float *ws_ptr = nullptr;
    if (use_workspace_)
        ws_ptr = is_fwd ?
            reinterpret_cast<float *>(this->memory(output_idx++)) :
            const_cast<float *>(reinterpret_cast<const float *>(
                    this->input_memory(input_idx++)));
    float *base_ptr = use_workspace_ ? ws_ptr : scratch_ptr;
    ws_gates_ = base_ptr + ws_gates_offset_;
    ws_states_ = base_ptr + ws_states_offset_;
    ws_diff_states_ = base_ptr + ws_diff_states_offset_;
    ws_grid_ = base_ptr + ws_grid_comp_offset_;
    ws_cell_ = base_ptr + ws_cell_comp_offset_;

    auto diff_src_layer = is_fwd ?
            nullptr :
            reinterpret_cast<float *>(this->memory(output_idx++));
    auto diff_src_iter = is_fwd || !conf_.with_src_iter() ?
            nullptr :
            reinterpret_cast<float *>(this->memory(output_idx++));
    auto diff_weights_layer = is_fwd ?
            nullptr :
            reinterpret_cast<float *>(this->memory(output_idx++));
    auto diff_weights_iter = is_fwd ?
            nullptr :
            reinterpret_cast<float *>(this->memory(output_idx++));
    auto diff_bias = is_fwd || !conf_.with_bias() ?
            nullptr :
            reinterpret_cast<float *>(this->memory(output_idx++));

    // Fetching extra buffers from scratchpad
    ws_weights_layer_ = scratch_ptr + ws_weights_layer_offset_;
    ws_weights_iter_ = scratch_ptr + ws_weights_iter_offset_;
    ws_diff_weights_layer_ = scratch_ptr + ws_diff_weights_layer_offset_;
    ws_diff_weights_iter_ = scratch_ptr + ws_diff_weights_iter_offset_;


// initialize diff_states to 0
    if (aprop == prop_kind::backward) {
        array_set(ws_diff_states_, 0.0f, conf_.ws_diff_states_size());
        // TODO: add a variable to check if good_ld_copy is necessary
        if (copy_diff_weights_layer_) {
            parallel_nd(conf_.ws_diff_weights_layer_size(), [&](size_t i) {
                ws_diff_weights_layer_[i] = 0.;
            });
        } else
            ws_diff_weights_layer_ = diff_weights_layer;
        if (copy_diff_weights_iter_) {
            parallel_nd(conf_.ws_diff_weights_iter_size(), [&](size_t i) {
                ws_diff_weights_iter_[i] = 0.;
            });
        } else
            ws_diff_weights_iter_ = diff_weights_iter;
    }

    // TODO: implement without copies
    bool is_lr = !one_of(exec_dir, b2t_r2l, t2b_r2l);
    bool is_rl = !one_of(exec_dir, b2t_l2r, t2b_l2r);
    // we pack the weights if we are using the packed API
    (this->*weights_state_pack_func)(n_layer, n_direction, n_weights_state,
            n_gates, batch, dic, sic, ptr_wei_state_, n_parts_wei_st,
            (is_orig_gru ? parts_wei_st_gru : &parts_wei_st), w_state,
            ws_weights_iter_, copy_weights_iter_);
    (this->*weights_input_pack_func)(n_layer, n_direction, n_weights_input,
            n_gates, batch, dic, slc, ptr_wei_input_, n_parts_wei_i,
            &parts_wei_i, w_input,
            ws_weights_layer_, copy_weights_layer_);

    // we first need to copy the initial states and input into ws
    copy_init_layer(is_lr, is_rl, n_layer, n_direction, n_iter, batch, slc, dic,
            dlc, wic, n_states, ws_states_, ws_diff_states_, input,
            diff_dst_layer);
    copy_init_iter(n_layer, n_direction, n_states, batch, sic, dic, wic, n_iter,
            ws_states_, ws_diff_states_, states, diff_dst_iter);

    // run the execution on the grid
    (this->*grid_computation)(dic, slc, sic, wic, batch, n_layer, n_direction,
            n_iter, n_gates, n_states, n_bias, ptr_wei_input_, n_parts_wei_i,
            ptr_wei_state_, n_parts_wei_st, (float *)bias, ws_states_,
            ws_diff_states_, ws_gates_, ws_cell_, ws_grid_, ws_per_cell,
            ws_diff_weights_layer_, ws_diff_weights_iter_, diff_bias);

    // Finally we copy the results to the result buffers
    copy_res_layer(is_lr, is_rl, n_layer, n_direction, n_iter, batch,
            n_output_features, slc, dic, wic, n_states, conf_.direction(),
            dst_last_layer, diff_src_layer, ws_states_, ws_diff_states_);
    copy_res_iter(n_layer, n_direction, n_states, batch, sic, dic, wic, n_iter,
            dst_last_iter, diff_src_iter, ws_states_, ws_diff_states_);

    // copy of the diff weights if bwd
    if (aprop == prop_kind::backward){
        // TODO: write an impl of matcopy in MKL-DNN
        // TODO: support ldgoi using the trans parameters
        AOC<float, 3> diff_weights_layer_aoc(diff_weights_layer, n_layer, n_direction, slc * n_gates * dic);
        AOC<float, 3> diff_weights_iter_aoc(diff_weights_iter, n_layer, n_direction, sic * n_gates * dic);
        AOC<float, 3> ws_diff_weights_layer_aoc(ws_diff_weights_layer_, n_layer, n_direction, slc * conf_.GC());
        AOC<float, 3> ws_diff_weights_iter_aoc(ws_diff_weights_iter_, n_layer, n_direction, sic * conf_.GC());

        /*
           - assumes column major and non transposed matrices
           - computes B = A + B
        */
        auto inplace_matadd = [=](const int nrows, const int ncols,
                const float *A, const int ldA, float *B, const int ldB){
            for(int i = 0; i < ncols; i++)
                for(int j = 0; j < nrows; j++)
                    B[i * ldB + j] += A[i * ldA + j];
        };
        mkldnn::impl::parallel_nd(n_layer, n_direction, [&](int i, int d) {
            auto wei_lay = &(diff_weights_layer_aoc(i, d, 0));
            auto wei_it = &(diff_weights_iter_aoc(i, d, 0));
            auto ws_wei_lay = &(ws_diff_weights_layer_aoc(i, d, 0));
            auto ws_wei_it = &(ws_diff_weights_iter_aoc(i, d, 0));
            if (copy_diff_weights_layer_)
                inplace_matadd(n_gates*dic, slc, ws_wei_lay, conf_.GC(),
                        wei_lay, n_gates*dic);
            if (copy_diff_weights_iter_)
                inplace_matadd(n_gates*dic, sic, ws_wei_it, conf_.GC(),
                        wei_it, n_gates*dic);
        });
    }

    // We free the packed weights if they were packed internally
    (this->*weights_state_free_packed_func)(n_layer, n_direction,
            n_parts_wei_st, ptr_wei_state_);
    (this->*weights_input_free_packed_func)(n_layer, n_direction,
            n_parts_wei_i, ptr_wei_input_);
};

template struct _ref_rnn_common_t<prop_kind::forward>;
template struct _ref_rnn_common_t<prop_kind::backward>;

#undef AOC
}
}
}
