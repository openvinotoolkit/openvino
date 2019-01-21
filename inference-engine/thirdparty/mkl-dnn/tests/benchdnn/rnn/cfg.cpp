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

#include "mkldnn_common.hpp"
#include "rnn/rnn.hpp"

namespace rnn {

/* cfgs definition
arrays:
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
params: {data_type, min, max, f_min,* f_max, f_base, f_step, f_sparsity, eps}
*/

const int int_max_exact = 1 << 24;
const _dt_conf_t conf_f32 = {
#if 0
        { mkldnn_f32, -int_max_exact, int_max_exact,  1,  1, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  1,  1, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  1,  1, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  1,  1, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  1,  1, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  777,  777, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  777,  777, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  777,  777, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  777,  777, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  777,  777, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  777,  777, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  777,  777, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  2,  2, 0, 1, .25, 1e-5 },
        { mkldnn_f32, -int_max_exact, int_max_exact,  2,  2, 0, 1, .25, 1e-5 },
#elif 0
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -4, 4, 0, 1, .25, 1e-5 },
#else
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
    { mkldnn_f32, -int_max_exact, int_max_exact, -64, 64, 0, 1, 1., 1e-5 },
#endif
};

const dt_conf_t *str2cfg(const char *str) {
#define CASE(cfg)                         \
    if (!strcasecmp(STRINGIFY(cfg), str)) \
    return CONCAT2(conf_, cfg)
    CASE(f32);
#undef CASE
    []() {
        SAFE(FAIL, CRIT);
        return 0;
    }();
    return (const dt_conf_t *)1;
}

const char *cfg2str(const dt_conf_t *cfg) {
#define CASE(_cfg)                   \
    if (cfg == CONCAT2(conf_, _cfg)) \
    return STRINGIFY(_cfg)
    CASE(f32);
#undef CASE
    []() {
        SAFE(FAIL, CRIT);
        return 0;
    }();
    return NULL;
}
} // namespace rnn
