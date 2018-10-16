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

#ifndef _INPUT_REORDER_HPP
#define _INPUT_REORDER_HPP

#include "mkldnn_common.hpp"
#include "reorder/reorder.hpp"

#define MASK_SUPPORTED 1
#define SCALE_SUPPORTED 1

#if SCALE_SUPPORTED
    const float default_scales[] = {0.125, 0.25, 0.5, 1, 2, 4, 8};
#else
    const float default_scales[] = {1};
#endif

const int int_max_exact = 1<<24;

const reorder::dt_conf_t conf_f32 = {mkldnn_f32, -int_max_exact, 2*int_max_exact};
const reorder::dt_conf_t conf_s8 = {mkldnn_s8, INT8_MIN, -2*INT8_MIN};
const reorder::dt_conf_t conf_u8 = {mkldnn_u8, 0, UINT8_MAX};
const reorder::dt_conf_t conf_s16 = {mkldnn_s16, INT16_MIN, -2*INT16_MIN};
const reorder::dt_conf_t conf_s32 = {mkldnn_s32, -int_max_exact, 2*int_max_exact};

static reorder::reorder_conf_t reorders[] = {
    /* ndims, dims, fmt_in, fmt_out */
    {4, {2, 64, 13, 13}, mkldnn_nchw, mkldnn_nchw},
    {4, {2, 64, 13, 13}, mkldnn_nChw8c, mkldnn_nhwc},
    {4, {2, 64, 13, 13}, mkldnn_nhwc, mkldnn_nChw8c},
    {4, {2, 64, 13, 13}, mkldnn_nChw16c, mkldnn_nhwc},
    {4, {2, 64, 13, 13}, mkldnn_nhwc, mkldnn_nChw16c},
    {4, {2, 64, 13, 13}, mkldnn_nchw, mkldnn_nhwc},
    {4, {2, 64, 13, 13}, mkldnn_nhwc, mkldnn_nchw},
    {4, {2, 64, 13, 13}, mkldnn_oihw, mkldnn_oihw},
    {4, {2, 64, 13, 13}, mkldnn_hwio, mkldnn_oihw},
};

static reorder::q10n_conf_t q10ns[] = {
    /* dt_{in,out}, conf, rmode, policy, scale */
    /* f32 <-> f32 */
    { conf_f32, conf_f32, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},
    /* u8 <-> u8 */
    { conf_u8, conf_u8, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},
    /* s8 <-> s8 */
    { conf_s8, conf_s8, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},
    /* s32 <-> s32 */
    { conf_s32, conf_s32, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},
    /* s8 <-> u8 */
    { conf_s8, conf_u8, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},
    { conf_u8, conf_s8, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},
    /* f32 <-> s32 */
    { conf_f32, conf_s32, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},
    { conf_f32, conf_s32, attr_t::round_mode_t::DOWN,
        attr_t::scale_t::policy_t::COMMON},
    { conf_s32, conf_f32, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},
    /* f32 <-> s8 */
    { conf_f32, conf_s8, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},
    { conf_s8, conf_f32, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},
    /* f32 <-> u8 */
    { conf_f32, conf_u8, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},
    { conf_u8, conf_f32, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},
    /* s32 <-> s8 */
    { conf_s32, conf_s8, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},
    { conf_s8, conf_s32, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},
    /* s32 <-> u8 */
    { conf_s32, conf_u8, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},
    { conf_u8, conf_s32, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::COMMON},

#if MASK_SUPPORTED
    { conf_f32, conf_f32, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::PER_OC},
    { conf_s8, conf_f32, attr_t::round_mode_t::NEAREST,
        attr_t::scale_t::policy_t::PER_OC},
#endif
};

#endif
