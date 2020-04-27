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

#include "mkldnn_debug.h"

#include "common.hpp"
#include "mkldnn_debug.hpp"

const char *status2str(mkldnn_status_t status) {
    return mkldnn_status2str(status);
}

const char *dt2str(mkldnn_data_type_t dt) {
    return mkldnn_dt2str(dt);
}

mkldnn_data_type_t str2dt(const char *str) {
#define CASE(_dt) \
    if (!strcasecmp(STRINGIFY(_dt), str) \
            || !strcasecmp(STRINGIFY(CONCAT2(mkldnn_, _dt)), str)) \
        return CONCAT2(mkldnn_, _dt);
    CASE(s8);
    CASE(u8);
    CASE(s16);
    CASE(s32);
    CASE(f32);
    CASE(bf16);
#undef CASE
    assert(!"unknown data type");
    return mkldnn_f32;
}

const char *rmode2str(mkldnn_round_mode_t rmode) {
#define CASE(_rmode) \
    if (CONCAT2(mkldnn_round_, _rmode) == rmode) return STRINGIFY(_rmode)
    CASE(nearest);
    CASE(down);
#undef CASE
    assert(!"unknown round mode");
    return "unknown round mode";
}

mkldnn_round_mode_t str2rmode(const char *str) {
#define CASE(_rmd) do { \
    if (!strncasecmp(STRINGIFY(_rmd), str, strlen(STRINGIFY(_rmd)))) \
        return CONCAT2(mkldnn_round_, _rmd); \
} while (0)
    CASE(nearest);
    CASE(down);
#undef CASE
    assert(!"unknown round_mode");
    return mkldnn_round_nearest;
}

const char *fmt2str(mkldnn_memory_format_t fmt) {
    return mkldnn_fmt2str(fmt);
}

mkldnn_memory_format_t str2fmt(const char *str) {
    return mkldnn_str2fmt(str);
}
