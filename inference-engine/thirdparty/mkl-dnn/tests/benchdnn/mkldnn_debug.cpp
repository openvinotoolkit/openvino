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

#include "common.hpp"
#include "mkldnn_debug.hpp"

const char *status2str(mkldnn_status_t status) {
#   define CASE(s) case s: return #s
    switch (status) {
    CASE(mkldnn_success);
    CASE(mkldnn_out_of_memory);
    CASE(mkldnn_try_again);
    CASE(mkldnn_invalid_arguments);
    CASE(mkldnn_not_ready);
    CASE(mkldnn_unimplemented);
    CASE(mkldnn_iterator_ends);
    CASE(mkldnn_runtime_error);
    CASE(mkldnn_not_required);
    }
    return "unknown error";
#   undef CASE
}

const char *dt2str(mkldnn_data_type_t dt) {
#define CASE(_dt) if (CONCAT2(mkldnn_, _dt) == dt) return STRINGIFY(_dt)
    CASE(s8);
    CASE(u8);
    CASE(s16);
    CASE(s32);
    CASE(f32);
#undef CASE
    assert(!"unknown data type");
    return "unknown data type";
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
#define CASE(_fmt) if (CONCAT2(mkldnn_, _fmt) == fmt) return STRINGIFY(_fmt)
    CASE(x);
    CASE(nc);
    CASE(nchw);
    CASE(ncdhw);
    CASE(nhwc);
    CASE(nChw8c);
    CASE(nChw16c);
    CASE(nCdhw16c);
    CASE(oidhw);
    CASE(oihw);
    CASE(hwio);
#undef CASE
    assert(!"unknown memory format");
    return "unknown memory format";
}

mkldnn_memory_format_t str2fmt(const char *str) {
#define CASE(_fmt) do { \
    if (!strcmp(STRINGIFY(_fmt), str) \
            || !strcmp("mkldnn_" STRINGIFY(_fmt), str)) \
        return CONCAT2(mkldnn_, _fmt); \
} while (0)
    CASE(x);
    CASE(nc);
    CASE(nchw);
    CASE(ncdhw);
    CASE(nhwc);
    CASE(nChw8c);
    CASE(nChw16c);
    CASE(nCdhw16c);
    CASE(oidhw);
    CASE(oihw);
    CASE(hwio);
#undef CASE
    assert(!"unknown memory format");
    return mkldnn_format_undef;
}
