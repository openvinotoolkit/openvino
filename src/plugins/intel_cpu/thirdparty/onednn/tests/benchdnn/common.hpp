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

#ifndef COMMON_HPP
#define COMMON_HPP

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <bitset>
#include <cinttypes>
#include <functional>
#include <string>
#include <vector>

#include "src/common/z_magic.hpp"

#include "utils/timer.hpp"

#define ABS(a) ((a) > 0 ? (a) : (-(a)))

#define MIN2(a, b) ((a) < (b) ? (a) : (b))
#define MAX2(a, b) ((a) > (b) ? (a) : (b))

#define MIN3(a, b, c) MIN2(a, MIN2(b, c))
#define MAX3(a, b, c) MAX2(a, MAX2(b, c))

#define IMPLICATION(cause, effect) (!(cause) || !!(effect))

#if defined(_WIN32) && !defined(__GNUC__)
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

#if defined(_MSC_VER) && !defined(__clang__) && !defined(__INTEL_COMPILER)
#define collapse(x)
#endif

#define IFMT "%" PRId64

#define OK 0
#define FAIL 1

enum { CRIT = 1, WARN = 2 };

#define SAFE(f, s) \
    do { \
        int status__ = (f); \
        if (status__ != OK) { \
            if (s == CRIT || s == WARN) { \
                fprintf(stderr, "@@@ error [%s:%d]: '%s' -> %d\n", \
                        __PRETTY_FUNCTION__, __LINE__, STRINGIFY(f), \
                        status__); \
                fflush(0); \
                if (s == CRIT) exit(1); \
            } \
            return status__; \
        } \
    } while (0)

#define SAFE_V(f) \
    do { \
        int status__ = (f); \
        if (status__ != OK) { \
            fprintf(stderr, "@@@ error [%s:%d]: '%s' -> %d\n", \
                    __PRETTY_FUNCTION__, __LINE__, STRINGIFY(f), status__); \
            fflush(0); \
            exit(1); \
        } \
    } while (0)

extern int verbose;
extern bool canonical;
extern bool mem_check;
extern bool attr_same_pd_check;
extern std::string skip_impl; /* empty or "" means skip nothing */

#define BENCHDNN_PRINT(v, fmt, ...) \
    do { \
        if (verbose >= v) { \
            printf(fmt, __VA_ARGS__); \
            /* printf("[%d][%s:%d]" fmt, v, __func__, __LINE__, __VA_ARGS__); */ \
            fflush(0); \
        } \
    } while (0)

#define BENCHDNN_DISALLOW_COPY_AND_ASSIGN(T) \
    T(const T &) = delete; \
    T &operator=(const T &) = delete;

using bench_mode_t = std::bitset<4>;
extern bench_mode_t RUN, CORR, PERF, LIST; // pre-defined modes
extern bench_mode_t bench_mode; // user mode
extern const char *driver_name;

bool is_bench_mode(bench_mode_t user_mode);

/* perf */
extern double max_ms_per_prb; /** maximum time spends per prb in ms */
extern int min_times_per_prb; /** minimal amount of runs per prb */
extern int fix_times_per_prb; /** if non-zero run prb that many times */

extern bool fast_ref_gpu;
extern bool allow_enum_tags_only;
extern int test_start;

/* global stats */
struct stat_t {
    int tests;
    int passed;
    int failed;
    int skipped;
    int mistrusted;
    int unimplemented;
    int listed;
    double ms[timer::timer_t::mode_t::n_modes];
};
extern stat_t benchdnn_stat;

/* result structure */
enum res_state_t {
    UNTESTED = 0,
    PASSED,
    SKIPPED,
    MISTRUSTED,
    UNIMPLEMENTED,
    FAILED,
    LISTED
};
const char *state2str(res_state_t state);

enum skip_reason_t {
    SKIP_UNKNOWN = 0,
    CASE_NOT_SUPPORTED,
    DATA_TYPE_NOT_SUPPORTED,
    INVALID_CASE,
    KNOWN_LIMITATION,
    NOT_ENOUGH_RAM,
    SKIP_IMPL_HIT,
    SKIP_START,
};
const char *skip_reason2str(skip_reason_t skip_reason);

struct res_t {
    res_state_t state;
    size_t errors, total;
    timer::timer_t timer;
    std::string impl_name;
    skip_reason_t reason;
    size_t ibytes, obytes;
};

void parse_result(
        res_t &res, bool &want_perf_report, int status, const char *pstr);

/* misc */
void init_fp_mode();

void *zmalloc(size_t size, size_t align);
void zfree(void *ptr);

bool str2bool(const char *str);
const char *bool2str(bool value);

/* TODO: why two functions??? */
bool match_regex(const char *str, const char *pattern);
bool maybe_skip(const std::string &impl_str);

typedef int (*bench_f)(int argc, char **argv);
int batch(const char *fname, bench_f bench);

/* returns 1 with given probability */
int flip_coin(ptrdiff_t seed, float probability);

int64_t div_up(const int64_t a, const int64_t b);
int64_t next_pow2(int64_t a);
int mxcsr_cvt(float f);

/* set '0' across *arr:+size */
void array_set(char *arr, size_t size);

/* wrapper to dnnl_sgemm
 * layout = 'F' - column major
 * layout = 'C' - row major*/
void gemm(const char *layout, const char *transa, const char *transb, int64_t m,
        int64_t n, int64_t k, const float alpha, const float *a,
        const int64_t lda, const float *b, const int64_t ldb, const float beta,
        float *c, const int64_t ldc);

int sanitize_desc(int &ndims, std::vector<std::reference_wrapper<int64_t>> d,
        std::vector<std::reference_wrapper<int64_t>> h,
        std::vector<std::reference_wrapper<int64_t>> w,
        const std::vector<int64_t> &def_values, bool must_have_spatial = false);

void print_dhw(bool &print_d, bool &print_h, bool &print_w, int ndims,
        const std::vector<int64_t> &d, const std::vector<int64_t> &h,
        const std::vector<int64_t> &w);

#endif
