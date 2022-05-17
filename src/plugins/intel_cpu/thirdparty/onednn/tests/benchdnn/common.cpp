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

#include <assert.h>
#include <limits.h>
#include <stdint.h>

#include <cctype>
#include <fstream>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "dnnl.h"

#include "common.hpp"
#include "utils/parser.hpp"

#include "tests/test_thread.hpp"

// BENCHDNN_MEMORY_CHECK macro enables guarding mechanism for memory allocation:
// memory block is allocated on a page boundary and the page after the block is
// protected to catch possible invalid accesses.
//
// Note that the macro affects the correctness mode only.
#ifdef __unix__
#define BENCHDNN_MEMORY_CHECK
#endif

#ifdef BENCHDNN_MEMORY_CHECK
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#endif

bench_mode_t RUN {0x1}; // Run mode.
bench_mode_t CORR {0x2}; // Correctness mode. The default one.
bench_mode_t PERF {0x4}; // Performance mode. May be combined with CORR.
bench_mode_t LIST {0x8}; // Listing mode. Standalone mode to only create prb.

bool is_bench_mode(bench_mode_t user_mode) {
    return !(bench_mode & user_mode).none();
}

/* result structure */
const char *state2str(res_state_t state) {
    if (state == UNTESTED) return "UNTESTED_FAILED"; // for easier fail search

#define CASE(x) \
    if (state == (x)) return STRINGIFY(x)
    CASE(PASSED);
    CASE(SKIPPED);
    CASE(MISTRUSTED);
    CASE(UNIMPLEMENTED);
    CASE(FAILED);
    CASE(LISTED);
#undef CASE
    assert(!"unknown res state");
    return "STATE_UNDEF";
}

const char *skip_reason2str(skip_reason_t skip_reason) {
#define CASE(x) \
    if (skip_reason == (x)) return STRINGIFY(x)
    CASE(CASE_NOT_SUPPORTED);
    CASE(DATA_TYPE_NOT_SUPPORTED);
    CASE(INVALID_CASE);
    CASE(KNOWN_LIMITATION);
    CASE(NOT_ENOUGH_RAM);
    CASE(SKIP_IMPL_HIT);
    CASE(SKIP_START);
#undef CASE
    return "SKIP_UNKNOWN";
}

void parse_result(
        res_t &res, bool &want_perf_report, int status, const char *pstr) {
    auto &bs = benchdnn_stat;
    const char *state = state2str(res.state);

    switch (res.state) {
        case UNTESTED:
            if (is_bench_mode(PERF)) {
                want_perf_report = true;
                if (status == FAIL)
                    bs.failed++;
                else
                    bs.passed++;
                break;
            } else if (is_bench_mode(RUN)) {
                assert(status == OK);
                BENCHDNN_PRINT(0, "%d:EXECUTED __REPRO: %s\n", bs.tests, pstr);
                want_perf_report = false;
                break;
            }
        case FAILED:
            assert(status == FAIL);
            bs.failed++;
            BENCHDNN_PRINT(0, "%d:%s (errors:%lu total:%lu) __REPRO: %s\n",
                    bs.tests, state, (unsigned long)res.errors,
                    (unsigned long)res.total, pstr);
            break;
        case SKIPPED:
            assert(status == OK);
            BENCHDNN_PRINT(0, "%d:%s (%s) __REPRO: %s\n", bs.tests, state,
                    skip_reason2str(res.reason), pstr);
            bs.skipped++;
            break;
        case UNIMPLEMENTED:
            assert(status == OK);
            BENCHDNN_PRINT(0, "%d:%s __REPRO: %s\n", bs.tests, state, pstr);
            bs.unimplemented++;
            bs.failed++;
            break;
        case MISTRUSTED:
            assert(status == OK);
            bs.mistrusted++;
            BENCHDNN_PRINT(0, "%d:%s __REPRO: %s\n", bs.tests, state, pstr);
            // bs.failed++; /* temporal workaround for some tests */
            break;
        case PASSED:
            assert(status == OK);
            BENCHDNN_PRINT(0, "%d:%s __REPRO: %s\n", bs.tests, state, pstr);
            want_perf_report = true;
            bs.passed++;
            break;
        case LISTED:
            assert(status == OK);
            BENCHDNN_PRINT(0, "%d:%s __REPRO: %s\n", bs.tests, state, pstr);
            want_perf_report = false;
            bs.listed++;
            break;
        default: assert(!"unknown state"); SAFE_V(FAIL);
    }

    if (is_bench_mode(PERF)) {
        using bt = timer::timer_t;
        for (int mode = 0; mode < (int)bt::n_modes; ++mode)
            bs.ms[mode] += res.timer.ms((bt::mode_t)mode);
    }
}

/* misc */

#ifdef BENCHDNN_MEMORY_CHECK
static void *zmalloc_protect(size_t size) {
    const size_t page_sz = getpagesize();

    const size_t block_sz = size + 3 * sizeof(void *);
    const size_t total_sz = div_up(block_sz, page_sz) * page_sz + page_sz;

    void *mem_ptr;
    int rc = ::posix_memalign(&mem_ptr, page_sz, total_sz);
    if (rc != 0) return nullptr;

    uint8_t *ptr_start = (uint8_t *)mem_ptr;
    uint8_t *ptr = ptr_start + total_sz - page_sz - size;

    // Aligned on a page boundary
    void *ptr_protect = ptr + size;

    // Layout of the allocated region:
    // ptr_start   <- start of the allocated region
    // ptr[-16]    <- stores start address: ptr_start
    // ptr[-8]     <- stores protected address: ptr_protect
    // ptr         <- pointer to be returned from the function
    // ptr_protect <- pointer to the block to protect

    // Protect one page right after the block of size bytes
    int err = mprotect(ptr_protect, page_sz, PROT_NONE);
    if (err != 0) {
        ::free(ptr_start);
        return nullptr;
    }

    // Align down `ptr` on 8 bytes before storing addresses to make behavior
    // defined.
    ptrdiff_t to_align = reinterpret_cast<ptrdiff_t>(ptr) % sizeof(void *);
    void *ptr_aligned_8 = ptr - to_align;
    // Save pointers for zfree_protect
    ((void **)ptr_aligned_8)[-2] = ptr_start;
    ((void **)ptr_aligned_8)[-1] = ptr_protect;

    return ptr;
}

static void zfree_protect(void *ptr) {
    // Get aligned ptr before obtaining addresses
    ptrdiff_t to_align = reinterpret_cast<ptrdiff_t>(ptr) % sizeof(void *);
    void *ptr_aligned_8 = reinterpret_cast<uint8_t *>(ptr) - to_align;

    // Restore read-write access for the protected region
    void *ptr_protect = ((void **)ptr_aligned_8)[-1];
    const size_t page_sz = getpagesize();
    mprotect(ptr_protect, page_sz, PROT_READ | PROT_WRITE);

    // Deallocate the whole region
    void *ptr_start = ((void **)ptr_aligned_8)[-2];
    ::free(ptr_start);
}
#endif

void *zmalloc(size_t size, size_t align) {
#ifdef BENCHDNN_MEMORY_CHECK
    if (is_bench_mode(CORR)) { return zmalloc_protect(size); }
#endif

    void *ptr;
#ifdef _WIN32
    ptr = _aligned_malloc(size, align);
    int rc = ((ptr) ? 0 : errno);
#else
    // posix_memalign requires alignment to be
    // a power of 2 and a multiple of sizeof(void *)
    if (align < sizeof(void *)) align = sizeof(void *);
    assert(((align & (align - 1)) == 0) && "align must be a power of 2");

    // TODO. Heuristics: Increasing the size to alignment increases
    // the stability of performance results.
    if (is_bench_mode(PERF) && (size < align)) size = align;
    int rc = ::posix_memalign(&ptr, align, size);
#endif /* _WIN32 */
    return rc == 0 ? ptr : nullptr;
}

void zfree(void *ptr) {
#ifdef BENCHDNN_MEMORY_CHECK
    if (is_bench_mode(CORR)) {
        zfree_protect(ptr);
        return;
    }
#endif

#ifdef _WIN32
    _aligned_free(ptr);
#else
    return ::free(ptr);
#endif /* _WIN32 */
}

bool str2bool(const char *str) {
    return !strcasecmp("true", str) || !strcasecmp("1", str);
}

const char *bool2str(bool value) {
    return value ? "true" : "false";
}

#ifdef _WIN32
/* NOTE: this should be supported on linux as well, but currently
 * having issues for ICC170 and Clang*/
#include <regex>

bool match_regex(const char *str, const char *pattern) {
    std::regex re(pattern);
    return std::regex_search(str, re);
}
#else
#include <regex.h>
#include <sys/types.h>

bool match_regex(const char *str, const char *pattern) {
    static regex_t regex;
    static const char *prev_pattern = nullptr;
    if (pattern != prev_pattern) {
        if (prev_pattern) regfree(&regex);

        if (regcomp(&regex, pattern, 0)) {
            fprintf(stderr, "could not create regex\n");
            return true;
        }

        prev_pattern = pattern;
    }

    return !regexec(&regex, str, 0, nullptr, 0);
}
#endif /* _WIN32 */

bool maybe_skip(const std::string &impl_str) {
    if (skip_impl.empty()) return false;

    size_t start_pos = 0;
    // Iterate over impls in skip list.
    while (start_pos != std::string::npos) {
        const auto skip_impl_item
                = parser::get_substr(skip_impl, start_pos, ',');
        if (skip_impl_item.empty()) continue;
        if (impl_str.find(skip_impl_item) != std::string::npos) return true;
    }

    return false;
}

#if defined(_WIN32) && !defined(__GNUC__)
#include <windows.h>
#define PATH_MAX MAX_PATH
static char *dirname(char *path) {
    char drive[_MAX_DRIVE];
    char dir[_MAX_DIR];
    SAFE_V(_splitpath_s(path, drive, sizeof(drive), dir, sizeof(dir), NULL, 0,
                   NULL, 0) == 0
                    ? OK
                    : FAIL);
    path[0] = '\0';
    SAFE_V(strncat_s(path, PATH_MAX, drive, _MAX_DRIVE) == 0 ? OK : FAIL);
    SAFE_V(strncat_s(path, PATH_MAX, dir, _MAX_DIR) == 0 ? OK : FAIL);
    if (path[0] == '\0') {
        path[0] = '.';
        path[1] = '\0';
    }
    return path;
}

int readlink(const char *path, char *buf, size_t buf_max) {
    (void)path;
    // NULL means take the path of myself
    return GetModuleFileName(NULL, buf, (DWORD)buf_max);
}
#else
#include <libgen.h>
#include <unistd.h>
#endif /* _WIN32 */

std::string locate_batch_file(const std::string &fname) {
    SAFE_V(fname.length() < PATH_MAX ? OK : FAIL);

    const int max_paths = 4;

    static int n_paths = 0;
    static std::string search_paths[max_paths];

    std::string fdir;
    {
        std::string fname_copy = fname;
        fname_copy.resize(PATH_MAX);
        char *c_fdir = dirname(&fname_copy[0]);
        fdir = std::string(c_fdir);
    }

    bool dir_found = false;
    for (int n = 0; n_paths < max_paths && n < n_paths; ++n)
        if (search_paths[n].find(fdir) == 0) {
            dir_found = true;
            break;
        }
    if (!dir_found) {
        SAFE_V(n_paths < max_paths ? OK : FAIL);
        search_paths[n_paths++] = std::move(fdir);
    }

    std::ifstream ifs(fname);
    if (ifs.is_open()) return fname;

    for (int n = 0; n < n_paths; ++n) {
        std::string fullname = search_paths[n] + "/" + fname;
        ifs.open(fullname);
        if (ifs.is_open()) {
            BENCHDNN_PRINT(50, "batch file used: %s\n", fullname.c_str());
            ifs.close();
            return fullname;
        }
        ifs.close();
    }

    // Search in default inputs directory
    // Takes dirname(executable)/inputs/file_name on Linux
    // Takes dirname(executable)/../inputs/file_name on Windows
    fdir.resize(PATH_MAX);
    int length = readlink("/proc/self/exe", &fdir[0], PATH_MAX);
    if (length) {
        std::string s_fdir = std::string(dirname(&fdir[0]));
        for (int i_try = 0; i_try < 2; ++i_try) {
            fdir = s_fdir + std::string("/inputs/") + std::string(driver_name);
            if (i_try == 1) { // Windows has different folder structure
                fdir = s_fdir + std::string("/../inputs/")
                        + std::string(driver_name);
            }
            // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
            std::string fullname = fdir + "/" + fname;
            ifs.open(fullname);
            if (ifs.is_open()) {
                search_paths[n_paths++] = std::move(fdir);
                BENCHDNN_PRINT(50, "batch file used: %s\n", fullname.c_str());
                ifs.close();
                return fullname;
            }
            ifs.close();
        }
    }

    fprintf(stderr, "cannot open file %s\n", fname.c_str());
    return fname;
}

int batch(const char *fname, bench_f bench) {
    std::ifstream ifs(locate_batch_file(std::string(fname)));
    SAFE(ifs.is_open() ? OK : FAIL, CRIT);

    std::vector<std::string> opts;
    std::string str;
    bool continued_line = false;
    while (ifs >> str) {
        if (str.length() == 0) continue;

        // shell style comments
        if (str.front() == '#') {
            std::getline(ifs, str); // take whole commented line out
            continue;
        }

        // shell style line break
        if (continued_line) {
            // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
            str = opts.back() + str; // update current line with previous
            opts.pop_back(); // take previous line out
        }

        if (str.back() == '\\') {
            continued_line = true;
            if (str.length() == 1) continue; // line break lives separately
            str.erase(str.size() - 1); // otherwise remove it
        } else {
            continued_line = false;
        }

        opts.push_back(std::move(str));
    }

    std::vector<char *> c_opts;
    c_opts.reserve(opts.size());
    for (const auto &opt : opts)
        c_opts.push_back(const_cast<char *>(opt.c_str()));

    return bench(static_cast<int>(c_opts.size()), c_opts.data());
}

int flip_coin(ptrdiff_t seed, float probability) {
    const ptrdiff_t big_prime = 1000003;
    const ptrdiff_t prime = 753737;
    seed *= prime;
    return (seed % big_prime) < (probability * big_prime);
}

int64_t div_up(const int64_t a, const int64_t b) {
    SAFE_V(b != 0 ? OK : FAIL);
    return (a + b - 1) / b;
}

int64_t next_pow2(int64_t a) {
    assert(a > 0 && a <= ((int64_t)1 << 62));
    if (a > 1) a--;
    while (a & (a - 1))
        a &= (a - 1);
    return a << 1;
}

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <xmmintrin.h>

int mxcsr_cvt(float f) {
    return _mm_cvtss_si32(_mm_load_ss(&f));
}
void init_fp_mode() {
    // We set ftz to avoid denormals in perf measurements
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
}
#else
int mxcsr_cvt(float f) {
    return (int)nearbyintf(f);
}
void init_fp_mode() {}
#endif

void array_set(char *arr, size_t size) {
    for (size_t i = 0; i < size; ++i)
        arr[i] = 0;
}

void gemm(const char *layout, const char *transa, const char *transb, int64_t m,
        int64_t n, int64_t k, const float alpha, const float *a,
        const int64_t lda, const float *b, const int64_t ldb, const float beta,
        float *c, const int64_t ldc) {
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    if (*layout == 'C') {
        dnnl_sgemm(
                *transa, *transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
        dnnl_sgemm(
                *transb, *transa, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
    }
#else
    if (std::toupper(*layout) != 'C') {
        gemm("C", transb, transa, n, m, k, alpha, b, ldb, a, lda, beta, c, ldc);
        return;
    }

    auto a_accessor = [&](int64_t i, int64_t j) {
        if (std::toupper(*transa) == 'N') return a[i * lda + j];
        return a[j * lda + i];
    };

    auto b_accessor = [&](int64_t i, int64_t j) {
        if (std::toupper(*transb) == 'N') return b[i * ldb + j];
        return b[j * ldb + i];
    };

    dnnl::impl::parallel_nd(m, n, [&](int64_t i, int64_t j) {
        float ab = 0.0f;
        for (int64_t _k = 0; _k < k; ++_k)
            ab += a_accessor(i, _k) * b_accessor(_k, j);
        float cij = (beta == 0) ? 0.0f : beta * c[i * ldc + j];
        c[i * ldc + j] = alpha * ab + cij;
    });
#endif
}

int sanitize_desc(int &ndims, std::vector<std::reference_wrapper<int64_t>> d,
        std::vector<std::reference_wrapper<int64_t>> h,
        std::vector<std::reference_wrapper<int64_t>> w,
        const std::vector<int64_t> &def_values, bool must_have_spatial) {
    size_t N = d.size();
    assert(h.size() == N && w.size() == N && def_values.size() == N);

    ndims = 5;

    // check output spatial values
    const bool no_d = d[0].get() == 0;
    const bool no_h = h[0].get() == 0;
    const bool no_w = w[0].get() == 0;

    if (no_d) ndims--;
    if (no_d && no_h) ndims--;
    if (no_d && no_h && no_w) ndims--;
    if (must_have_spatial && ndims <= 2) return FAIL;

    if (ndims == 5) {
        if (no_h && no_w) {
            // User specified values for the d dimension but not values for h
            // and w dimensions. Propagate d values to h and w dimensions.
            for (size_t n = 0; n < N; ++n)
                w[n].get() = h[n].get() = d[n].get();
        } else if (!no_h && !no_w) {
            // User specified them all, good to go.
        } else {
            // Problem is not cubic and one of h or w dimension is missing.
            return FAIL;
        }
    } else if (ndims == 4 && no_w) {
        // User specified values for the h dimension but not values for the w
        // dimension. Propagate h values to the w dimension.
        for (size_t n = 0; n < N; ++n)
            w[n].get() = h[n].get();
    }

    for (size_t n = 0; n < N; ++n) {
        if (ndims < 5) d[n].get() = def_values[n];
        if (ndims < 4) h[n].get() = def_values[n];
        if (ndims < 3) w[n].get() = def_values[n];
    }

    return OK;
}

void print_dhw(bool &print_d, bool &print_h, bool &print_w, int ndims,
        const std::vector<int64_t> &d, const std::vector<int64_t> &h,
        const std::vector<int64_t> &w) {
    size_t N = d.size();
    assert(h.size() == N && w.size() == N);

    bool square_shape = true, cubic_shape = true;
    for (size_t n = 0; n < N; ++n) {
        square_shape = square_shape && h[n] == w[n];
        cubic_shape = cubic_shape && d[n] == h[n] && h[n] == w[n];
    }

    print_d = ndims == 5;
    print_h = ndims == 4 || (ndims == 5 && (!cubic_shape || canonical));
    print_w = ndims == 3 || (ndims == 5 && (!cubic_shape || canonical))
            || (ndims == 4 && (!square_shape || canonical));
}
