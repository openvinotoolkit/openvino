/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "dnnl.h"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "tests/test_thread.hpp"

#include "zeropad/zeropad.hpp"

extern "C" {
dnnl_status_t dnnl_impl_zero_pad(
        const dnnl::impl::memory_t *memory, dnnl::impl::stream_t *stream);
}

namespace zeropad {

static int compare(const dnn_mem_t &test_mem, res_t *res) {
    const int ndims = test_mem.md_.ndims;
    const auto *dims = test_mem.md_.dims;

    if (ndims == 0) return OK;
    if (test_mem.md_.format_kind != dnnl_blocked) return OK;

    std::atomic<int> ok(true);

    const uint8_t *mem = (const uint8_t *)test_mem;
    size_t type_size = test_mem.sizeof_dt();

    const auto increment
            = [&](dnnl_dims_t &pos, dnnl_dim_t &idx, bool &done, int stop_dim) {
                  for (int i = ndims - 1; i >= stop_dim; i--) {
                      pos[i]++;
                      if (pos[i] < dims[i]) {
                          break;
                      } else {
                          pos[i] = 0;
                          if (i == stop_dim) done = true;
                      }
                  }
                  idx = md_off_v(test_mem.md_, pos);
              };

    dnnl::impl::parallel_nd(dims[0], [&](dnnl_dim_t dim0) {
        dnnl_dims_t pos = {0};
        pos[0] = dim0;
        dnnl_dim_t idx = md_off_v(test_mem.md_, pos);
        bool done = false;

        while (!done && ok) {
            for (size_t i = 0; i < type_size; i++) {
                uint8_t mem_value = mem[type_size * idx + i];
                if (mem_value != dnnl_mem_default_value) ok = false;
            }
            increment(pos, idx, done, 1);
        }
    });

    // Serially check for errors for data dumping purposes
    if (!ok) {
        int errors = 0;
        dnnl_dims_t pos = {0};
        dnnl_dim_t idx = md_off_v(test_mem.md_, pos);
        bool done = false;
        while (!done) {
            for (size_t i = 0; i < type_size; i++) {
                uint8_t mem_value = mem[type_size * idx + i];
                bool idx_ok = (mem_value == dnnl_mem_default_value);
                if (!idx_ok) errors++;
                const bool dump = (!idx_ok && (errors < 10 || verbose >= 10))
                        || (verbose >= 99);
                if (dump) {
                    BENCHDNN_PRINT(0,
                            "[%4ld][arg:%d]"
                            "[" IFMT "," IFMT "," IFMT "," IFMT "," IFMT
                            "," IFMT "] dt:% 9.6g \n",
                            (long)idx, test_mem.dt(), pos[0], pos[1], pos[2],
                            pos[3], pos[4], pos[5], test_mem.get_elem(idx));
                    break;
                }
            }
            increment(pos, idx, done, 0);
        }

        BENCHDNN_PRINT(0, "@@@ [arg:%d] check_non_zeroed_elements failed\n",
                test_mem.dt());
        res->errors += errors;
    }

    int errors = 0;
    auto status = check_zero_padding(test_mem, test_mem.dt(), &errors);
    res->errors += errors;

    bool passed = ok && (status == OK);
    res->state = passed ? PASSED : FAILED;
    return passed ? OK : FAIL;
}

static dnnl_status_t perf_func(
        const dnnl_stream_t &stream, const std::vector<dnnl_exec_arg_t> &args) {
    return dnnl_impl_zero_pad(args[0].memory, stream);
}

void check_known_skipped_case(const prb_t *prb, res_t *res) {
    check_known_skipped_case_common({prb->dt}, FWD_D, res);
    if (res->state == SKIPPED) return;

    if (is_nvidia_gpu()) {
        res->state = SKIPPED, res->reason = CASE_NOT_SUPPORTED;
        return;
    }
}

int doit(const prb_t *prb, res_t *res) {
    if (bench_mode == LIST) return res->state = LISTED, OK;

    check_known_skipped_case(prb, res);
    if (res->state == SKIPPED) return OK;

    dnnl_memory_desc_t data_md {};
    SAFE(init_md(&data_md, prb->ndims, prb->dims.data(), prb->dt, prb->tag),
            WARN);
    if (res->state == SKIPPED || res->state == UNIMPLEMENTED) return OK;

    if (check_mem_size(data_md) != OK) {
        return res->state = SKIPPED, res->reason = NOT_ENOUGH_RAM, OK;
    }

    const auto &test_engine = get_test_engine();

    dnn_mem_t test_mem(data_md, test_engine);

    args_t args;
    args.set(0, test_mem);
    perf_function_t perf_func_ = &perf_func;

    if (is_bench_mode(CORR)) {
        execute_and_wait(perf_func_, test_engine, args);
        SAFE(compare(test_mem, res), WARN);
    }
    if (is_bench_mode(PERF)) {
        // Get plain memory desc size to have a proper padded area size.
        dnnl_memory_desc_t plain_data_md {};
        SAFE(init_md(&plain_data_md, prb->ndims, prb->dims.data(), prb->dt,
                     tag::abx),
                WARN);
        // Fill output bytes for perf_report.
        res->ibytes = 0; // Since we don't read any data from padding.
        res->obytes = dnnl_memory_desc_get_size(&data_md)
                - dnnl_memory_desc_get_size(&plain_data_md);
    }

    measure_perf(res->timer, perf_func_, args);

    return OK;
}

} // namespace zeropad
