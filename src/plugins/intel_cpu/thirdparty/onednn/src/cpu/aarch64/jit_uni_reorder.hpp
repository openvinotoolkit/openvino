/*******************************************************************************
* Copyright 2018-2020 Intel Corporation
* Copyright 2020 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_UNI_REORDER_HPP
#define CPU_AARCH64_JIT_UNI_REORDER_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"

#include "cpu/reorder/cpu_reorder_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace tr {

constexpr int max_ndims = DNNL_MAX_NDIMS;

struct node_t {
    size_t n;
    ptrdiff_t is; // input stride
    ptrdiff_t os; // output stride
    ptrdiff_t ss; // scale stride
};

enum class scale_type_t { NONE, COMMON, MANY };

struct prb_t {
    data_type_t itype;
    data_type_t otype;
    int ndims;
    node_t nodes[max_ndims];
    ptrdiff_t ioff;
    ptrdiff_t ooff;
    scale_type_t scale_type;
    float beta;
};

status_t prb_init(prb_t &prb, const memory_desc_t &imd,
        const memory_desc_t &omd, const primitive_attr_t *attr);

/** sorts the problem nodes so that output strides come in ascending order */
void prb_normalize(prb_t &p);

/** folds nodes together if possible */
void prb_simplify(prb_t &p);

/** splits the node dim into two of sizes n1 and n / n1
 * @warning n must be multiple of n1 */
void prb_node_split(prb_t &p, int dim, size_t n1);

/** swaps d0 and d1 nodes */
void prb_node_swap(prb_t &p, int d0, int d1);

/** moves node d0 to the d1 position.
 * nodes (d0, d1] are shifted to the left if d0 < d1 or
 * to the right if d0 > d1 */
void prb_node_move(prb_t &p, int d0, int d1);

/** dumps the problem to stdout */
void prb_dump(const prb_t &p);

struct call_param_t {
    const void *in;
    void *out;
    const float *scale;
};

struct kernel_t {
    struct desc_t {
        int id;
        prb_t prb;
    };

    kernel_t(const desc_t &desc) : desc_(desc) {}
    virtual void operator()(const call_param_t *c) const = 0;
    virtual status_t create_kernel() = 0;
    virtual ~kernel_t() {}

    /** inits kernel descriptor:
     *      desc            -- kernel descriptor (output)
     *      prb             -- transposition problem (input)
     *      ndims_ker_max   -- limit the maximum number of dimensions kernel
     *                         will process (optional, 0 -- no limitation) */
    static status_t desc_init(
            desc_t &desc, const prb_t &prb, int ndims_ker_max = 0);

    /** creates kernel for the problem described in desc */
    static kernel_t *create(const desc_t &desc);

protected:
    const desc_t desc_;
    const prb_t &prb_ = desc_.prb;
};

/* TODO: add trans_t class */

} // namespace tr

struct jit_uni_reorder_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("jit:uni", jit_uni_reorder_t);

        tr::prb_t prb_;
        tr::kernel_t::desc_t ker_desc_;
        int nthr_;

    private:
        static status_t create(reorder_pd_t **reorder_pd, engine_t *engine,
                const primitive_attr_t *attr, engine_t *src_engine,
                const memory_desc_t *src_md, engine_t *dst_engine,
                const memory_desc_t *dst_md);

        friend dnnl::impl::impl_list_item_t;
    };

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

    enum { ndims_driver_max = 4 };

private:
    void omp_driver_0d(
            int off, const char *in, char *out, const float *scale) const;
    void omp_driver_1d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const;
    void omp_driver_2d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const;
    void omp_driver_3d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const;
    void omp_driver_4d(int ithr, int nthr, int off, const char *in, char *out,
            const float *scale) const;

    void omp_driver(const char *in, char *out, const float *scale) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<tr::kernel_t> kernel_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
