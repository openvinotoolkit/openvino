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

#ifndef SIMPLE_CONCAT_HPP
#define SIMPLE_CONCAT_HPP

#include "cpu_concat.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t data_type>
struct simple_concat_t: public cpu_primitive_t {
    using cpu_memory_pd_t = cpu_memory_t::pd_t;

    struct pd_t: public cpu_concat_pd_t {
        pd_t(const memory_desc_t *output_d, int n,
                int concat_dim, const cpu_memory_pd_t **input_pds,
                const primitive_attr_t *attr)
            : cpu_concat_pd_t(output_d, n, concat_dim, input_pds, attr)
        {}
        pd_t(const pd_t &rhs) : cpu_concat_pd_t(rhs) {
            for (size_t i = 0; i < sizeof(perm_)/sizeof(perm_[0]); i++) {
                perm_[i] = rhs.perm_[i];
                iperm_[i] = rhs.iperm_[i];
            }
        }
        DECLARE_CPU_CONCAT_PD_T("simple:any", simple_concat_t);

        virtual status_t init() override {
            auto is_dense = [&](const memory_desc_wrapper &data_d) {
                return nelems_to_concat(concat_dim_, perm_, iperm_, data_d)
                        == _size_to_concat(concat_dim_, perm_, iperm_, data_d);
            };
            const memory_desc_wrapper dst_d(&dst_pd_);
            bool ok = true
                && cpu_concat_pd_t::init() == success
                && dst_d.ndims() <= 6;

            if (!ok) return unimplemented;

            for (size_t i = 0; i < src_pds_.size(); ++i) {
                const memory_desc_wrapper i_d(&src_pds_[i]);
                const memory_desc_wrapper o_d(&src_image_pds_[i]);
                ok = ok
                    && utils::everyone_is(data_type, i_d.data_type(),
                            o_d.data_type())
                    && i_d.format() == o_d.format()
                    && !utils::one_of(i_d.format(), memory_format::blocked,
                        memory_format::wino_fmt)
                    && !i_d.is_additional_buffer();
            }

            if (!ok)
                return unimplemented;

            format_perm(dst_d.ndims(), dst_d.blocking_desc().strides[0], perm_,
                    iperm_);

            for (size_t i = 0; i < src_pds_.size(); ++i) {
                const memory_desc_wrapper i_d(&src_pds_[i]);
                const memory_desc_wrapper o_d(&src_image_pds_[i]);
                ok = ok && is_dense(i_d) && is_dense(o_d);
            }

            return ok ? success : unimplemented;
        }
        dims_t perm_;
        dims_t iperm_;
    };

    simple_concat_t(const pd_t *conf, const input_vector &inputs,
            const output_vector &outputs)
        : cpu_primitive_t(&conf_, inputs, outputs), conf_(*conf)
    {
        const int n = conf_.n_inputs();
        input_ptrs_ = (decltype(input_ptrs_))malloc(
                sizeof(*input_ptrs_) * n, 64);
        output_ptrs_ = (decltype(output_ptrs_))malloc(
                sizeof(*output_ptrs_) * n, 64);
        nelems_to_copy_ = (decltype(nelems_to_copy_))malloc(
                sizeof(*nelems_to_copy_) * n, 64);
        is_ = (decltype(is_))malloc(sizeof(*is_) * n, 64);
    }

    ~simple_concat_t() {
        free(input_ptrs_);
        free(output_ptrs_);
        free(nelems_to_copy_);
        free(is_);
    }

    virtual void execute(event_t *e) {
        execute();
        e->set_state(event_t::ready);
    }

    typedef typename prec_traits<data_type>::type data_t;

private:
    static void format_perm(
            const int ndims, const stride_t *strides, int *perm, int *iperm) {
        assert(ndims >= 0);
        bool swapped;
        strides_t strides_tmp;
        utils::array_copy(strides_tmp, strides, ndims);
        for (int i = 0; i < ndims; i++)
            iperm[i] = i;
        for (int i = 0; i < ndims - 1; i++) {
            swapped = false;
            for (int j = 0; j < ndims - i - 1; j++) {
                if (strides_tmp[j] < strides_tmp[j + 1]) {
                    nstl::swap(strides_tmp[j], strides_tmp[j + 1]);
                    nstl::swap(iperm[j], iperm[j + 1]);
                    swapped = true;
                }
            }
            if (swapped == false)
                break;
        }
        for (int i = 0; i < ndims; i++)
            perm[iperm[i]] = i;
    }

    static size_t nelems_to_concat(const int concat_dim, int *perm, int *iperm,
            const memory_desc_wrapper &data_d) {
        const int ndims = data_d.ndims();
        auto &blk = data_d.blocking_desc();
        int nelems = 1;
        for (int i = perm[concat_dim]; i < ndims; i++) {
            nelems *= data_d.dims()[iperm[i]] / blk.block_dims[iperm[i]];
        }
        for (int i = 0; i < ndims; i++) {
            nelems *= blk.block_dims[i];
        }
        return nelems;
    }

    static size_t _size_to_concat(const int concat_dim, int *perm, int *iperm,
            const memory_desc_wrapper &data_d) {
        size_t max_size = 0;
        auto &blk = data_d.blocking_desc();
        for (int d = perm[concat_dim]; d < data_d.ndims(); ++d) {
            auto block = blk.block_dims[iperm[d]];
            max_size = nstl::max(max_size,
                    size_t(blk.padding_dims[iperm[d]] / block)
                            * blk.strides[0][iperm[d]]);
            if (block > 1)
                max_size = nstl::max(max_size,
                        size_t(block * blk.strides[1][iperm[d]]));
        }
        return max_size;
    }

    void execute();
    pd_t conf_;

    const data_t **input_ptrs_ = nullptr;
    data_t **output_ptrs_ = nullptr;
    size_t *nelems_to_copy_ = nullptr;
    strides_t *is_ = nullptr;
};

}
}
}

#endif
