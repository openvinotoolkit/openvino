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

#include "mkldnn_thread.hpp"

#include "simple_concat.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace memory_tracking::names;

template <data_type_t data_type>
void simple_concat_t<data_type>::execute() const {
    auto scratchpad = this->scratchpad();
    auto iptrs = scratchpad.template get<const data_t *>(key_concat_iptrs);
    auto optrs = scratchpad.template get<data_t *>(key_concat_optrs);
    auto nelems_to_copy = scratchpad.template get<size_t>(key_concat_nelems);
    auto is = scratchpad.template get<strides_t>(key_concat_istrides);

    const int num_arrs = pd()->n_inputs();
    const ptrdiff_t *perm = pd()->perm_, *iperm = pd()->iperm_;
    const int concat_dim = pd()->concat_dim();
    auto o_base_ptr = reinterpret_cast<data_t *>(this->memory());

    for (int a = 0; a < num_arrs; ++a) {
        const memory_desc_wrapper i_d(pd()->src_pd(a));
        const memory_desc_wrapper o_d(pd()->src_image_pd(a));

        iptrs[a] = reinterpret_cast<const data_t *>(
                this->input_memory(a)) + i_d.blk_off(0);
        optrs[a] = o_base_ptr + o_d.blk_off(0);
        nelems_to_copy[a] = pd()->nelems_to_concat(i_d);
        for (int i = 0; i < TENSOR_MAX_DIMS; i++) {
            if (i < perm[concat_dim])
                is[a][i] = size_t(i_d.blocking_desc().strides[0][iperm[i]]);
            else
                is[a][i] = 0;
        }
    }

    const memory_desc_wrapper o_d(pd()->src_image_pd());
    auto &blk = o_d.blocking_desc();

    strides_t os = { 0 };
    for (int i = 0; i < perm[concat_dim]; i++)
        os[i] = o_d.blocking_desc().strides[0][iperm[i]];

    dims_t phys_dims;
    for (size_t i = 0; i < sizeof(phys_dims)/sizeof(phys_dims[0]); i++)
        phys_dims[i] = (i < (size_t)perm[concat_dim])
            ?  o_d.dims()[iperm[i]] / blk.block_dims[iperm[i]] : 1;

    if (perm[concat_dim] == 0) {
        for (int a = 0; a < num_arrs; ++a) {
            const data_t *i = &iptrs[a][0];
            data_t *o = &optrs[a][0];
            parallel_nd((ptrdiff_t)nelems_to_copy[a],
                    [&](ptrdiff_t e) { o[e] = i[e]; });
        }
    } else {
        parallel_nd(phys_dims[0], phys_dims[1], phys_dims[2], phys_dims[3],
            phys_dims[4], num_arrs,
            [&](int n0, int n1, int n2, int n3, int n4, int a) {
            // XXX: this code may access uninitialized values in is[*][0-4] --
            // that's why we have to set them to zero although this is
            // probably benign
            size_t in_off = is[a][0] * n0 + is[a][1] * n1 + is[a][2] * n2
                    + is[a][3] * n3 + is[a][4] * n4;
            size_t out_off = os[0] * n0 + os[1] * n1 + os[2] * n2
                    + os[3] * n3 + os[4] * n4;
            const data_t *i = &iptrs[a][in_off];
            data_t *o = &optrs[a][out_off];
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
            // The code below performs data copying: o[e] = i[e]
            // and uses a workaround to make GNU compilers optimize it
            uint8_t *ptro = reinterpret_cast<uint8_t *>(o);
            const uint8_t *ptri = reinterpret_cast<const uint8_t *>(i);
            const size_t main_part =
                (nelems_to_copy[a] * sizeof(data_t)) / sizeof(uint32_t);
            const size_t tail_part =
                (nelems_to_copy[a] * sizeof(data_t)) % sizeof(uint32_t);
            PRAGMA_OMP_SIMD()
            for (size_t e = 0; e < main_part; ++e) {
                *(reinterpret_cast<uint32_t *>(ptro))
                    = *(reinterpret_cast<const uint32_t *>(ptri));
                ptro += sizeof(uint32_t);
                ptri += sizeof(uint32_t);
            }
            for (size_t e = 0; e < tail_part; ++e) {
                *ptro = *ptri;
                ++ptro;
                ++ptri;
            }
#else
            PRAGMA_OMP_SIMD()
            for (size_t e = 0; e < nelems_to_copy[a]; ++e) o[e] = i[e];
#endif
        });
    }
}

template struct simple_concat_t<data_type::f32>;
template struct simple_concat_t<data_type::u8>;
template struct simple_concat_t<data_type::s8>;
template struct simple_concat_t<data_type::s32>;
template struct simple_concat_t<data_type::bf16>;

}
}
}
