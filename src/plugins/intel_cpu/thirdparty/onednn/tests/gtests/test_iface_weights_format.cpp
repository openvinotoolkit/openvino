/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

#include <string>
#include <vector>

namespace dnnl {

// short names for brevity
using data_type = memory::data_type;
using tag = memory::format_tag;

class weights_format_test_t : public ::testing::Test {
protected:
    const engine eng = get_test_engine();

    struct inner_product_shape_t {
    public:
        inner_product_shape_t(memory::dim mb, memory::dim ic, memory::dim oc,
                memory::dim kw = 1, memory::dim kh = 1, memory::dim kd = 1)
            : mb_(mb), ic_(ic), oc_(oc), kw_(kw), kh_(kh), kd_(kd) {}

        memory::dims src_dims() const { return maybe_spatial(mb_, ic_); }
        memory::dims wei_dims() const { return maybe_spatial(oc_, ic_); }

        memory::dims dst_dims() const { return {mb_, oc_}; }
        memory::dims bia_dims() const { return {oc_}; }

    private:
        memory::dim mb_, ic_, oc_, kw_, kh_, kd_;
        bool is_1d() const { return kd_ == 1 && kh_ == 1 && kw_ != 1; }
        bool is_2d() const { return kd_ == 1 && kh_ != 1; }
        bool is_3d() const { return kd_ != 1; }

        memory::dims maybe_spatial(
                memory::dim param1, memory::dim param2) const {
            if (is_3d())
                return {param1, param2, kd_, kh_, kw_};
            else if (is_2d())
                return {param1, param2, kh_, kw_};
            else if (is_1d())
                return {param1, param2, kw_};
            else
                return {param1, param2};
        }
    };

    // Iterate primitive descriptor iterator till either of the following
    //   - brgemm kernel implementation is found
    //   - end of the primitive descriptor iterator is reached
    // return `true` iff brgemm kernel is found
    template <typename PD>
    bool seek_brgemm_impl(PD &pd) {
        const std::string brgemm("brgemm");
        std::string impl_info;
        bool brgemm_ker_found = false, seek_next_impl = true;
        do {
            std::string impl_info(pd.impl_info_str());
            brgemm_ker_found = impl_info.find(brgemm) != std::string::npos;

            seek_next_impl = !brgemm_ker_found && pd.next_impl();
        } while (seek_next_impl);

        return brgemm_ker_found;
    }

    std::vector<inner_product_shape_t> inner_product_shapes;
    std::vector<data_type> inner_product_data_types;

    void SetUp() override {
        for (auto dt : {data_type::f32, data_type::bf16}) {
            if (!unsupported_data_type(dt))
                inner_product_data_types.push_back(dt);
        }

        // inner product shapes of zero dimension [majority case]
        // dims format: {mb, ic, oc}
        inner_product_shapes.insert(inner_product_shapes.end(),
                {{2, 16, 16}, {2, 16, 32}, {2, 16, 64}, {2, 32, 16},
                        {2, 32, 32}, {2, 32, 64}, {2, 64, 16}, {2, 64, 32},
                        {2, 64, 64}, {2, 512, 16}, {2, 512, 32}, {2, 512, 64},
                        {2, 512, 512}, {2, 512, 1024}, {2, 1024, 512}});

        // inner product zero dimension shapes with channel tails
        for (auto sz : {1, 3, 15, 17, 31, 33, 63, 65, 127, 129})
            inner_product_shapes.emplace_back(
                    inner_product_shape_t {sz, sz, sz});

        // inner product zero dimensional regression shapes
        inner_product_shapes.emplace_back(
                inner_product_shape_t {2, 1024, 30522});

        // inner product shapes of higher dimensions
        // dims format: either of {mb, ic, oc, kw}, {mb, ic, oc, kw, kh},
        // or {mb, ic, oc, kw, kh, kd}
        inner_product_shapes.insert(inner_product_shapes.end(),
                {{2, 16, 16, 2}, {2, 16, 32, 2, 3}, {2, 16, 64, 4, 3, 2},
                        {2, 32, 16, 2}, {2, 32, 32, 2, 3}, {2, 32, 64, 4, 3, 2},
                        {2, 64, 16, 2}, {2, 64, 32, 2, 3},
                        {2, 64, 64, 4, 3, 2}});
    }
};

// Check for weights consistency in inner product, that is weights are same
// across forward and backward pass
// TODO: Enable similar tests for convolution once brgemm kernel's support
// is complete
HANDLE_EXCEPTIONS_FOR_TEST_F(weights_format_test_t, InnerProductWeightsCheck) {
    const bool do_skip = !DNNL_X64 || (DNNL_CPU_RUNTIME == DNNL_RUNTIME_NONE)
            || (get_test_engine_kind() != engine::kind::cpu);
    SKIP_IF(do_skip,
            "Inner Product weight check is applicable only for x64 CPU");

    for_(const auto &input_shape : inner_product_shapes)
    for (const auto &input_dt : inner_product_data_types) {
        // Note: For inner product with mixed data types, e.g. with bf16 src
        // and f32 dst, we do not require weights consistency.
        memory::desc src_md {input_shape.src_dims(), input_dt, tag::any};
        memory::desc wei_md {input_shape.wei_dims(), input_dt, tag::any};
        memory::desc bia_md {input_shape.bia_dims(), input_dt, tag::any};
        memory::desc dst_md {input_shape.dst_dims(), input_dt, tag::any};

        auto fwd_desc = inner_product_forward::desc(
                prop_kind::forward_training, src_md, wei_md, bia_md, dst_md);
        auto bwdd_desc
                = inner_product_backward_data::desc(src_md, wei_md, dst_md);
        auto bwdw_desc = inner_product_backward_weights::desc(
                src_md, wei_md, bia_md, dst_md);

        auto fwd_pd = inner_product_forward::primitive_desc(fwd_desc, eng);
        auto bwdd_pd = inner_product_backward_data::primitive_desc(
                bwdd_desc, eng, fwd_pd);
        auto bwdw_pd = inner_product_backward_weights::primitive_desc(
                bwdw_desc, eng, fwd_pd);

        bool fwd_brgemm_ker_found = false, bwdd_brgemm_ker_found = false,
             bwdw_brgemm_ker_found = false;
        // Currently only brgemm kernel supports same weight tags
        // for forward and backward data/weight inner product, therefore
        // skip if the forward impl kernel is not brgemm
        ASSERT_NO_THROW(fwd_brgemm_ker_found = seek_brgemm_impl(fwd_pd));
        if (!fwd_brgemm_ker_found) continue;

        // If the forward inner product can be handled by brgemm then so
        // should be the backward data/weights one
        ASSERT_NO_THROW(bwdd_brgemm_ker_found = seek_brgemm_impl(bwdd_pd));
        ASSERT_NO_THROW(bwdw_brgemm_ker_found = seek_brgemm_impl(bwdw_pd));

        ASSERT_TRUE(bwdd_brgemm_ker_found);
        ASSERT_TRUE(bwdw_brgemm_ker_found);

        // Check for weights consistency
        const auto &fwd_wei
                = fwd_pd.query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS);
        const auto &bwdd_wei
                = bwdd_pd.query_md(query::exec_arg_md, DNNL_ARG_WEIGHTS);
        const auto &bwdw_wei
                = bwdw_pd.query_md(query::exec_arg_md, DNNL_ARG_DIFF_WEIGHTS);

        ASSERT_TRUE(fwd_wei == bwdd_wei && fwd_wei == bwdw_wei);
    }
}

} // namespace dnnl
