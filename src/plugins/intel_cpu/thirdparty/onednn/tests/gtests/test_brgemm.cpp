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

#include "test_gemm_data_preparation.hpp"
#include "test_gemm_params.hpp"
#include "test_gemm_validation.hpp"
#include "gtest/gtest.h"

#include "dnnl_test_common.hpp"

#include "oneapi/dnnl/dnnl.hpp"
#include "tests/test_isa_common.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"

namespace dnnl {

struct brgemm_params_t : test_params {
    impl::data_type_t dt_a;
    impl::data_type_t dt_b;
    impl::cpu::x64::brgemm_batch_kind_t batch_kind;
    impl::cpu::x64::brgemm_layout_t layout;

    impl::cpu::x64::brgemm_attr_t attrs;

    int bs;
};

class params_creator_t {
public:
    std::vector<brgemm_params_t> create_simple_brgemm_params() {
        params = {};

        transpose_ = {'n'};
        sizes_and_leading_dims_[0]
                = {{1, 4}, {3, 3}, {3, 8}, {30, 30}, {64, 64}, {31, 61}};
        sizes_and_leading_dims_[1]
                = {{1, 4}, {2, 6}, {2, 5}, {20, 20}, {64, 64}, {21, 51}};
        sizes_and_leading_dims_[2]
                = {{1, 4}, {1, 3}, {1, 2}, {10, 20}, {64, 64}, {11, 81}};

        alpha_values_ = {1.0f, 2.0f, 2.5f};
        beta_values_ = {0.0f, 1.0f, 2.0f};

        amx_dts_ = {{dnnl_f32, dnnl_f32}};
        dts_ = {{dnnl_f32, dnnl_f32}};

        put_params();

        sizes_and_leading_dims_[0] = {{4, 4}, {8, 12}, {64, 1024}};
        sizes_and_leading_dims_[1] = {{4, 4}, {16, 32}, {128, 512}};
        sizes_and_leading_dims_[2] = {{4, 4}, {12, 56}, {16, 256}};

        amx_dts_ = {
                {dnnl_bf16, dnnl_bf16}, {dnnl_u8, dnnl_u8}, {dnnl_s8, dnnl_s8}};
        dts_ = {{dnnl_bf16, dnnl_bf16}, {dnnl_u8, dnnl_s8}};

        put_params();

        return params;
    }

private:
    void put_params() {
        for_(auto tr : transpose_)
        for_(size_t i = 0; i < sizes_and_leading_dims_[0].size(); i++)
        for_(auto alpha : alpha_values_)
        for_(auto beta : beta_values_)
        for (auto dt : is_amx ? amx_dts_ : dts_) {
            brgemm_params_t param = {};
            param.transA = tr;
            param.transB = 'n';
            param.M = sizes_and_leading_dims_[0][i].first;
            param.lda = sizes_and_leading_dims_[0][i].second;
            param.N = sizes_and_leading_dims_[1][i].first;
            param.ldb = sizes_and_leading_dims_[1][i].second;
            param.K = sizes_and_leading_dims_[2][i].first;
            param.ldc = sizes_and_leading_dims_[2][i].second;
            param.alpha = alpha;
            param.beta = beta;
            param.dt_a = dt.first;
            param.dt_b = dt.second;
            param.batch_kind = impl::cpu::x64::brgemm_addr;
            param.layout = impl::cpu::x64::brgemm_row_major;
            param.bs = 1;
            param.attrs.max_bs = 1;
            param.attrs.max_top_vpad = 0;
            param.attrs.max_bottom_vpad = 0;
            param.expect_to_fail = false;
            param.expected_status = dnnl_success;

            params.emplace_back(param);
        }
    }

    const bool is_amx = dnnl::mayiuse(cpu_isa::avx512_core_amx);

    std::vector<char> transpose_;
    std::vector<std::pair<int64_t, int64_t>> sizes_and_leading_dims_[3];
    std::vector<float> alpha_values_;
    std::vector<float> beta_values_;
    std::vector<std::pair<impl::data_type_t, impl::data_type_t>> amx_dts_;
    std::vector<std::pair<impl::data_type_t, impl::data_type_t>> dts_;

    std::vector<brgemm_params_t> params;
};

class brgemm_test_t : public ::testing::TestWithParam<brgemm_params_t> {
protected:
    void SetUp() override {
        const auto &p = GetParam();

        SKIP_IF(get_test_engine_kind() == engine::kind::gpu,
                "Brgemm is unimplemented on gpu.");

        SKIP_IF(!impl::cpu::platform::has_data_type_support(p.dt_a),
                "Engine does not support this data type.");

        catch_expected_failures(
                [=]() { Test(); }, p.expect_to_fail, p.expected_status, true);
    }

    void Test() {
        const auto &p = ::testing::TestWithParam<brgemm_params_t>::GetParam();
        run_proper_test(p);
    }

private:
    template <typename b_dt>
    void reorder_B(const brgemm_params_t &p, const mapped_ptr_t<b_dt> &b_mem,
            mapped_ptr_t<b_dt> &b_mem_reordered) const {
        static constexpr int k_pack = 4 / sizeof(b_dt);

        dnnl::impl::parallel_nd(p.K, p.N, [&](int64_t k, int64_t n) {
            size_t b_off = k * p.ldb + n;
            size_t b_reordered_off
                    = (k / k_pack) * p.ldb * k_pack + n * k_pack + k % k_pack;
            b_mem_reordered[b_reordered_off] = b_mem[b_off];
        });
    }

    template <typename b_dt>
    mapped_ptr_t<b_dt> get_B_mem(const brgemm_params_t &p) {
        mapped_ptr_t<b_dt> B = map_memory<b_dt>(*gemm_data_.b_mem);

        static constexpr int k_pack = 4 / sizeof(b_dt);
        if (k_pack > 1) {
            size_t sizeA, sizeB, sizeC;
            get_matrix_size(p, sizeA, sizeB, sizeC);

            b_mem_reordered_ = std::make_shared<test_memory>(
                    get_matrix_md<b_dt>(sizeB, p.off.b), get_test_engine());
            auto B_reordered = map_memory<b_dt>(*b_mem_reordered_);

            reorder_B(p, B, B_reordered);

            return B_reordered;
        }

        return B;
    }

    template <typename a_dt, typename b_dt, typename c_dt>
    dnnl_status_t run_brgemm(const brgemm_params_t &p) {
        using namespace dnnl::impl::cpu;
        using namespace dnnl::impl::cpu::x64;

        mapped_ptr_t<a_dt> A = map_memory<a_dt>(*gemm_data_.a_mem);
        mapped_ptr_t<b_dt> B = get_B_mem<b_dt>(p);
        mapped_ptr_t<c_dt> C = map_memory<c_dt>(*gemm_data_.c_mem);

        //initialize brgemm kernel
        char palette[64];
        char tile_buffer[1024];
        x64::brgemm_t desc;
        auto res = brgemm_desc_init(&desc, x64::cpu_isa_t::isa_any,
                p.batch_kind, p.dt_a, p.dt_b, p.tr_a(), p.tr_b(), p.layout,
                p.alpha, p.beta, p.lda, p.ldb, p.ldc, p.M, p.N, p.K);
        if (res != dnnl_success) return res;

        if (desc.is_amx) res = brgemm_init_tiles(desc, palette);
        if (!desc.is_amx) brgemm_desc_set_attr(&desc, p.attrs);

        if (res != dnnl_success) return res;

        x64::brgemm_kernel_t *_t_ptr;
        res = brgemm_kernel_create(&_t_ptr, desc);

        x64::brgemm_batch_element_t batch_element;
        batch_element.ptr.A = A;
        batch_element.ptr.B = B;
        batch_element.vvpad.top = 0;
        batch_element.vvpad.bottom = 0;
        if (desc.is_amx) amx_tile_configure(palette);
        brgemm_kernel_execute(_t_ptr, p.bs, &batch_element, C,
                desc.is_amx ? tile_buffer : nullptr);

        brgemm_kernel_destroy(_t_ptr);
        if (desc.is_amx) amx_tile_release();

        return res;
    }

    template <typename a_dt, typename b_dt, typename c_dt>
    void test_brgemm(const brgemm_params_t &p) {
        gemm_data_ = {};
        prepare_data_for_gemm_testing<a_dt, b_dt, c_dt>(p, gemm_data_);

        dnnl_status_t status = run_brgemm<a_dt, b_dt, c_dt>(p);

        if (status == dnnl_success) {
            validate<a_dt, b_dt, c_dt>(p, gemm_data_);
        }

        if (status != dnnl_success)
            throw error(status, "oneDNN brgemm returned error");
    }

    void run_proper_test(const brgemm_params_t &p) {
        using namespace impl::cpu::x64;

        if (dnnl::mayiuse(cpu_isa::avx512_core_amx)) {
            if (p.dt_a == dnnl_f32 && p.dt_b == dnnl_f32)
                test_brgemm<float, float, float>(p);
            else if (p.dt_a == dnnl_bf16 && p.dt_b == dnnl_bf16)
                test_brgemm<bfloat16_t, bfloat16_t, float>(p);
            else if (p.dt_a == dnnl_s8 && p.dt_b == dnnl_s8)
                test_brgemm<int8_t, int8_t, int32_t>(p);
            else if (p.dt_a == dnnl_u8 && p.dt_b == dnnl_u8)
                test_brgemm<uint8_t, uint8_t, int32_t>(p);
            else
                throw error(dnnl_unimplemented, "Brgemm unimplemented.");
        } else {
            if (p.dt_a == dnnl_f32 && p.dt_b == dnnl_f32)
                test_brgemm<float, float, float>(p);
            else if (p.dt_a == dnnl_bf16 && p.dt_b == dnnl_bf16)
                test_brgemm<bfloat16_t, bfloat16_t, float>(p);
            else if (p.dt_a == dnnl_u8 && p.dt_b == dnnl_s8) {
                assert(p.layout == brgemm_layout_t::brgemm_row_major);
                test_brgemm<uint8_t, int8_t, int32_t>(p);
            } else if (p.dt_a == dnnl_s8 && p.dt_b == dnnl_u8) {
                assert(p.layout == brgemm_layout_t::brgemm_col_major);
                test_brgemm<int8_t, uint8_t, int32_t>(p);
            } else
                throw error(dnnl_unimplemented, "Brgemm unimplemented.");
        }
    }

    test_gemm_data gemm_data_;
    std::shared_ptr<test_memory> b_mem_reordered_;
};

TEST_P(brgemm_test_t, TestsBRGEMM) {}
INSTANTIATE_TEST_SUITE_P(TestBRGEMMSimple, brgemm_test_t,
        ::testing::ValuesIn(params_creator_t().create_simple_brgemm_params()));

} // namespace dnnl
