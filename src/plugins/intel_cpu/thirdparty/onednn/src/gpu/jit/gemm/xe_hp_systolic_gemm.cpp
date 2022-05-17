/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include "gpu/jit/gemm/xe_hp_systolic_gemm.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/float16.hpp"
#include "common/type_helpers.hpp"
#include "gpu/jit/gemm/gemm_walk_orders.hpp"
#include "gpu/jit/ngen_type_bridge.hpp"
#include "gpu/ocl/gemm/xe_hp_systolic_gemm_copy_kernel.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

status_t xe_hp_systolic_gemm_t::pd_t::init(engine_t *engine) {
    using namespace prop_kind;
    using namespace data_type;
    using namespace primitive_kind;
    using smask_t = primitive_attr_t::skip_mask_t;
    using arch_t = compute::gpu_arch_t;

    assert(engine->kind() == engine_kind::gpu);
    auto *compute_engine = utils::downcast<compute::compute_engine_t *>(engine);

    if (!compute_engine->mayiuse_ngen_kernels()) return status::unimplemented;
    if (!compute_engine->mayiuse_large_grf_mode()) return status::unimplemented;

    dev_info_ = compute_engine->device_info();
    auto arch = dev_info_->gpu_arch();

    const auto &d = desc();

    bool dt_float_ok = (d->a_type() == d->b_type()
            && utils::one_of(d->a_type(), bf16, f16)
            && utils::one_of(d->c_type(), f32, d->a_type()));

    bool dt_int_ok = (utils::one_of(d->a_type(), u8, s8)
            && utils::one_of(d->b_type(), u8, s8) && (d->c_type() == s32));

    if (dt_int_ok) {
        if (attr()->zero_points_.defined(DNNL_ARG_SRC)) {
            const int *ao_i32 = nullptr;
            attr()->zero_points_.get(DNNL_ARG_SRC, nullptr, nullptr, &ao_i32);
            a_zp_ = (*ao_i32 != 0);
        } else if (!attr()->zero_points_.has_default_values(DNNL_ARG_SRC))
            return status::unimplemented;

        if (attr()->zero_points_.defined(DNNL_ARG_WEIGHTS)) {
            const int *bo_i32 = nullptr;
            attr()->zero_points_.get(
                    DNNL_ARG_WEIGHTS, nullptr, nullptr, &bo_i32);
            b_zp_ = (*bo_i32 != 0);
        } else if (!attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS))
            return status::unimplemented;

        c_zp_ = !attr()->zero_points_.has_default_values(DNNL_ARG_DST);
    }

    bool ok = set_default_formats(d->a_type());
    if (!ok) return status::unimplemented;

    CHECK(attr_.set_default_formats(dst_md(0)));

    if (use_fma()) return status::unimplemented;

    // LIMITATIONS:
    // - batch is not supported for unpacked inputs.
    // - runtime dims are not supported
    bool limits_ok
            = !utils::one_of(DNNL_RUNTIME_DIM_VAL, d->m(), d->n(), d->k());
    if (!packed_a())
        limits_ok = limits_ok && (d->lda() != DNNL_RUNTIME_DIM_VAL)
                && (d->batch() == 1);
    if (!packed_b())
        limits_ok = limits_ok && (d->ldb() != DNNL_RUNTIME_DIM_VAL)
                && (d->batch() == 1);
    if (!packed_c())
        limits_ok = limits_ok && (d->ldc() != DNNL_RUNTIME_DIM_VAL);

    auto attr_skip_mask = smask_t::oscale | smask_t::post_ops;

    if (dt_int_ok) attr_skip_mask |= smask_t::zero_points_runtime;

    bool arch_ok = (arch == arch_t::xe_hp) | (arch == arch_t::xe_hpg);

    ok = true && limits_ok && (dt_float_ok || dt_int_ok) && arch_ok
            && compute_engine->mayiuse(compute::device_ext_t::
                            intel_subgroup_split_matrix_multiply_accumulate)
            && attr()->has_default_values(attr_skip_mask)
            && attr()->output_scales_.mask_ == 0 && attr()->post_ops_.len() <= 2
            && IMPLICATION(with_bias(),
                    dt_float_ok
                            && utils::one_of(d->bias_type(), d->a_type(), f32)
                            && utils::one_of(bias_cmask(), 0, 1 << 0, 1 << 1));

    // check if there is sum post op and only at first place
    ok &= IMPLICATION(attr()->post_ops_.find(sum) != -1,
            attr()->post_ops_.find(sum) == 0
                    && attr()->post_ops_.find(sum, 1) == -1);

    // check if post ops are supported
    ok &= IMPLICATION(attr()->post_ops_.len() > 0,
            jit_post_op_injector_is_supported(attr()->post_ops_, true));

    if (dt_int_ok) {
        ok &= IMPLICATION(a_zp_, !packed_b()) && IMPLICATION(b_zp_, !packed_a())
                && IMPLICATION(
                        c_zp_, !attr()->zero_points_.defined(DNNL_ARG_DST));

        int cmask_a = 0, cmask_b = 0, cmask_c = 0;
        attr()->zero_points_.get(DNNL_ARG_WEIGHTS, nullptr, &cmask_b, nullptr);
        attr()->zero_points_.get(DNNL_ARG_SRC, nullptr, &cmask_a, nullptr);
        attr()->zero_points_.get(DNNL_ARG_DST, nullptr, &cmask_c, nullptr);
        ok &= (cmask_a == 0) && (cmask_b == 0)
                && utils::one_of(cmask_c, 0, 1 << 0, 1 << 1);
    }

    if (!ok) return status::unimplemented;

    return status::success;
}

namespace {
struct nocopy_table_t {
    int mn_limit[2][2]; // Use no-copy if m*n < mn_limit * mn_limit and
    int k_limit[2][2]; // Use no-copy if k < k_limit
};

const nocopy_table_t xe_hp_f16_nocopy_table[] = {
        // NN     NT     TN    TT
        {{{1280, 768}, {512, 384}}, {{512, 768}, {1024, 512}}}};

const nocopy_table_t xe_hp_bf16_nocopy_table[] = {
        // NN   NT     TN   TT
        {{{512, 256}, {512, 512}}, {{512, 256}, {384, 384}}}};

const nocopy_table_t xe_hp_x8x8s32_nocopy_table[] = {
        // NN   NT     TN   TT
        {{{384, 384}, {384, 384}}, {{384, 512}, {384, 256}}}};
} // namespace

bool xe_hp_systolic_gemm_t::pd_t::use_fma() {
    using namespace data_type;

    const auto &d = desc();

    if (any_prepacked_) return false;

    // Use FMA implementation if one matrix is very small.
    if (d->m() < 32 && d->n() < 32) return true;
    if (d->m() < 32 && d->k() < 32) return true;
    if (d->n() < 32 && d->k() < 32) return true;

    // Use FMA for small/medium sizes.
    if (utils::one_of(d->c_type(), bf16, f16, s32)) {
        const nocopy_table_t *all_tables[3] = {xe_hp_f16_nocopy_table,
                xe_hp_bf16_nocopy_table, xe_hp_x8x8s32_nocopy_table};
        const int type_idx
                = (d->c_type() == f16) ? 0 : (d->c_type() == bf16) ? 1 : 2;
        const nocopy_table_t *table = all_tables[type_idx];
        const long mnl = table->mn_limit[d->transa()][d->transb()];
        const long kl = table->k_limit[d->transa()][d->transb()];

        if ((d->m() * d->n() < mnl * mnl) && (d->k() < kl)) return true;
    }

    return false;
}

bool xe_hp_systolic_gemm_t::pd_t::set_default_formats(data_type_t dt) {
    using namespace format_tag;
    using new_kernel_t = gen_gemm_xehp_systolic_kernel_t;

    auto sz = types::data_type_size(dt);
    const auto &d = desc();
    auto arch = dev_info_->gpu_arch();

    memory_desc_wrapper a_mdw(&desc_.b_desc);
    memory_desc_wrapper b_mdw(&desc_.a_desc);
    memory_desc_wrapper c_mdw(&desc_.c_desc);

    bool a_any = a_mdw.format_any();
    bool b_any = b_mdw.format_any();
    bool c_any = c_mdw.format_any();
    bool batch = d->is_batched();

    format_tag_t a_packed_tag = batch ? ((sz == 2) ? aCB4c8b8c2b : aCB4c8b8c4b)
                                      : ((sz == 2) ? BA4b8a8b2a : BA4b8a8b4a);
    format_tag_t b_packed_tag_48 = batch ? ((sz == 2) ? aBC48b16c : aBC48b32c)
                                         : ((sz == 2) ? AB48a16b : AB48a32b);
    format_tag_t b_packed_tag_32 = batch ? ((sz == 2) ? aBC32b16c : aBC32b32c)
                                         : ((sz == 2) ? AB32a16b : AB32a32b);
    format_tag_t unpacked_tag = batch ? abc : ab;

    bool a_prepacked = a_mdw.matches_tag(a_packed_tag);
    bool bc_prepacked_32 = b_mdw.matches_tag(b_packed_tag_32)
            || c_mdw.matches_tag(b_packed_tag_32);
    bool bc_prepacked_48 = b_mdw.matches_tag(b_packed_tag_48)
            || c_mdw.matches_tag(b_packed_tag_48);
    bool c_prepacked = c_mdw.matches_tag(b_packed_tag_32)
            || c_mdw.matches_tag(b_packed_tag_48);

    any_prepacked_ = a_prepacked || bc_prepacked_32 || bc_prepacked_48;

    unroll_m_ = 32;
    unroll_n_ = 0;
    kernel_tag_ = 0;
    if (bc_prepacked_32)
        unroll_n_ = 32;
    else if (bc_prepacked_48)
        unroll_n_ = 48;

    use_new_kernels_ = !c_prepacked && !with_ab_zero_points() && (d->k() >= 64);

    new_kernel_t::choose_unrolls(arch, dev_info_->eu_count(), d->a_type(),
            d->b_type(), d->c_type(), d->m(), d->n(), d->k(), d->batch(),
            unroll_m_, unroll_n_, kernel_tag_);

    format_tag_t b_packed_tag
            = (unroll_n_ == 48) ? b_packed_tag_48 : b_packed_tag_32;
    format_tag_t c_packed_tag = use_new_kernels_ ? unpacked_tag : b_packed_tag;

    packed_a_ = packed_b_ = packed_c_ = false;

    if (a_any) {
        CHECK(memory_desc_init_by_tag(
                desc_.b_desc, b_zp_ ? unpacked_tag : a_packed_tag));
        auto &ld = desc_.b_desc.padded_dims[batch ? 1 : 0];
        ld = nice_ld(ld, int(sz));
        desc_.b_desc.format_desc.blocking.strides[batch ? 2 : 1]
                = unroll_m_ * ld;
        packed_a_ = true;
    } else if (a_mdw.matches_one_of_tag(a_packed_tag, ab, ba, abc, acb)
            == undef)
        return false;

    if (b_any) {
        CHECK(memory_desc_init_by_tag(
                desc_.a_desc, a_zp_ ? unpacked_tag : b_packed_tag));
        auto &ld = desc_.a_desc.padded_dims[batch ? 2 : 1];
        ld = nice_ld(ld, int(sz));
        desc_.a_desc.format_desc.blocking.strides[batch ? 1 : 0]
                = unroll_n_ * ld;
        packed_b_ = true;
    } else if (b_mdw.matches_one_of_tag(b_packed_tag, ab, ba, abc, acb)
            == undef)
        return false;

    if (c_any)
        CHECK(memory_desc_init_by_tag(desc_.c_desc, c_packed_tag));
    else if (c_mdw.matches_one_of_tag(c_packed_tag, ab, abc) == undef)
        return false;

    packed_a_ = packed_a_ || a_mdw.matches_tag(a_packed_tag);
    packed_b_ = packed_b_ || b_mdw.matches_tag(b_packed_tag);
    packed_c_ = c_mdw.matches_tag(b_packed_tag);

    return gpu_gemm_pd_t::set_default_formats();
}

status_t xe_hp_systolic_gemm_t::init(engine_t *engine) {
    arch_ = pd()->dev_info_->gpu_arch();
    eu_count_ = pd()->dev_info_->eu_count();

    auto a_type = pd()->desc()->a_type();
    auto b_type = pd()->desc()->b_type();

    int cmask = -1;

    if (pd()->with_c_zero_points())
        pd()->attr()->zero_points_.get(DNNL_ARG_DST, nullptr, &cmask, nullptr);
    else if (pd()->with_bias())
        cmask = pd()->bias_cmask();

    switch (cmask) {
        case 0: co_kind_ = 'F'; break;
        case (1 << 1): co_kind_ = 'R'; break;
        case (1 << 0): co_kind_ = 'C'; break;
        case -1:
        default: co_kind_ = 'N'; break;
    }

    if (get_verbose() >= 2) {
        char tag_s[2] = {pd()->kernel_tag(), 0};
        printf("dnnl_verbose,info,gpu,gemm,kernel:%dx%d,%s,new:%c\n",
                pd()->unroll_m(), pd()->unroll_n(), tag_s,
                pd()->use_new_kernels() ? 'Y' : 'N');
    }

    // Initialize compute kernels (assembly)
    {
        auto status = pd()->use_new_kernels() ? init_compute_new(engine)
                                              : init_compute_old(engine);
        if (status != status::success) return status;
    }

    // Initialize copy kernels (OpenCL)
    for (bool copy_b : {false, true}) {
        for (bool clear_sum : {false, true}) {
            if (clear_sum && !pd()->with_ab_zero_points()) continue;
            if (!copy_b ? pd()->packed_a() : pd()->packed_b()) continue;

            compute::kernel_ctx_t kernel_ctx;

            auto trans
                    = !copy_b ? pd()->desc()->transa() : pd()->desc()->transb();
            auto status
                    = ocl::xe_hp_systolic_gemm_copy_kernel_t::init_kernel_ctx(
                            kernel_ctx, !copy_b ? a_type : b_type,
                            pd()->unroll_n(), copy_b, trans,
                            pd()->with_ab_zero_points(), clear_sum);
            if (status != status::success) return status;

            create_kernel(engine, &copy_kernel_[copy_b][clear_sum],
                    "xe_hp_systolic_gemm_copy", kernel_ctx);
            if (!copy_kernel_[copy_b][clear_sum]) return status::runtime_error;
        }
    }

    return status::success;
}

status_t xe_hp_systolic_gemm_t::init_compute_old(engine_t *engine) {
    using kernel_t = xehp_systolic_gemm_kernel_t<gpu_xe_hp>;
    using arch_t = compute::gpu_arch_t;

    kernel_t::config_t cfg;

    auto a_type = pd()->desc()->a_type();
    auto b_type = pd()->desc()->b_type();
    auto c_type = pd()->desc()->c_type();
    auto acc_type = pd()->impl_acc_type();

    cfg.a_type = convert_dnnl_type_to_ngen(a_type);
    cfg.b_type = convert_dnnl_type_to_ngen(b_type);
    cfg.c_type = convert_dnnl_type_to_ngen(c_type);
    cfg.acc_type = convert_dnnl_type_to_ngen(acc_type);
    cfg.alpha1 = (pd()->alpha() == 1.0f);
    cfg.beta0 = (pd()->beta() == 0.0f);
    cfg.beta1 = (pd()->beta() == 1.0f);
    cfg.post_ops = pd()->attr()->post_ops_;
    cfg.a_bias = cfg.b_bias = pd()->with_ab_zero_points();
    cfg.c_packed = pd()->packed_c();
    cfg.batch = pd()->with_batch();
    walk_n_first_ = cfg.walk_n_first
            = (pd()->desc()->m() >= 2 * pd()->desc()->n());
    cfg.tile_m = pd()->unroll_m();
    cfg.tile_n = pd()->unroll_n();
    cfg.global_3x_buf = (cfg.tile_n == 32);

    if (pd()->with_c_zero_points())
        cfg.co_type = cfg.c_type;
    else if (pd()->with_bias()) {
        cfg.early_c_bias = true;
        cfg.co_type = convert_dnnl_type_to_ngen(pd()->desc()->bias_type());
    }

    switch (co_kind_) {
        case 'F': cfg.c_bias = kernel_t::bias_t::fixed; break;
        case 'R': cfg.c_bias = kernel_t::bias_t::row; break;
        case 'C': cfg.c_bias = kernel_t::bias_t::column; break;
        case 'N':
        default: cfg.c_bias = kernel_t::bias_t::none; break;
    }

    bool may_k_block = (pd()->desc()->k() > kernel_t::min_block_k(a_type));
    bool got_info = false;

    for (bool first_k_block : {false, true}) {
        for (bool last_k_block : {false, true}) {
            if ((!first_k_block || !last_k_block) && !may_k_block) continue;
            if (may_k_block && last_k_block
                    && (cfg.c_bias == kernel_t::bias_t::none)
                    && !cfg.have_post_op())
                kernel_[first_k_block][last_k_block]
                        = kernel_[first_k_block][false];
            else if (may_k_block && first_k_block && cfg.beta1)
                kernel_[first_k_block][last_k_block]
                        = kernel_[false][last_k_block];
            else {
                auto cfg_copy = cfg;
                if (!first_k_block) {
                    cfg_copy.beta0 = false;
                    cfg_copy.beta1 = true;
                }
                if (!last_k_block) {
                    cfg_copy.c_bias = kernel_t::bias_t::none;
                    cfg_copy.post_ops = post_ops_t();
                }

                switch (arch_) {
                    case arch_t::xe_hp: {
                        auto kernel = kernel_t(cfg_copy);

                        create_kernel(engine,
                                &kernel_[first_k_block][last_k_block], kernel);

                        if (!got_info) {
                            compute_info_ = kernel.driver_info(eu_count_);
                            got_info = true;
                        }
                        break;
                    }
                    case arch_t::xe_hpg: {
                        using kernel_xe_hpg_t
                                = xehp_systolic_gemm_kernel_t<gpu_xe_hpg>;
                        cfg_copy.emulate64 = true;
                        auto kernel = kernel_xe_hpg_t(
                                cfg_copy.cast<kernel_xe_hpg_t::config_t>());

                        create_kernel(engine,
                                &kernel_[first_k_block][last_k_block], kernel);

                        if (!got_info) {
                            compute_info_ = kernel.driver_info(eu_count_);
                            got_info = true;
                        }
                        break;
                    }
                    default:
                        assert(!"Unsupported GPU architecture.");
                        return status::unimplemented;
                        break;
                }

                if (!kernel_[first_k_block][last_k_block])
                    return status::runtime_error;
            }
        }
    }

    return status::success;
}

status_t xe_hp_systolic_gemm_t::init_compute_new(engine_t *engine) {
    using kernel_t = gen_gemm_xehp_systolic_kernel_t;
    using offset_t = kernel_t::offset_t;

    auto a_type = pd()->desc()->a_type();
    auto b_type = pd()->desc()->b_type();
    auto c_type = pd()->desc()->c_type();
    auto co_type = pd()->with_bias() ? pd()->desc()->bias_type() : c_type;
    auto acc_type = pd()->impl_acc_type();

    offset_t ab_offset
            = pd()->with_ab_zero_points() ? offset_t::fixed : offset_t::none;
    offset_t c_offset
            = pd()->with_c_zero_points() ? offset_t::runtime : offset_t::none;
    offset_t bias_offset
            = pd()->with_bias() ? offset_t::runtime : offset_t::none;

    bool may_k_block = (pd()->desc()->k() > kernel_t::min_block_k(a_type));
    bool got_info = false;

    bool with_eltwise
            = (pd()->attr()->post_ops_.find(primitive_kind::eltwise) != -1);

    for (bool first_k_block : {false, true}) {
        for (bool last_k_block : {false, true}) {
            if ((!first_k_block || !last_k_block) && !may_k_block) continue;
            if (may_k_block && last_k_block && (c_offset == offset_t::none)
                    && !with_eltwise)
                kernel_[first_k_block][last_k_block]
                        = kernel_[first_k_block][false];
            else if (may_k_block && first_k_block && pd()->beta() == 1.0f)
                kernel_[first_k_block][last_k_block]
                        = kernel_[false][last_k_block];
            else {
                auto this_beta = pd()->beta();
                auto this_c_offset = c_offset;
                auto *this_post_ops = &pd()->attr()->post_ops_;
                post_ops_t no_post_ops;

                if (!first_k_block) this_beta = 1.0f;
                if (!last_k_block) {
                    this_c_offset = offset_t::none;
                    this_post_ops = &no_post_ops;
                }

                kernel_t kernel;

                auto status = kernel.init(arch_, pd()->with_batch(), ab_offset,
                        ab_offset, this_c_offset, bias_offset, pd()->alpha(),
                        this_beta, *this_post_ops, a_type, b_type, c_type,
                        co_type, acc_type, pd()->unroll_m(), pd()->unroll_n(),
                        pd()->kernel_tag());

                if (status != status::success) return status;

                if (!got_info) {
                    compute_info_ = kernel.driver_info();
                    got_info = true;
                }

                create_kernel(
                        engine, &kernel_[first_k_block][last_k_block], kernel);

                if (!kernel_[first_k_block][last_k_block])
                    return status::runtime_error;
            }
        }
    }

    return status::success;
}

status_t xe_hp_systolic_gemm_t::init_res_storage(
        engine_t *engine, gpu_resource_t *r) const {
    auto a_type = pd()->desc()->a_type();
    auto b_type = pd()->desc()->b_type();

    auto m = pd()->desc()->m();
    auto n = pd()->desc()->n();
    auto k = pd()->desc()->k();

    int64_t align_m = compute_info_.wgTile(LoopM);
    int64_t align_n = compute_info_.wgTile(LoopN);

    auto m_aligned = utils::rnd_up(m, align_m);
    auto n_aligned = utils::rnd_up(n, align_n);

    auto max_ldab_packed = max_ld_packed(k);

    if (!pd()->packed_a()) {
        memory_storage_t *a_packed_ptr;
        engine->create_memory_storage(&a_packed_ptr,
                m_aligned * max_ldab_packed * types::data_type_size(a_type));
        if (!a_packed_ptr) return status::runtime_error;

        std::unique_ptr<memory_storage_t> a_packed(a_packed_ptr);
        r->add_memory_storage(A_PACKED_, std::move(a_packed));
    }

    if (!pd()->packed_b()) {
        memory_storage_t *b_packed_ptr;
        engine->create_memory_storage(&b_packed_ptr,
                n_aligned * max_ldab_packed * types::data_type_size(b_type));
        if (!b_packed_ptr) return status::runtime_error;

        std::unique_ptr<memory_storage_t> b_packed(b_packed_ptr);
        r->add_memory_storage(B_PACKED_, std::move(b_packed));
    }

    return status::success;
}

bool xe_hp_systolic_gemm_t::enable_mn_blocking() const {
    return (pd()->desc()->m() >= 8192) && (pd()->desc()->n() >= 8192);
}

std::tuple<int64_t, int64_t, int64_t>
xe_hp_systolic_gemm_t::get_blocking() const {
    int64_t m = pd()->desc()->m();
    int64_t n = pd()->desc()->n();
    int64_t k = pd()->desc()->k();

    int64_t unroll_k = compute_info_.unroll[LoopK];

    int64_t align_m = compute_info_.wgTile(LoopM);
    int64_t align_n = compute_info_.wgTile(LoopN);

    m = utils::rnd_up(m, align_m);
    n = utils::rnd_up(n, align_n);

    // Decide on m/n blocking.
    int64_t block_m = compute_info_.blocking[LoopM];
    int64_t block_n = compute_info_.blocking[LoopN];
    int64_t max_block_m = utils::rnd_up(m, align_m);
    int64_t max_block_n = utils::rnd_up(n, align_n);

    if (enable_mn_blocking()) {
        if (n <= block_n)
            block_m = (block_m * block_n) / n;
        else if (m <= block_m)
            block_n = (block_m * block_n) / m;
        else if (n < 2 * block_n) {
            block_n = utils::rnd_up(n / 2, align_n);
            block_m = (2 * block_m * block_n) / n;
        } else if (m < 2 * block_m) {
            block_m = utils::rnd_up(m / 2, align_m);
            block_n = (2 * block_m * block_n) / m;
        }

        block_m = utils::rnd_dn(nstl::min(block_m, max_block_m), align_m);
        block_n = utils::rnd_dn(nstl::min(block_n, max_block_n), align_n);
    } else {
        block_m = m;
        block_n = n;
    }

    // Decide on k blocking.
    int64_t block_k = compute_info_.blocking[LoopK];
    int64_t nblock_k = utils::div_up(k, block_k);
    block_k = utils::div_up(k, nblock_k);
    block_k = utils::rnd_up(
            (pd()->desc()->acc_type != pd()->desc()->c_type()) ? k : block_k,
            unroll_k);

    return std::make_tuple(block_m, block_n, block_k);
}

status_t xe_hp_systolic_gemm_t::launch_copy(const gemm_exec_ctx_t &ctx,
        int64_t r, int64_t c, const memory_storage_t &src, int64_t offset_src,
        int64_t ld_src, const memory_storage_t &dst, int32_t offset_dst,
        int32_t ld_dst, bool copyb) const {

    using copy_kernel_t = ocl::xe_hp_systolic_gemm_copy_kernel_t;

    if (pd()->with_ab_zero_points()) {
        auto status
                = launch_clear_sum(ctx, r, c, dst, offset_dst, ld_dst, copyb);
        if (status) return status;
    }

    int64_t unroll_k = compute_info_.unroll[LoopK];

    int64_t align_r = 0, align_c = 0;

    if (!copyb) {
        align_r = compute_info_.wgTile(LoopM);
        align_c = unroll_k;
    } else {
        align_r = unroll_k;
        align_c = compute_info_.wgTile(LoopN);
    }

    bool transa = (pd()->desc()->transa() == dnnl_trans);
    bool transb = (pd()->desc()->transb() == dnnl_trans);
    bool trans = !copyb ? transa : transb;

    auto &kernel = copy_kernel_[copyb][false];

    assert(kernel);
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, r);
    arg_list.set(1, c);
    arg_list.set(2, src);
    arg_list.set(3, offset_src);
    arg_list.set(4, ld_src);
    arg_list.set(5, dst);
    arg_list.set(6, offset_dst);
    arg_list.set(7, ld_dst);

    auto elt_size = types::data_type_size(pd()->desc()->a_type());
    size_t r_threads = utils::div_up(utils::rnd_up(r, align_r),
            copy_kernel_t::unroll_r(elt_size, pd()->unroll_n(), copyb, trans));
    size_t c_threads = utils::div_up(utils::rnd_up(c, align_c),
            copy_kernel_t::unroll_c(elt_size, pd()->unroll_n(), copyb, trans));
    size_t sg = copy_kernel_t::subgroup_size(elt_size, copyb, trans);

    size_t r_lsz = trans ? 1 : 16;
    size_t c_lsz = trans ? 16 : 1;

    if (r_threads > r_lsz)
        r_threads = utils::rnd_up(r_threads, r_lsz);
    else
        r_lsz = r_threads;

    if (c_threads > c_lsz)
        c_threads = utils::rnd_up(c_threads, c_lsz);
    else
        c_lsz = c_threads;

    size_t gws[3] = {r_threads * sg, c_threads, 1};
    size_t lws[3] = {r_lsz * sg, c_lsz, 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, kernel, arg_list);
}

status_t xe_hp_systolic_gemm_t::launch_clear_sum(const gemm_exec_ctx_t &ctx,
        int64_t r, int64_t c, const memory_storage_t &dst, int32_t offset_dst,
        int32_t ld_dst, bool copyb) const {

    auto &kernel = copy_kernel_[copyb][true];

    assert(kernel);
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, r);
    arg_list.set(1, c);
    arg_list.set(2, dst);
    arg_list.set(3, offset_dst);
    arg_list.set(4, ld_dst);

    auto elt_size = types::data_type_size(pd()->desc()->a_type());
    size_t threads = !copyb ? utils::div_up(r, pd()->unroll_m())
                            : utils::div_up(c, pd()->unroll_n());
    size_t sg = ocl::xe_hp_systolic_gemm_copy_kernel_t::subgroup_size_clear_sum(
            elt_size, copyb);

    size_t gws[3] = {threads * sg, 1, 1};
    size_t lws[3] = {sg, 1, 1};

    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, kernel, arg_list);
}

status_t xe_hp_systolic_gemm_t::launch_compute(const gemm_exec_ctx_t &ctx,
        int32_t m, int32_t n, int32_t k, const memory_storage_t &ap,
        int64_t offset_a, int32_t lda, const memory_storage_t &bp,
        int64_t offset_b, int32_t ldb, const memory_storage_t &c,
        int64_t offset_c, int32_t ldc, float alpha, float beta, int16_t ao,
        int16_t bo, const memory_storage_t &co, int32_t offset_co,
        bool first_k_block, bool last_k_block, int32_t batch, int32_t stride_a,
        int32_t stride_b, int32_t stride_c) const {

    auto tg_m = compute_info_.wg[LoopM];
    auto tg_n = compute_info_.wg[LoopN];

    auto &kernel = kernel_[first_k_block][last_k_block];

    //   kernel void gemm_kernel(global char *Ap, global uchar *Bp, global int *C,
    //                           int k, int ldc,
    //                           long offsetA, long offsetB, long offsetC,
    //                           int m, int n,
    //                           float alpha, float beta,
    //                           int lda, int ldb)

    assert(kernel);

    compute::kernel_arg_list_t arg_list;
    int argn = 0;
    arg_list.set(argn++, ap);
    arg_list.set(argn++, bp);
    arg_list.set(argn++, c);
    arg_list.set(argn++, offset_a);
    arg_list.set(argn++, offset_b);
    arg_list.set(argn++, offset_c);
    arg_list.set(argn++, lda);
    arg_list.set(argn++, ldb);
    arg_list.set(argn++, ldc);
    arg_list.set(argn++, m);
    arg_list.set(argn++, n);
    arg_list.set(argn++, k);
    arg_list.set(argn++, alpha);
    arg_list.set(argn++, beta);
    if (pd()->with_ab_zero_points()) {
        uint32_t abo = (uint16_t(ao) | (uint16_t(bo) << 16));
        arg_list.set(argn++, abo);
    }
    if ((pd()->with_bias() || pd()->with_c_zero_points())) {
        arg_list.set(argn++, co);
        arg_list.set(argn++, offset_co);
    }
    if (pd()->use_new_kernels()) {
        uint32_t flags = 0;
        if (co_kind_ == 'R') flags |= FlagCORow;
        if (co_kind_ == 'C') flags |= FlagCOColumn;
        if (!first_k_block) flags |= FlagNoninitialKBlock;
        if (!last_k_block) flags |= FlagNonfinalKBlock;
        arg_list.set(argn++, flags);
    }
    if (pd()->with_batch()) {
        arg_list.set(argn++, stride_a);
        arg_list.set(argn++, stride_b);
        arg_list.set(argn++, stride_c);
    }

    auto thread_m = utils::div_up(m, pd()->unroll_m() * tg_m) * tg_m;
    auto thread_n = utils::div_up(n, pd()->unroll_n() * tg_n) * tg_n;

    if (walk_n_first_) std::swap(thread_m, thread_n);

    size_t gws[3] = {size_t(thread_m), size_t(thread_n), 1};
    size_t lws[3] = {size_t(tg_m), size_t(tg_n), 1};
    if (pd()->with_batch()) gws[2] = batch;

    lws[1] *= compute_info_.wgExpand;
    gws[1] *= compute_info_.wgExpand;

    gemm_linear_order_args(arg_list, argn, lws, gws, m, n, false, compute_info_,
            pd()->dev_info_);

    lws[0] *= compute_info_.subgroupSize;
    gws[0] *= compute_info_.subgroupSize;

    auto nd_range = compute::nd_range_t(gws, lws);

    return parallel_for(ctx, nd_range, kernel, arg_list);
}

status_t xe_hp_systolic_gemm_t::execute(const gemm_exec_ctx_t &ctx) const {
    auto a_type = pd()->desc()->a_type();
    auto b_type = pd()->desc()->b_type();
    auto c_type = pd()->desc()->c_type();
    auto bias_type = pd()->desc()->bias_type();

    auto m = pd()->desc()->m();
    auto n = pd()->desc()->n();
    auto k = pd()->desc()->k();
    auto batch = pd()->desc()->batch();

    bool packed_a = pd()->packed_a();
    bool packed_b = pd()->packed_b();
    bool packed_c = pd()->packed_c();

    auto lda = packed_a ? 0 : pd()->desc()->lda();
    auto ldb = packed_b ? 0 : pd()->desc()->ldb();
    auto ldc = packed_c ? pd()->ldc_packed() : pd()->desc()->ldc();

    auto stride_a = pd()->desc()->stride_a();
    auto stride_b = pd()->desc()->stride_b();
    auto stride_c = pd()->desc()->stride_c();

    auto alpha = pd()->alpha();
    auto beta = pd()->beta();

    auto &a = GEMM_CTX_ARG_STORAGE(b);
    auto &b = GEMM_CTX_ARG_STORAGE(a);
    auto &c = GEMM_CTX_ARG_STORAGE(c);
    auto &c_zp = GEMM_CTX_ARG_STORAGE(c_zero_point);
    auto &bias = GEMM_CTX_ARG_STORAGE(bias);
    auto *co = &c_zp;

    auto &a_packed = packed_a ? a : CTX_GPU_RES_STORAGE(A_PACKED_);
    auto &b_packed = packed_b ? b : CTX_GPU_RES_STORAGE(B_PACKED_);

    int32_t ao = 0, bo = 0;

    size_t off_a0
            = a.offset() / types::data_type_size(a_type) + pd()->dyn_offset_a;
    size_t off_b0
            = b.offset() / types::data_type_size(b_type) + pd()->dyn_offset_b;
    size_t off_c0
            = c.offset() / types::data_type_size(c_type) + pd()->dyn_offset_c;
    size_t off_co0 = 0;

    if (pd()->with_ab_zero_points()) {
        const int *ao_i32 = nullptr;
        const int *bo_i32 = nullptr;
        pd()->attr()->zero_points_.get(DNNL_ARG_SRC, nullptr, nullptr, &ao_i32);
        pd()->attr()->zero_points_.get(
                DNNL_ARG_WEIGHTS, nullptr, nullptr, &bo_i32);
        ao = -*ao_i32;
        bo = -*bo_i32;
    }

    if (pd()->with_bias()) {
        off_co0 = bias.offset() / types::data_type_size(bias_type);
        co = &bias;
    }

    int64_t block_m = 0, block_n = 0, block_k = 0;
    std::tie(block_m, block_n, block_k) = get_blocking();

    auto ld_packed = get_ld_packed(k);
    auto lda_packed = packed_a ? pd()->lda_packed() : ld_packed;
    auto ldb_packed = packed_b ? pd()->ldb_packed() : ld_packed;

    status_t status;

    if (!packed_a) {
        assert(batch == 1);
        status = launch_copy(
                ctx, m, k, a, off_a0, lda, a_packed, 0, lda_packed, false);
        if (status) return status;
    }

    if (!packed_b) {
        assert(batch == 1);
        status = launch_copy(
                ctx, k, n, b, off_b0, ldb, b_packed, 0, ldb_packed, true);
        if (status) return status;
    }

    for (int64_t Bk = 0; Bk < k; Bk += block_k) {
        int64_t size_k = k - Bk;
        bool first_k_block = (Bk == 0);
        bool last_k_block = (size_k <= block_k);
        if (!last_k_block) size_k = block_k;

        for (int64_t Bm = 0; Bm < m; Bm += block_m) {
            int64_t size_m = m - Bm;
            if (size_m > block_m) size_m = block_m;

            auto off_a_packed = Bm * lda_packed + Bk * pd()->unroll_m();
            if (packed_a) off_a_packed += off_a0;

            for (int64_t Bn = 0; Bn < n; Bn += block_n) {
                int64_t size_n = n - Bn;
                if (size_n > block_n) size_n = block_n;

                auto off_b_packed = Bn * ldb_packed + Bk * pd()->unroll_n();
                if (packed_b) off_b_packed += off_b0;

                auto off_c = off_c0 + Bm + Bn * ldc;
                auto off_co = int32_t(off_co0);
                switch (co_kind_) {
                    case 'R': off_co += Bm; break;
                    case 'C': off_co += Bn; break;
                    default: break;
                }

                float this_beta = first_k_block ? beta : 1.0f;
                status = launch_compute(ctx, size_m, size_n, size_k, a_packed,
                        off_a_packed, lda_packed, b_packed, off_b_packed,
                        ldb_packed, c, off_c, ldc, alpha, this_beta, ao, bo,
                        *co, off_co, first_k_block, last_k_block, batch,
                        stride_a, stride_b, stride_c);
                if (status) return status;
            }
        }
    }

    return status::success;
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
