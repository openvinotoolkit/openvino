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

#ifndef CPU_X64_BRGEMM_INNER_PRODUCT_UTILS_HPP
#define CPU_X64_BRGEMM_INNER_PRODUCT_UTILS_HPP

#include "dnnl_types.h"

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_engine.hpp"
#include "cpu/cpu_inner_product_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_brgemm_primitive_conf.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace brgemm_inner_product_utils {

status_t init_ip_conf(cpu_isa_t isa, jit_brgemm_primitive_conf_t &jbgp,
        const inner_product_desc_t &ipd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads);

void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_brgemm_primitive_conf_t &jbgp);

static const int max_num_brg_kernels_ip = 2 * 2 * 2 * 2;

int get_brg_kernel_index(const jit_brgemm_primitive_conf_t &jbgp,
        bool do_initialization, bool is_M_tail, bool is_N_tail, bool is_K_tail);

int get_os_block(const jit_brgemm_primitive_conf_t &jbgp, bool try_to_adjust,
        bool is_adjustment);
int get_oc_block(
        const jit_brgemm_primitive_conf_t &jbgp, bool try_to_adjust = false);

int ip_fwd_get_oc_block(const jit_brgemm_primitive_conf_t &jbgp);
int ip_fwd_get_nb_oc_blocking(
        const jit_brgemm_primitive_conf_t &jbgp, bool is_adjustment = false);
bool ip_fwd_adjust_thread_balance(const jit_brgemm_primitive_conf_t &jbgp);
int ip_fwd_get_adjusted_oc_block(const jit_brgemm_primitive_conf_t &jbgp);

format_tag_t get_brgemm_ip_weights_tag(
        cpu_isa_t isa, const jit_brgemm_primitive_conf_t &jbgp);
bool post_ops_ok(jit_brgemm_primitive_conf_t &jbgp,
        const primitive_attr_t &attr, const memory_desc_wrapper &dst_d);
void thread_balance(const jit_brgemm_primitive_conf_t &j, int &nb_os_blocking_,
        int &nthr_, int &nthr_mb_, int &nthr_oc_b_, int &nthr_ic_b_);
status_t init_ip_conf_fwd(jit_brgemm_primitive_conf_t &jbgp,
        const primitive_attr_t &attr, const memory_desc_wrapper &dst_d);
status_t init_ip_conf_bwd_d(jit_brgemm_primitive_conf_t &jbgp);
status_t init_ip_conf_bwd_w(jit_brgemm_primitive_conf_t &jbgp);

} // namespace brgemm_inner_product_utils

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
