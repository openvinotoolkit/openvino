/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#include <cstring>
#include <mutex>

#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/cpu_engine.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace {
#ifdef DNNL_ENABLE_MAX_CPU_ISA
cpu_isa_t init_max_cpu_isa() {
    cpu_isa_t max_cpu_isa_val = isa_all;
    char buf[64];
    if (getenv("DNNL_MAX_CPU_ISA", buf, sizeof(buf)) > 0) {

#define IF_HANDLE_CASE(cpu_isa) \
    if (std::strcmp(buf, cpu_isa_traits<cpu_isa>::user_option_env) == 0) \
    max_cpu_isa_val = cpu_isa
#define ELSEIF_HANDLE_CASE(cpu_isa) else IF_HANDLE_CASE(cpu_isa)

        IF_HANDLE_CASE(isa_all);
        ELSEIF_HANDLE_CASE(asimd);
        ELSEIF_HANDLE_CASE(sve_512);

#undef IF_HANDLE_CASE
#undef ELSEIF_HANDLE_CASE
    }

    return max_cpu_isa_val;
}

set_once_before_first_get_setting_t<cpu_isa_t> &max_cpu_isa() {
    static set_once_before_first_get_setting_t<cpu_isa_t> max_cpu_isa_setting(
            init_max_cpu_isa());
    return max_cpu_isa_setting;
}
#endif
} // namespace

struct isa_info_t {
    isa_info_t(cpu_isa_t aisa) : isa(aisa) {};

    // this converter is needed as code base defines certain ISAs
    // that the library does not expose (eg sve),
    // so the internal and external enum types do not coincide.
    dnnl_cpu_isa_t convert_to_public_enum(void) const {
        switch (isa) {
            case sve_512:
                return static_cast<dnnl_cpu_isa_t>(dnnl_cpu_isa_sve_512);
            case asimd: return static_cast<dnnl_cpu_isa_t>(dnnl_cpu_isa_asimd);
            default: return dnnl_cpu_isa_all;
        }
    }

    const char *get_name() const {
        switch (isa) {
            case sve_512: return "AArch64 SVE (512 bits)";
            case asimd: return "AArch64 (with Advanced SIMD & floating-point)";
            default: return "AArch64";
        }
    }

    cpu_isa_t isa;
};

static isa_info_t get_isa_info_t(void) {
    // descending order due to mayiuse check
#define HANDLE_CASE(cpu_isa) \
    if (mayiuse(cpu_isa)) return isa_info_t(cpu_isa);
    HANDLE_CASE(sve_512);
    HANDLE_CASE(asimd);
#undef HANDLE_CASE
    return isa_info_t(isa_any);
}

const char *get_isa_info() {
    return get_isa_info_t().get_name();
}

cpu_isa_t get_max_cpu_isa() {
    return get_isa_info_t().isa;
}

cpu_isa_t get_max_cpu_isa_mask(bool soft) {
    MAYBE_UNUSED(soft);
#ifdef DNNL_ENABLE_MAX_CPU_ISA
    return max_cpu_isa().get(soft);
#else
    return isa_all;
#endif
}

status_t set_max_cpu_isa(dnnl_cpu_isa_t isa) {
    using namespace dnnl::impl::status;
#ifdef DNNL_ENABLE_MAX_CPU_ISA
    using namespace dnnl::impl;
    using namespace dnnl::impl::cpu;

    cpu_isa_t isa_to_set = isa_any;
#define HANDLE_CASE(cpu_isa) \
    case cpu_isa_traits<cpu_isa>::user_option_val: isa_to_set = cpu_isa; break;
    switch (isa) {
        HANDLE_CASE(isa_all);
        HANDLE_CASE(asimd);
        HANDLE_CASE(sve_512);
        default: return invalid_arguments;
    }
    assert(isa_to_set != isa_any);
#undef HANDLE_CASE

    if (max_cpu_isa().set(isa_to_set))
        return success;
    else
        return invalid_arguments;
#else
    return unimplemented;
#endif
}

dnnl_cpu_isa_t get_effective_cpu_isa() {
    return get_isa_info_t().convert_to_public_enum();
}
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
