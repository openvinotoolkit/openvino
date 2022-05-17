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

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct jit_amx_tilecfg_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_amx_tilecfg_t)

    // TODO: Need to check status
    jit_amx_tilecfg_t()
        : jit_generator(nullptr, MAX_CODE_SIZE, true, avx512_core_amx) {
        create_kernel();
    }

    void tile_configure(const char *palette) const { (*this)(palette); }

private:
    void generate() override {
        preamble();

        ldtilecfg(ptr[abi_param1]);

        postamble();
    }
};

struct jit_amx_tilerelease_t : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_amx_tilecfg_t)

    // TODO: Need to check status
    jit_amx_tilerelease_t()
        : jit_generator(nullptr, MAX_CODE_SIZE, true, avx512_core_amx) {
        create_kernel();
    }

    void tile_release() const { (*this)(); }

private:
    void generate() override {
        preamble();

        tilerelease();

        postamble();
    }
};

status_t amx_tile_configure(const char palette[AMX_PALETTE_SIZE]) {
    static const jit_amx_tilecfg_t tilecfg;
    tilecfg.tile_configure(palette);
    return status::success;
};

status_t amx_tile_release() {
    static const jit_amx_tilerelease_t tilerls;
    tilerls.tile_release();
    return status::success;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
