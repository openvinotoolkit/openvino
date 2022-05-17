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

#include "brgemm_cell_common_utils.hpp"

#include "cpu/x64/amx_tile_configure.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

void amx_tile_configuration_loader_t::operator()(
        const char *requested_cfg_addr) {
    if (current_cfg_addr != requested_cfg_addr) {
        amx_tile_configure(requested_cfg_addr);
        current_cfg_addr = requested_cfg_addr;
    }
}

amx_tile_configuration_loader_t::~amx_tile_configuration_loader_t() {
    if (current_cfg_addr) { amx_tile_release(); }
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
