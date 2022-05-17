/*******************************************************************************
* Copyright 2019-2020 Intel Corporation
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

#ifndef SYCL_DEVICE_INFO_HPP
#define SYCL_DEVICE_INFO_HPP

#include <CL/sycl.hpp>

#include "gpu/compute/device_info.hpp"

namespace dnnl {
namespace impl {
namespace sycl {

class sycl_device_info_t : public gpu::compute::device_info_t {
protected:
    status_t init_device_name(engine_t *engine) override;
    status_t init_arch(engine_t *engine) override;
    status_t init_runtime_version(engine_t *engine) override;
    status_t init_extensions(engine_t *engine) override;
    status_t init_attributes(engine_t *engine) override;
};

} // namespace sycl
} // namespace impl
} // namespace dnnl

#endif // SYCL_DEVICE_INFO_HPP
