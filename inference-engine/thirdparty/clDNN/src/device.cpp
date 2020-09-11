/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "device_impl.h"
#include "gpu/ocl_builder.h"

#include <map>
#include <string>

namespace cldnn {

device device::create_default() {
    device_query query;
    auto devices = query.get_available_devices();
    // ToDo Maybe some heuristic should be added to decide what device is the default? (i.e number of EUs)
    return devices.begin()->second;
}

device_info device::get_info() const {
    return _impl->get_info().convert_to_api();
}

void device::retain() {
    _impl->add_ref();
}
void device::release() {
    _impl->release();
}

// --- device query ---
device_query::device_query(void* clcontext, void* user_device)
    : _impl(new device_query_impl(clcontext, user_device)) {
}

std::map<std::string, device> device_query::get_available_devices() const {
    std::map<std::string, device> ret;
    auto device_list = _impl->get_available_devices();
    for (auto dev : device_list) {
        ret.insert({ dev.first, device(dev.second.detach())});
    }
    return ret;
}

void device_query::retain() {
    _impl->add_ref();
}
void device_query::release() {
    _impl->release();
}

// --- device query impl ---
device_query_impl::device_query_impl(void* user_context, void* user_device) {
    gpu::ocl_builder builder;
    _available_devices = builder.get_available_devices(user_context, user_device);
}
}  // namespace cldnn
