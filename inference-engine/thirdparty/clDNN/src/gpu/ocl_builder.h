/*
// Copyright (c) 2018 Intel Corporation
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
#pragma once
// we want exceptions
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <cl2_wrapper.h>
#include <list>
#include <string>

namespace cldnn {
namespace gpu {
struct configuration;

class ocl_builder {
public:
    explicit ocl_builder(const configuration& config);
    cl::Context get_context() const { return _context; }
    const cl::Device& get_device() const { return _device; }
    cl_platform_id get_platform_id() const { return _platform_id; }
    bool is_user_context() const { return _is_user_context; }

private:
    cl::Context _context;
    cl::Device _device;
    cl_platform_id _platform_id;
    bool _is_user_context;

    void build_device_from_user_context(const configuration& config);
    void build_device(const configuration& config);
    void build_context();
    bool does_device_match_config(const configuration& config, const cl::Device& dev, std::list<std::string>& reasons);
    void build_platform_id();
};

}  // namespace gpu
}  // namespace cldnn
