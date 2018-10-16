/*
// Copyright (c) 2016 Intel Corporation
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
#pragma once
#include <cstdint>
#include "api/CPP/engine.hpp"

namespace cldnn { namespace gpu {

class gpu_toolkit;
struct engine_info_internal : cldnn::engine_info
{
    #ifdef GPU_CONFIGURATION
        #undef GPU_CONFIGURATION
    #endif
    #ifdef GPU_MODEL
        #undef GPU_MODEL
    #endif
    #ifdef GPU_ARCHITECTURE
        #undef GPU_ARCHITECTURE
    #endif


    enum configurations
    {
        #define GPU_CONFIGURATION(enum_name, enum_value) enum_name = enum_value,
        #define GPU_MODEL(enum_name, enum_value)
        #define GPU_ARCHITECTURE(enum_name, enum_value)
        #include "gpu_enums.inc"
        #undef GPU_CONFIGURATION
        #undef GPU_MODEL
        #undef GPU_ARCHITECTURE
    };

    

    enum models
    {
        #define GPU_CONFIGURATION(enum_name, enum_value)
        #define GPU_MODEL(enum_name, enum_value) enum_name = enum_value,
        #define GPU_ARCHITECTURE(enum_name, enum_value)
        #include "gpu_enums.inc"
        #undef GPU_CONFIGURATION
        #undef GPU_MODEL
        #undef GPU_ARCHITECTURE
    };

    

    enum architectures
    {
        #define GPU_CONFIGURATION(enum_name, enum_value)
        #define GPU_MODEL(enum_name, enum_value)
        #define GPU_ARCHITECTURE(enum_name, enum_value) enum_name = enum_value,
        #include "gpu_enums.inc"
        #undef GPU_CONFIGURATION
        #undef GPU_MODEL
        #undef GPU_ARCHITECTURE
    };

    #undef GPU_CONFIGURATION


    configurations configuration;
    models model;
    architectures architecture;
    std::string dev_id;
    std::string driver_version;
private:
    friend class gpu_toolkit;
    explicit engine_info_internal(const gpu_toolkit& context);
};

}}
