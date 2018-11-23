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

#pragma once

#include <string>
#include <mutex>
#include "auto_tuner.h"
#include "kernel_selector_common.h"

namespace kernel_selector 
{
    // SKL GT4e
    void tuning_cache_193B(tuning_data&);
    void tuning_cache_193B_B1_B16(tuning_data&);
    void tuning_cache_193B_B8(tuning_data&);
    void tuning_cache_193B_B32_B64(tuning_data&);
    //SKL GT2
    void tuning_cache_1912(tuning_data&);
    void tuning_cache_1912_B1_B16(tuning_data&);
    void tuning_cache_1912_B8(tuning_data&);
    void tuning_cache_1912_B32_B64(tuning_data&);
    //KBL GT3e
    void tuning_cache_5927(tuning_data&);
    void tuning_cache_5927_B1(tuning_data&);
    //ICL GT2
    void tuning_cache_8A52(tuning_data&);
    void tuning_cache_8A52_B1_B16(tuning_data&);
    //APL 10W
    void tuning_cache_5A84(tuning_data&);
    // Device ID for APL E3930.
    void tuning_cache_5A85(tuning_data&);

    class auto_tuner_offline
    {
    private:
        static std::shared_ptr<auto_tuner_offline> instance;
        static std::mutex mutex;
        auto_tuner_offline() = delete;
        // this is singleton implementation, if called twice with different parameter, 
        // second call param will be ignored
        auto_tuner_offline(const std::string& hw_id);
        tuning_data t_data;

        const std::map<std::string, void(*)(tuning_data&)> sku_cache_fillers
        {
            { "0x193B" , tuning_cache_193B },
            { "0x1912" , tuning_cache_1912 },
            { "0x5927" , tuning_cache_5927 },
            { "0x8A52" , tuning_cache_8A52 },
            { "0x5A84" , tuning_cache_5A84 },
            { "0x5A85" , tuning_cache_5A84 },
            { "0x3184" , tuning_cache_5A84 },
        };

    public:
        static std::shared_ptr<auto_tuner_offline> get_instance(const std::string& hw_id);
        tuning_data get_tuning_data() const { return t_data; }
   };
}