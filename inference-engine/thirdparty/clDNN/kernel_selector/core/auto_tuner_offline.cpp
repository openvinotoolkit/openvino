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

#include "auto_tuner.h"
#include "auto_tuner_offline.h"
namespace kernel_selector 
{
    std::shared_ptr<auto_tuner_offline> auto_tuner_offline::instance = 0;
    std::mutex auto_tuner_offline::mutex;

    auto_tuner_offline::auto_tuner_offline(const std::string& hw_id)
    {
        std::string temp_hw_id = hw_id;
        // TODO: this is temporary solution of cases where user has non-tuned configuration. needs to implement better logic
        // i.e. create table with number of eu's configuration that will point to common cache.
        if (sku_cache_fillers.count(hw_id) == 0)
            temp_hw_id = "0x1912";
        sku_cache_fillers.at(temp_hw_id)(t_data);
    }

    std::shared_ptr<auto_tuner_offline> auto_tuner_offline::get_instance(const std::string& hw_id)
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (instance == nullptr)
        {
            instance = std::make_shared<auto_tuner_offline>(auto_tuner_offline(hw_id));
        }
        return instance;
    }
}