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


#include "auto_tuner.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <string>
#include "istreamwrapper.h"
#include "stringbuffer.h"
#include "prettywriter.h"
#include <memory>
#include <utility>
#include <tuple>

namespace kernel_selector {
std::tuple<std::string, int> AutoTuner::LoadKernelOnline(const TuningMode tuningMode,
                                                         const std::string& cacheFilePath,
                                                         const uint32_t computeUnitsCount,
                                                         const std::string& hash) {
    std::lock_guard<std::mutex> lock(mutex);
    rapidjson::Document cacheData;
    std::ifstream tuningFile(cacheFilePath);
    if (tuningFile && tuningFile.good()) {
        rapidjson::IStreamWrapper isw{tuningFile};
        cacheData.ParseStream(isw);
    } else {  // Tuning file doesn't exist
        if (tuningMode == TuningMode::TUNING_USE_CACHE) {
            throw std::runtime_error("Tuning file: " + cacheFilePath +
                                     " could not be read! Must provide a valid cache file in USE_CACHE mode.");
        }

        // Create a new tuning file and write the versions
        std::ofstream newTuningFile(cacheFilePath, std::ofstream::out);
    }
    tuningFile.close();

    onlineCache = std::make_shared<rapidjson::Document>(std::move(cacheData));

    // Tuning file is loaded
    auto computeUnitsStr = std::to_string(computeUnitsCount);
    auto defaultComputeUnitsCount = "24";
    if (!onlineCache->IsNull()) {
        auto cacheObject = onlineCache->GetObject();
        if (onlineCache->HasMember(computeUnitsStr.c_str())) {
            if (cacheObject[computeUnitsStr.c_str()].HasMember(hash.c_str())) {
                const rapidjson::Value& prog = cacheObject[computeUnitsStr.c_str()][hash.c_str()];
                return std::make_tuple(prog[0].GetString(), prog[1].GetInt());
            }
        } else if (onlineCache->HasMember(defaultComputeUnitsCount)) {
            if (cacheObject[defaultComputeUnitsCount].HasMember(hash.c_str())) {
                const rapidjson::Value& prog = cacheObject[defaultComputeUnitsCount][hash.c_str()];
                return std::make_tuple(prog[0].GetString(), prog[1].GetInt());
            }
        }
    }
    return std::make_pair("", 0);
}

void AutoTuner::StoreKernel(const std::string& cacheFilePath,
                            const std::string& hash,
                            std::string implementationName,
                            const int tuneIndex,
                            const uint32_t computeUnitsCount) {
    std::lock_guard<std::mutex> lock(mutex);
    auto computeUnitsStr = std::to_string(computeUnitsCount);
    rapidjson::Document::AllocatorType& allocator = onlineCache->GetAllocator();
    rapidjson::Value dataArray(rapidjson::kArrayType);
    rapidjson::Value hashStr(rapidjson::kStringType);
    hashStr.Set(hash.c_str(), allocator);
    dataArray.PushBack(rapidjson::Value().Set(implementationName.c_str(), allocator), allocator);
    dataArray.PushBack(rapidjson::Value().SetInt(tuneIndex), allocator);

    rapidjson::Value newVal(rapidjson::kObjectType);
    newVal.SetObject();
    if (onlineCache->IsNull()) {
        onlineCache->Parse("{}");
    }
    if (!onlineCache->HasMember(computeUnitsStr.c_str())) {
        onlineCache->AddMember(rapidjson::Value(computeUnitsStr.c_str(), allocator), newVal, allocator);
    }

    auto cache = onlineCache->GetObject();
    cache[computeUnitsStr.c_str()].AddMember(hashStr, dataArray, allocator);

    std::ofstream cachedKernelsFile(cacheFilePath);
    rapidjson::StringBuffer buffer(0, 1024);
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    writer.SetFormatOptions(rapidjson::PrettyFormatOptions::kFormatSingleLineArray);
    onlineCache->Accept(writer);
    auto temp = buffer.GetString();
    cachedKernelsFile << temp;
    cachedKernelsFile.close();
}

std::tuple<std::string, int> AutoTuner::LoadKernelOffline(std::shared_ptr<rapidjson::Document> deviceCache,
                                                          const std::string& hash) {
    if (!deviceCache->IsNull()) {
        auto cache = deviceCache->GetObject();
        if (deviceCache->HasMember(hash.c_str())) {
            const rapidjson::Value& prog = cache[hash.c_str()];
            return std::make_tuple(prog[0].GetString(), prog[1].GetInt());
        }
    }
    return std::make_tuple("", 0);
}
}  // namespace kernel_selector
