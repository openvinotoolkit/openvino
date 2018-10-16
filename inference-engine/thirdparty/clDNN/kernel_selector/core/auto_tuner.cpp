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

#include "auto_tuner.h"
#include "auto_tuner_offline.h"
#include <iostream>
#include <sstream>
#include <fstream>

 
namespace kernel_selector 
{
    std::tuple<std::string, int> AutoTuner::LoadKernelOnline(const TuningMode tuningMode, const std::string& tuningFilePath, const std::string& deviceID, const std::string& driverVersion, const std::string& hostVersion, const std::string& hash)
    {
        std::lock_guard<std::mutex> lock(mutex);

        //First, check if the tuning file has been already loaded to cache
        auto const& tuningFileCache = onlineCache.find(tuningFilePath);
        if (tuningFileCache == onlineCache.end())
        {
            // Load tuning file to cache
            onlineCache[tuningFilePath] = {};

            std::ifstream tuningFile(tuningFilePath);
            std::string cachedDeviceId;
            std::string cachedDriverVersion;
            std::string cachedHostVersion;
            std::string cachedhash;
            std::string cachedkernelName;
            int cachedIndex;
            std::string line;

            if (tuningFile) // Tuning file exists
            {
                // Read device ID
                tuningFile >> cachedDeviceId;
                if (!tuningFile.good() || (cachedDeviceId.compare(deviceID) != 0))
                {
                    throw std::runtime_error("Tuning file bad structure or wrong device ID. Re-generate cache in TUNE_AND_CACHE mode.");
                }

                // Read driver version
                tuningFile >> cachedDriverVersion;
                if (!tuningFile.good() || (cachedDriverVersion.compare(driverVersion) != 0))
                {
                    throw std::runtime_error("Tuning file bad structure or wrong driver version. Re-generate cache in TUNE_AND_CACHE mode.");
                }

                // Read host version
                tuningFile >> cachedHostVersion;
                if (!tuningFile.good() || (cachedHostVersion.compare(hostVersion) != 0))
                {
                    throw std::runtime_error("Tuning file bad structure or wrong host version. Re-generate cache in TUNE_AND_CACHE mode.");
                }

                // Read optimal kernel/config data 
                while (std::getline(tuningFile, line))
                {
                    if (line.empty())
                    {
                        continue;
                    }
                    std::istringstream iss(line);
                    iss >> cachedhash >> cachedkernelName >> cachedIndex;
                    if (iss.fail())
                    {
                        throw std::runtime_error("Tuning file bad structure. Re-generate cache in TUNE_AND_CACHE mode.");
                    }

                    // Update tuning cache 
                    onlineCache[tuningFilePath].td[cachedhash] = std::make_tuple(cachedkernelName, cachedIndex);
                }

                tuningFile.close();
            }
            else // Tuning file doesn't exist
            {
                if (tuningMode == TuningMode::TUNING_USE_CACHE)
                {
                    throw std::runtime_error("Tuning file: " + tuningFilePath + " could not be read! Must provide a valid cache file in USE_CACHE mode.");
                }

                // Create a new tuning file and write the versions
                std::ofstream newTuningFile(tuningFilePath, std::ofstream::out);

                newTuningFile << deviceID << "\n";
                newTuningFile << driverVersion << "\n";
                newTuningFile << hostVersion << "\n";
            }
        }

        // Tuning file is loaded
        auto const& tuningFileData = onlineCache[tuningFilePath];
        auto const& hashData = tuningFileData.td.find(hash);
        if (hashData != tuningFileData.td.end())
        {
            // Tuning data exists for this hash.
            return hashData->second;
        }
        else
        {
            // Tuning data doesn't exists for this hash - on-line tuning is needed.
            return std::make_pair("", 0);
        }
    }

    void AutoTuner::StoreKernel(const std::string& tuningFilePath, const std::string& hash, const std::string& implementationName, const int tuneIndex)
    {
        std::lock_guard<std::mutex> lock(mutex);

        // Add the new tuning data to cache
        onlineCache[tuningFilePath].td[hash] = std::make_tuple(implementationName, tuneIndex);

        // Add the new tuning data to tuning file
        std::ofstream cachedKernelsFile(tuningFilePath, std::ofstream::out | std::ofstream::app);
        if (!cachedKernelsFile.good())
        {
            throw std::runtime_error("Tuning file: " + tuningFilePath + " could not be written!");
        }
        cachedKernelsFile << hash << " ";
        cachedKernelsFile << implementationName << " ";
        cachedKernelsFile << tuneIndex << "\n";
        cachedKernelsFile.close();
    }

    std::tuple<std::string, int> AutoTuner::LoadKernelOffline(const std::string& deviceID, const std::string& hash)
    {
        auto const& deviceCache = auto_tuner_offline::get_instance(deviceID)->get_tuning_data();
        if (deviceCache.td.empty())
        {
            return std::make_pair("", 0);
        }
        auto const& deviceCacheData = deviceCache.td;
        auto const& hashData = deviceCacheData.find(hash);
        if (hashData == deviceCacheData.end())
        {
            return std::make_pair("", 0);
        }
        else
        {
            return hashData->second;
        }
    }
}
