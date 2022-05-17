/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef NGEN_LEVEL_ZERO_HPP
#define NGEN_LEVEL_ZERO_HPP

#include "ngen_config.hpp"

#include <ze_api.h>

#include <sstream>

#include "ngen_elf.hpp"
#include "ngen_interface.hpp"

namespace ngen {

// Exceptions.
class level_zero_error : public std::runtime_error {
public:
    level_zero_error(ze_result_t status_ = ZE_RESULT_SUCCESS) : std::runtime_error("A Level Zero error occurred."), status(status_) {}
protected:
    ze_result_t status;
};

// OpenCL program generator class.
template <HW hw>
class LevelZeroCodeGenerator : public ELFCodeGenerator<hw>
{
public:
    inline ze_module_handle_t getModule(ze_context_handle_t context, ze_device_handle_t device, const std::string &options = "");
    static void getHWInfo(ze_context_handle_t context, ze_device_handle_t device, HW &_hw, int &steppingID);
    static inline HW detectHW(ze_context_handle_t context, ze_device_handle_t device);
};

#define NGEN_FORWARD_LEVEL_ZERO(hw) NGEN_FORWARD_ELF(hw)

namespace detail {

static inline void handleL0(ze_result_t result)
{
    if (result != ZE_RESULT_SUCCESS)
        throw level_zero_error{result};
}

}; /* namespace detail */

template <HW hw>
ze_module_handle_t LevelZeroCodeGenerator<hw>::getModule(ze_context_handle_t context, ze_device_handle_t device, const std::string &options)
{
    using super = ELFCodeGenerator<hw>;

    auto binary = super::getBinary();

    ze_module_desc_t moduleDesc = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        nullptr,
        ZE_MODULE_FORMAT_NATIVE,
        binary.size(),
        binary.data(),
        options.c_str(),
        nullptr
    };

    ze_module_handle_t module;
    detail::handleL0(zeModuleCreate(context, device, &moduleDesc, &module, nullptr));

    if (module == nullptr)
        throw level_zero_error{};

    return module;
}

template <HW hw>
void LevelZeroCodeGenerator<hw>::getHWInfo(ze_context_handle_t context, ze_device_handle_t device, HW &_hw, int &steppingID)
{
    static const uint8_t dummySPV[] = {0x03, 0x02, 0x23, 0x07, 0x00, 0x00, 0x01, 0x00, 0x0E, 0x00, 0x06, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00, 0x04, 0x00, 0x00, 0x00, 0x11, 0x00, 0x02, 0x00, 0x06, 0x00, 0x00, 0x00, 0x0B, 0x00, 0x05, 0x00, 0x01, 0x00, 0x00, 0x00, 0x4F, 0x70, 0x65, 0x6E, 0x43, 0x4C, 0x2E, 0x73, 0x74, 0x64, 0x00, 0x00, 0x0E, 0x00, 0x03, 0x00, 0x02, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0F, 0x00, 0x04, 0x00, 0x06, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x5F, 0x00, 0x00, 0x00, 0x07, 0x00, 0x07, 0x00, 0x06, 0x00, 0x00, 0x00, 0x6B, 0x65, 0x72, 0x6E, 0x65, 0x6C, 0x5F, 0x61, 0x72, 0x67, 0x5F, 0x74, 0x79, 0x70, 0x65, 0x2E, 0x5F, 0x2E, 0x00, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x70, 0x8E, 0x01, 0x00, 0x05, 0x00, 0x04, 0x00, 0x05, 0x00, 0x00, 0x00, 0x65, 0x6E, 0x74, 0x72, 0x79, 0x00, 0x00, 0x00, 0x13, 0x00, 0x02, 0x00, 0x02, 0x00, 0x00, 0x00, 0x21, 0x00, 0x03, 0x00, 0x03, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x36, 0x00, 0x05, 0x00, 0x02, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0xF8, 0x00, 0x02, 0x00, 0x05, 0x00, 0x00, 0x00, 0xFD, 0x00, 0x01, 0x00, 0x38, 0x00, 0x01, 0x00};
    ze_module_desc_t moduleDesc = {
        ZE_STRUCTURE_TYPE_MODULE_DESC,
        nullptr,
        ZE_MODULE_FORMAT_IL_SPIRV,
        sizeof(dummySPV),
        dummySPV,
        nullptr,
        nullptr
    };

    ze_module_handle_t module;
    detail::handleL0(zeModuleCreate(context, device, &moduleDesc, &module, nullptr));

    if (module == nullptr)
        throw level_zero_error{};

    std::vector<uint8_t> binary;
    size_t binarySize;

    detail::handleL0(zeModuleGetNativeBinary(module, &binarySize, nullptr));
    binary.resize(binarySize);
    detail::handleL0(zeModuleGetNativeBinary(module, &binarySize, binary.data()));
    detail::handleL0(zeModuleDestroy(module));

    ELFCodeGenerator<hw>::getHWInfo(binary, _hw, steppingID);
}

template <HW hw>
HW LevelZeroCodeGenerator<hw>::detectHW(ze_context_handle_t context, ze_device_handle_t device)
{
    HW _hw;
    int steppingID;
    getHWInfo(context, device, _hw, steppingID);
    (void)steppingID;
    return _hw;
}

} /* namespace ngen */

#endif
