// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ios>
#include <string>
#include <sstream>

#include "gna2-model-api.h"
#include "gna_device.hpp"

void WriteInputAndOutputTextGNAImpl(const Gna2Model & gnaModel, const std::string dumpFolderNameGNA, const std::string refFolderName);

void DumpGna2Model(const Gna2Model& gnaModel, const std::string& dumpFolderNameGNA, bool dumpData, const GnaAllocations& allAllocations,
    const std::string& modeOfOperation);

template <class T>
std::string toHexString(T t) {
    std::ostringstream o;
    o << std::hex << t;
    return o.str();
}
