// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "gna2-model-api.h"
#include "gna_device.hpp"

void WriteInputAndOutputTextGNAImpl(const Gna2Model & gnaModel, const std::string dumpFolderNameGNA, const std::string refFolderName);

void DumpGna2Model(const Gna2Model& gnaModel, const std::string dumpFolderNameGNA, bool dumpData, const GnaAllAllocations& allAllocations,
    std::string modeOfOperation);
