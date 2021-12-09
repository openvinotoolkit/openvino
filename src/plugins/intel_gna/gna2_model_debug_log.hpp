// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if GNA_LIB_VER == 2

#include <string>

#include "gna2-model-api.h"

void WriteInputAndOutputTextGNAImpl(const Gna2Model & gnaModel, const std::string dumpFolderNameGNA, const std::string refFolderName);
void DumpGna2Model(const Gna2Model& gnaModel, const std::string dumpFolderNameGNA, bool dumpData);

#endif
