// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <fstream>

std::map<std::string, std::string> parseConfigFile(char comment = '#');
