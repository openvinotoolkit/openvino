// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

namespace transformation_sample {

struct GNAConfiguration {
    std::string target_version;
    std::string gna_precision;
    std::string pwl_max_error_precentage;
};

struct TransformerConfiguration {
    std::map<std::string, std::string> gna_configuration;
    std::vector<std::string> transformations_names;
};

}  // namespace transformation_sample
