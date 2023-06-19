// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transformer_configuration.hpp"

namespace transformation_sample {

class TransformerConfigurationLoader {
public:
    class JSONParsingException : public std::runtime_error {
    public:
        JSONParsingException(const std::string& message) : std::runtime_error(message) {}
    };

    virtual ~TransformerConfigurationLoader() = default;
    virtual TransformerConfiguration parse_configuration(const std::string& path_to_file) const = 0;
    virtual TransformerConfiguration parse_configuration(std::istream& stream_with_json) const = 0;
};

}  // namespace transformation_sample
