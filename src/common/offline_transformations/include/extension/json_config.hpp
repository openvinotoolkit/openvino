// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <type_traits>

#include "openvino/frontend/extension/decoder_transformation.hpp"

namespace ov {
namespace frontend {

/// \brief JsonConfigExtension reads MO config file and delegate transformation functionality to specified
/// transformation ID specified in the config.
class JsonConfigExtension : public DecoderTransformationExtension {
public:
    explicit JsonConfigExtension(const std::string& config_path);

protected:
    std::vector<Extension::Ptr> m_loaded_extensions;
    std::vector<std::pair<DecoderTransformationExtension::Ptr, std::string>> m_target_extensions;
};
}  // namespace frontend
}  // namespace ov
