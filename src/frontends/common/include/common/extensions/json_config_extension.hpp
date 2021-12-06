// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <iostream>
#include <type_traits>

#include "common/extensions/decoder_transformation_extension.hpp"
#include "common/frontend_defs.hpp"
#include "nlohmann/json.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {

/// \brief Reads MO config file and delegate transformation functionality to specified transformation ID
/// specified in the config.
class FRONTEND_API JsonConfigExtension : public DecoderTransformationExtension {
public:
    explicit JsonConfigExtension(const std::string& config_path);
    ~JsonConfigExtension() override;

protected:
    std::vector<Extension::Ptr> m_loaded_extensions;
    std::vector<std::pair<std::shared_ptr<DecoderTransformationExtension>, nlohmann::json>> m_target_extensions;
};
}  // namespace frontend
}  // namespace ov
