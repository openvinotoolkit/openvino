// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <functional>

#include <caseless.hpp>
#include <pugixml.hpp>

#include <vpu/utils/enums.hpp>
#include <vpu/utils/small_vector.hpp>
#include <vpu/frontend/custom_kernel.hpp>

#include <ie_common.h>

namespace vpu {

namespace ie = InferenceEngine;

class CustomLayer final {
public:
    using Ptr = std::shared_ptr<CustomLayer>;
    explicit CustomLayer(std::string configDir, const pugi::xml_node& customLayer);

    std::vector<CustomKernel> kernels() const { return _kernels; }
    std::string layerName() const { return _layerName; }
    std::map<int, CustomDataFormat> inputs() { return _inputs; }
    std::map<int, CustomDataFormat> outputs() { return _outputs; }

    static ie::details::caseless_map<std::string, std::vector<CustomLayer::Ptr>> loadFromFile(
                const std::string& configFile,
                bool canBeMissed = false);

    bool meetsWhereRestrictions(const std::map<std::string, std::string>& params) const;
    static bool isLegalSizeRule(const std::string& rule, std::map<std::string, std::string> layerParams);
    static CustomDataFormat formatFromLayout(const InferenceEngine::Layout& layout);

private:
    std::string _configDir;
    std::string _layerName;
    std::map<std::string, std::string> _whereParams;

    std::vector<CustomKernel> _kernels;

    std::map<int, CustomDataFormat> _inputs;
    std::map<int, CustomDataFormat> _outputs;
};

};  // namespace vpu
