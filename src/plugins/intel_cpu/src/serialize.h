// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <functional>
#include <iostream>

#include "cpp/ie_cnn_network.h"
#include "extension_mngr.h"

namespace ov {
namespace intel_cpu {

class ModelSerializer {
public:
    ModelSerializer(std::ostream& ostream, ExtensionManager::Ptr extensionManager);
    void operator<<(std::pair<const std::shared_ptr<ov::Model>, const std::shared_ptr<const ov::Model>>& models);
private:
    std::ostream& _ostream;
    ExtensionManager::Ptr _extensionManager;
};

class ModelDeserializer {
public:
    typedef std::function<std::shared_ptr<ov::Model>(const std::string&, const ov::Tensor&)> model_builder;
    ModelDeserializer(std::istream& istream, model_builder fn);
    void operator>>(std::pair<std::shared_ptr<ov::Model>, std::shared_ptr<ov::Model>>& models);

private:
    std::istream& _istream;
    model_builder _model_builder;
};

}   // namespace intel_cpu
}   // namespace ov
