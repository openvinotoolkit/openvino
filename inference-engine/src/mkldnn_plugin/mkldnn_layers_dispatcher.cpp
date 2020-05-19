// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_layers_dispatcher.hpp"
#include <details/ie_exception.hpp>
#include "nodes/list.hpp"
#include <memory>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

void addDefaultExtensions(MKLDNNExtensionManager::Ptr mngr) {
    if (!mngr)
        THROW_IE_EXCEPTION << "Cannot add default extensions! Extension manager is empty.";

    auto defaultExtensions = std::make_shared<Extensions::Cpu::MKLDNNExtensions>();
    mngr->AddExtension(defaultExtensions);
}

}  // namespace MKLDNNPlugin
