// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_load_network_case.hpp"

//------------------------------------------------------------------------------
// Implementation of methods of class MyriadLoadNetworkTestCase
//------------------------------------------------------------------------------

void MyriadLoadNetworkTestCase::SetUp() {
    try {
        ie = std::make_shared<InferenceEngine::Core>();
    }
    catch (...) {
        std::cerr << "create core error";
    }

    cnnNetwork = ie->ReadNetwork(convReluNormPoolFcModelFP16.model_xml_str,
                                 convReluNormPoolFcModelFP16.weights_blob);
}

void MyriadLoadNetworkTestCase::LoadNetwork() {
    ASSERT_NO_THROW(ie->LoadNetwork(cnnNetwork, "MYRIAD"));
}

bool MyriadLoadNetworkTestCase::IsDeviceAvailable(std::string device_name) {
    auto act_devices = getDevicesList(NC_ANY_PROTOCOL, NC_ANY_PLATFORM, X_LINK_UNBOOTED);
    return std::find(act_devices.begin(), act_devices.end(), device_name) != act_devices.end();
}
