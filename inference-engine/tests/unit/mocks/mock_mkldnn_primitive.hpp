// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <gmock/gmock.h>
#include "mkldnn/mkldnn_generic_primitive.hpp"


class MockMKLDNNGenericPrimitive : public InferenceEngine::MKLDNNPlugin::IMKLDNNGenericPrimitive {
 public:
    // Add the following two lines to the mock class.
    MOCK_METHOD0(die, void());
    ~MockMKLDNNGenericPrimitive () { die(); }

    MOCK_QUALIFIED_METHOD0(Release, noexcept, void ());

    MOCK_QUALIFIED_METHOD2(SetMemory, noexcept, void (const std::vector<InferenceEngine::MKLDNNPlugin::MKLDNNPrimitiveMemory>& input,
                           const InferenceEngine::MKLDNNPlugin::MKLDNNPrimitiveMemory& output));
    MOCK_QUALIFIED_METHOD0(GetSupportedFormats, noexcept, std::vector<InferenceEngine::MKLDNNPlugin::MKLDNNGenericFormats> ());
    MOCK_QUALIFIED_METHOD0(Execute,  noexcept, void () );
};

