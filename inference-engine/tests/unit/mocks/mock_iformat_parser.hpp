// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief mock file for header file for IFormatParser
 * \file mock_iformat_parser.hpp
 */
#pragma once

#include "ie_icnn_network.hpp"
#include <gmock/gmock-generated-function-mockers.h>
#include <inference_engine/ie_cnn_net_reader_impl.h>
#include <inference_engine/parsers.h>
#include "pugixml.hpp"

struct MockIFormatParser : public InferenceEngine::details::IFormatParser {
    public:
    MOCK_METHOD1(Parse, InferenceEngine::details::CNNNetworkImplPtr(pugi::xml_node &));

    MOCK_METHOD1(SetWeights, void(const InferenceEngine::TBlob<uint8_t>::Ptr &));

    MOCK_METHOD2(CopyBlobsByName, void(void*, std::string));
};

