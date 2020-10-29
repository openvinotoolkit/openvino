// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief mock file for header file for IFormatParser
 * \file mock_iformat_parser.hpp
 */
#pragma once

#include <gmock/gmock.h>

#include "ie_icnn_network.hpp"
#include <ie_cnn_net_reader_impl.h>
#include <parsers.h>
#include "pugixml.hpp"

struct MockIFormatParser : public InferenceEngine::details::IFormatParser {
    public:
    MOCK_METHOD1(Parse, InferenceEngine::details::CNNNetworkImplPtr(pugi::xml_node &));

    MOCK_METHOD1(SetWeights, void(const InferenceEngine::TBlob<uint8_t>::Ptr &));
};

