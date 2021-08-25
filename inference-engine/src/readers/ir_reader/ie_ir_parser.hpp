// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef IR_READER_V10
#include <ie_ngraph_utils.hpp>
#include <ngraph/node.hpp>
#include <ngraph/op/util/sub_graph_base.hpp>
#include <ngraph/op/util/variable.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/opsets/opset5.hpp>
#endif  // IR_READER_V10

#include <cpp/ie_cnn_network.h>
#include <ie_blob.h>
#include <ie_iextension.h>
#include <xml_parse_utils.h>

#include <algorithm>
#include <cctype>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace InferenceEngine {

class IParser {
public:
    using Ptr = std::shared_ptr<IParser>;
    virtual ~IParser() = default;
    virtual CNNNetwork parse(
        const pugi::xml_node& root, const Blob::CPtr& weights) = 0;
};

class IRParser {
public:
    explicit IRParser(size_t version);
    IRParser(size_t version, const std::vector<InferenceEngine::IExtensionPtr>& exts);
    CNNNetwork parse(const pugi::xml_node& root, const Blob::CPtr& weights);
    virtual ~IRParser() = default;

private:
    IParser::Ptr parser;
};

class CNNParser : public IParser {
public:
    CNNParser() = default;
    CNNNetwork parse(
        const pugi::xml_node& root, const Blob::CPtr& weights) override;
};

#ifdef IR_READER_V10
class V10Parser : public IParser {
public:
    explicit V10Parser(const std::vector<IExtensionPtr>& exts);

    CNNNetwork parse(
        const pugi::xml_node& root, const Blob::CPtr& weights) override;

private:
    void parsePreProcess(
        CNNNetwork& network, const pugi::xml_node& root, const Blob::CPtr& weights);

    const std::vector<IExtensionPtr> _exts;
};

#endif  // IR_READER_V10

}  // namespace InferenceEngine
