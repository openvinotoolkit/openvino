// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <set>
#include <vector>
#include <utility>

#include <ie_layers.h>

#include <vpu/graph_transformer.hpp>
#include <vpu/model/model.hpp>

namespace vpu {

namespace ie = InferenceEngine;

class BackEnd final : public std::enable_shared_from_this<BackEnd> {
public:
    using Ptr = std::shared_ptr<BackEnd>;

    CompiledGraph::Ptr build(
            const Model::Ptr& model,
            const std::vector<ie::CNNLayerPtr>& allLayers);

    void dumpModel(
            const Model::Ptr& model,
            const std::string& postfix = std::string());

private:
    void serialize(
            const Model::Ptr& model,
            std::vector<char>& blob,
            std::pair<char*, size_t>& blobHeader,
            int& numActiveStages);

    void getMetaData(
            const Model::Ptr& model,
            const std::vector<ie::CNNLayerPtr>& allLayers,
            std::vector<StageMetaInfo>& metaData);

    void extractDataInfo(
            const Model::Ptr& model,
            DataInfo& inputInfo,
            DataInfo& outputInfo);

#ifndef NDEBUG
    void dumpModelToDot(
            const Model::Ptr& model,
            const std::string& fileName);
#endif
};

}  // namespace vpu
