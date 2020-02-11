// Copyright (C) 2018-2020 Intel Corporation
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

class BackEnd final {
public:
    using Ptr = std::shared_ptr<BackEnd>;

    CompiledGraph::Ptr build(
            const Model& model,
            const std::vector<ie::CNNLayerPtr>& allLayers);

    void dumpModel(
            const Model& model,
            const std::string& postfix = std::string());

private:
    void serialize(
            const Model& model,
            std::vector<char>& blob,
            std::pair<char*, size_t>& blobHeader,
            int& numActiveStages);

    void getMetaData(
            const Model& model,
            const std::vector<ie::CNNLayerPtr>& allLayers,
            GraphMetaInfo& graphMetaData);

    void extractDataInfo(
            const Model& model,
            DataInfo& inputInfo,
            DataInfo& outputInfo);

    void dumpModelToDot(
            const Model& model,
            const std::string& fileName);
};

}  // namespace vpu
