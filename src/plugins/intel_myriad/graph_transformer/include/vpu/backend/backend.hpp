// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/graph_transformer.hpp>

#include <vpu/model/model.hpp>
#include <vpu/backend/blob_format.hpp>
#include <legacy/ie_layers.h>

#include <string>
#include <memory>
#include <set>
#include <vector>
#include <utility>

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

    int serializeIOInfoSection(
            const Model& model,
            DataUsage dataUsage,
            BlobSerializer& blobSerializer);

    void serializeConstData(
            const Model& model,
            const mv_blob_header& blobHdr,
            std::vector<char>& blob);

    void serializeConstShapes(
            const Model& model,
            const mv_blob_header& blobHdr,
            std::vector<char>& blob);

    void serializeParamsAndResults(
            const Model& model,
            const mv_blob_header& blobHdr,
            std::vector<char>& blob);

    ElfN_Ehdr createElfHeader();

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
