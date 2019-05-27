// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/backend/backend.hpp>

#include <memory>
#include <string>
#include <vector>
#include <assert.h>

#include <vpu/compile_env.hpp>
#include <vpu/utils/file_system.hpp>
#include <vpu/utils/io.hpp>

namespace vpu {

void BackEnd::extractDataInfo(
        const Model::Ptr& model,
        DataInfo& inputInfo,
        DataInfo& outputInfo) {
    for (const auto& data : model->datas()) {
        if (DataUsage::Input == data->usage()) {
            IE_ASSERT(inputInfo.offset.count(data->name()) == 0);

            auto ioBufferOffset = data->attrs().get<int>("ioBufferOffset");
            IE_ASSERT(ioBufferOffset + data->totalByteSize() <= inputInfo.totalSize);

            inputInfo.offset[data->name()] = ioBufferOffset;
        } else if (DataUsage::Output == data->usage()) {
            IE_ASSERT(outputInfo.offset.count(data->name()) == 0);

            auto ioBufferOffset = data->attrs().get<int>("ioBufferOffset");
            IE_ASSERT(ioBufferOffset + data->totalByteSize() <= outputInfo.totalSize);

            outputInfo.offset[data->name()] = ioBufferOffset;
        }
    }
}

CompiledGraph::Ptr BackEnd::build(
        const Model::Ptr& model,
        const std::vector<ie::CNNLayerPtr>& allLayers) {
    auto compiledGraph = std::make_shared<CompiledGraph>();

    compiledGraph->networkName = model->name();
    compiledGraph->networkBatch = model->batchSize();

    auto usedMemory = model->attrs().get<UsedMemory>("usedMemory");
    compiledGraph->inputBufSize = usedMemory.input;
    compiledGraph->outputBufSize = usedMemory.output;

    compiledGraph->inputInfo.totalSize  = usedMemory.input;
    compiledGraph->outputInfo.totalSize = usedMemory.output;

    extractDataInfo(model, compiledGraph->inputInfo, compiledGraph->outputInfo);

    serialize(model, compiledGraph->blob, compiledGraph->blobHeader, compiledGraph->numActiveStages);
    getMetaData(model, allLayers, compiledGraph->stagesMeta);

    return compiledGraph;
}

void BackEnd::dumpModel(
        const Model::Ptr& model,
        const std::string& postfix) {
#ifdef NDEBUG
    (void)model;
    (void)postfix;
#else
    std::string fileName;

    if (auto envVar = std::getenv("IE_VPU_DUMP_INTERNAL_GRAPH_DIRECTORY")) {
        auto modelName = model->name();

        // Replace "bad" characters
        for (auto& ch : modelName) {
            if (!std::isalnum(ch)) {
                ch = '_';
            }
        }

        std::ostringstream ostr;
        ostr << envVar << "/" << "vpu_graph_" << std::setw(2) << std::setfill('0') << model->attrs().get<int>("index") << "_" << modelName;

        fileName = ostr.str();
    } else if (auto envVar = std::getenv("IE_VPU_DUMP_INTERNAL_GRAPH_FILE_NAME")) {
        fileName = envVar;
    }

    if (fileName.empty()) {
        return;
    }

    if (!postfix.empty()) {
        if (auto envVar = std::getenv("IE_VPU_DUMP_ALL_PASSES")) {
            if (std::stoi(envVar) == 0) {
                return;
            }

            fileName = formatString("%s_%s", fileNameNoExt(fileName), postfix);
        } else {
            return;
        }
    }

    auto dotFileName = formatString("%s.dot", fileNameNoExt(fileName));
    dumpModelToDot(model, dotFileName);
#endif
}

}  // namespace vpu
