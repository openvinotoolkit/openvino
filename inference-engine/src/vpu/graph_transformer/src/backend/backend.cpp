// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/backend/backend.hpp>

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <iomanip>

#include <vpu/configuration/options/dump_internal_graph_file_name.hpp>
#include <vpu/configuration/options/dump_all_passes_directory.hpp>
#include <vpu/configuration/options/dump_all_passes.hpp>

#include <vpu/compile_env.hpp>
#include <vpu/utils/file_system.hpp>
#include <vpu/utils/io.hpp>

namespace vpu {

void BackEnd::extractDataInfo(
        const Model& model,
        DataInfo& inputInfo,
        DataInfo& outputInfo) {
    for (const auto& data : model->datas()) {
        if (DataUsage::Input == data->usage()) {
            IE_ASSERT(inputInfo.offset.count(data->name()) == 0);

            auto ioBufferOffset = data->attrs().get<int>("ioBufferOffset");
            IE_ASSERT(ioBufferOffset + data->totalByteSize() <= inputInfo.totalSize);

            inputInfo.descFromPlugin[data->name()] = data->desc().toTensorDesc();
            inputInfo.offset[data->name()] = ioBufferOffset;
        } else if (DataUsage::Output == data->usage()) {
            IE_ASSERT(outputInfo.offset.count(data->name()) == 0);

            auto ioBufferOffset = data->attrs().get<int>("ioBufferOffset");
            IE_ASSERT(ioBufferOffset + data->totalByteSize() <= outputInfo.totalSize);

            outputInfo.descFromPlugin[data->name()] = data->desc().toTensorDesc();
            outputInfo.offset[data->name()] = ioBufferOffset;
        }
    }
}

CompiledGraph::Ptr BackEnd::build(
        const Model& model,
        const std::vector<ie::CNNLayerPtr>& allLayers) {
    auto compiledGraph = std::make_shared<CompiledGraph>();

    compiledGraph->networkName = model->name();
    compiledGraph->networkBatch = model->batchSize();

    auto usedMemory = model->attrs().get<UsedMemory>("usedMemory");
    compiledGraph->inputBufSize = usedMemory.input;
    compiledGraph->outputBufSize = usedMemory.output;

    const auto& resources = model->attrs().get<Resources>("resources");
    compiledGraph->numShaves = checked_cast<std::uint32_t>(resources.numSHAVEs);
    compiledGraph->numSlices = checked_cast<std::uint32_t>(resources.numCMXSlices);
    compiledGraph->numExecutors = checked_cast<std::uint32_t>(resources.numExecutors);

    compiledGraph->inputInfo.totalSize  = usedMemory.input;
    compiledGraph->outputInfo.totalSize = usedMemory.output;

    extractDataInfo(model, compiledGraph->inputInfo, compiledGraph->outputInfo);

    serialize(model, compiledGraph->blob, compiledGraph->blobHeader, compiledGraph->numActiveStages);
    getMetaData(model, allLayers, compiledGraph->graphMeta);

    return compiledGraph;
}

void BackEnd::dumpModel(
        const Model& model,
        const std::string& postfix) {
    const auto replaceBadCharacters = [](std::string str) {
        for (auto& ch : str) {
            if (!std::isalnum(ch)) {
                ch = '_';
            }
        }
        return str;
    };

    const auto& env = CompileEnv::get();

    std::string fileName;

    if (!env.config.get<DumpInternalGraphFileNameOption>().empty()) {
        fileName = fileNameNoExt(env.config.get<DumpInternalGraphFileNameOption>());
    } else if (!env.config.get<DumpAllPassesDirectoryOption>().empty()) {
        fileName = formatString(
            "%s/vpu_graph_%f%f%i_%s",
            env.config.get<DumpAllPassesDirectoryOption>(),
            std::setw(2), std::setfill('0'),
            model->attrs().get<int>("index"),
            replaceBadCharacters(model->name()));
    } else {
        return;
    }

    if (!postfix.empty()) {
        if (!env.config.get<DumpAllPassesOption>()) {
            return;
        }

        fileName = formatString("%s_%s", fileName, replaceBadCharacters(postfix));
    }

    const auto dotFileName = formatString("%s.dot", fileName);
    dumpModelToDot(model, dotFileName);
}

}  // namespace vpu
