// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/backend/backend.hpp>

#include <vpu/compile_env.hpp>
#include <vpu/utils/auto_scope.hpp>
#include <vpu/utils/dot_io.hpp>
#include <vpu/utils/file_system.hpp>
#include <vpu/utils/numeric.hpp>

#include <precision_utils.h>
#include <legacy/graph_tools.hpp>
#include <description_buffer.hpp>
#include <xml_parse_utils.h>

#include <ngraph/ops.hpp>
#include <transformations/utils/utils.hpp>

#include <climits>
#include <cstring>
#include <string>
#include <memory>
#include <list>
#include <vector>
#include <array>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <fstream>
#include <utility>
#include <algorithm>
#include <map>
#include <streambuf>
#include <tuple>
#include <sstream>
#include <iomanip>
#include <atomic>

namespace vpu {

struct ModelStagesStat final {
    bool hasHwStage;
    bool hasShaveStage;
    bool hasDmaStage;
};

int BackEnd::serializeIOInfoSection(
        const Model& model,
        DataUsage dataUsage,
        BlobSerializer& blobSerializer) {
    VPU_INTERNAL_CHECK(dataUsage == DataUsage::Input || dataUsage == DataUsage::Output,
        "serializeIOInfoSection was called with {} usage while only {} and {} usages are supported",
        dataUsage, DataUsage::Input, DataUsage::Output);

    int datasNumber = 0;

    for (const auto& data : model->datas()) {
        if (data->usage() != dataUsage) {
            continue;
        }

        if (dataUsage == DataUsage::Input) {
            VPU_INTERNAL_CHECK(data->producerEdge() == nullptr,
                "serializeIOInfoSection failed on input data {}. Input must have no producer but actually it has: {} with type {}",
                data->name(), data->producerEdge()->producer()->name(), data->producerEdge()->producer()->type());
        }

        if (dataUsage == DataUsage::Output) {
            VPU_INTERNAL_CHECK(data->producerEdge() != nullptr,
                "serializeIOInfoSection failed on output data {}. Output must have any producer but it doesn't",
                data->usage());
        }

        VPU_INTERNAL_CHECK(data->parentDataToDataEdge() == nullptr,
            "serializeIOInfoSection failed on {} with usage {}. IO data must have no parentDatas but it does");

        VPU_INTERNAL_CHECK(!data->attrs().has("ioIdx"),
            "serializeIOInfoSection failed: IO data {} with usage {} doesn't have ioIdx attribute",
            data->name(), data->usage());

        data->attrs().set("ioIdx", datasNumber);

        data->serializeIOInfo(blobSerializer);

        ++datasNumber;
    }

    return datasNumber;
}

ElfN_Ehdr BackEnd::createElfHeader() {
    ElfN_Ehdr elfHdr = {};
    elfHdr.e_ident[0] = 0x7f;
    elfHdr.e_ident[1] = 'e';
    elfHdr.e_ident[2] = 'l';
    elfHdr.e_ident[3] = 'f';
    for (int i = 4; i < 16; i++) {
        elfHdr.e_ident[i] = 0;
    }
    elfHdr.e_type = 1;
    elfHdr.e_machine = 2;
    elfHdr.e_version = 2;
    elfHdr.e_entry = 0;
    elfHdr.e_phoff = 0;
    elfHdr.e_shoff = 0;
    elfHdr.e_ehsize = 8 * sizeof(elfHdr);

    return elfHdr;
}

void BackEnd::serializeConstData(const Model& model, const mv_blob_header& blobHdr, std::vector<char>& blob) {
    for (const auto& data : model->datas()) {
        if (data->usage() != DataUsage::Const) {
            continue;
        }

        IE_ASSERT(data->producerEdge() == nullptr);
        IE_ASSERT(data->parentDataToDataEdge() == nullptr);
        IE_ASSERT(data->numConsumers() != 0);
        IE_ASSERT(data->dataLocation().location == Location::Blob);

        const auto content = data->content();
        IE_ASSERT(content != nullptr);

        std::copy_n(content->get<uint8_t>(), content->byteSize(), blob.data() + blobHdr.const_data_section_offset + data->dataLocation().offset);
    }
}

void BackEnd::serializeConstShapes(const Model& model, const mv_blob_header& blobHdr, std::vector<char>& blob) {
    for (const auto& data : model->datas()) {
        const auto dimsOrder = data->desc().dimsOrder();
        const auto storedPerm = dimsOrder.toPermutation();

        const auto serializeToBlob = [&data, &blob, &blobHdr, &storedPerm](const DimValues& values, int offset) {
            BlobSerializer serializer;

            for (const auto& d : storedPerm) {
                serializer.append(checked_cast<uint32_t>(values[d]));
            }

            std::copy_n(serializer.data(), data->desc().numDims() * sizeof(uint32_t), blob.data() + blobHdr.const_data_section_offset + offset);
        };

        const auto shapeLocation = data->shapeLocation();

        if (shapeLocation.dimsLocation == Location::Blob) {
            serializeToBlob(data->desc().dims(), shapeLocation.dimsOffset);
        } else if (data->usage() == DataUsage::Output || data->usage() == DataUsage::Input) {
            auto ioDimsUpperBoundOffset = data->attrs().get<int>("ioDimsUpperBoundOffset");
            auto d = data->desc().dims();
            if (d.has(Dim::N))
                d.set(Dim::N, d.get(Dim::N, 1) * data->attrs().getOrDefault<int>("batch", 1));
            serializeToBlob(d, ioDimsUpperBoundOffset);
        }

        if (shapeLocation.stridesLocation == Location::Blob) {
            serializeToBlob(data->strides(), shapeLocation.stridesOffset);
        } else if (data->usage() == DataUsage::Output || data->usage() == DataUsage::Input) {
            auto ioStridesUpperBoundOffset = data->attrs().get<int>("ioStridesUpperBoundOffset");
            serializeToBlob(data->strides(), ioStridesUpperBoundOffset);
        }
    }
}

void BackEnd::serializeParamsAndResults(const Model& model, const mv_blob_header& blobHdr,
                        std::vector<char>& blob) {
    const auto networkParams = model->attrs().getOrDefault<ov::ParameterVector>("networkParameters");
    const auto networkResults = model->attrs().getOrDefault<ov::ResultVector>("networkResults");

    auto getNetworkParameterHeader = [](const std::shared_ptr<ov::Node>& node) {
        network_params_header nph;
        nph.element_type_bytesize = sizeof(node->get_element_type().operator ov::element::Type_t());
        nph.name_lenght = node->get_friendly_name().size();
        nph.shape_size = node->get_shape().size();
        nph.output_tensor_names_size = node->get_output_tensor(0).get_names().size();
        return nph;
    };

    uint32_t networkInfoOffset = blob.size();
    auto serializeParameters = [&blob, &networkInfoOffset,
                                &getNetworkParameterHeader](
                                const std::shared_ptr<ov::Node>& node) {
        BlobSerializer headerSerializer;
        BlobSerializer shapeSerializer;
        BlobSerializer elementTypeSerializer;
        BlobSerializer tensorNamesSerializer;
        BlobSerializer inputNameForResultSerializer;

        const auto nph = getNetworkParameterHeader(node);
        const bool isResult = ov::is_type<ov::op::v0::Result>(node);
        int totalNetworkInfoOffset =
            networkInfoOffset + sizeof(nph) + nph.name_lenght +
            nph.element_type_bytesize +
            sizeof(size_t) * (nph.output_tensor_names_size + nph.shape_size);

        for (const auto& name : node->get_output_tensor(0).get_names()) {
            totalNetworkInfoOffset += sizeof(size_t) + name.size();
        }
        if (isResult) {
            totalNetworkInfoOffset +=
                sizeof(size_t) +
                ngraph::op::util::create_ie_output_name(node->input_value(0)).size();
        }

        blob.resize(totalNetworkInfoOffset);

        headerSerializer.append(nph);
        std::copy_n(headerSerializer.data(), sizeof(nph),
                    blob.data() + networkInfoOffset);

        networkInfoOffset += sizeof(nph);
        const auto nodeName = node->get_friendly_name();
        VPU_THROW_UNLESS(
            node->get_output_partial_shape(0).rank().is_static(),
            "Serialization of shapes with dynamic rank is not supported");
        const auto nodeShape = node->get_output_partial_shape(0).get_shape();
        const auto nodeElType =
            node->get_element_type().operator ov::element::Type_t();

        std::copy_n(nodeName.data(), nodeName.size(),
                    blob.data() + networkInfoOffset);
        networkInfoOffset += nph.name_lenght;

        for (const auto shapeIdx : nodeShape) {
            shapeSerializer.append(shapeIdx);
        }
        std::copy_n(shapeSerializer.data(),
                    shapeSerializer.size(), blob.data() + networkInfoOffset);
        networkInfoOffset += shapeSerializer.size();
        elementTypeSerializer.append(nodeElType);
        std::copy_n(elementTypeSerializer.data(), nph.element_type_bytesize,
                    blob.data() + networkInfoOffset);
        networkInfoOffset += nph.element_type_bytesize;

        for (const auto& name : node->get_output_tensor(0).get_names()) {
            tensorNamesSerializer.append(name.size());
            for (const auto ch : name) {
                tensorNamesSerializer.append(ch);
            }
        }
        std::copy_n(tensorNamesSerializer.data(), tensorNamesSerializer.size(),
                    blob.data() + networkInfoOffset);
        networkInfoOffset += tensorNamesSerializer.size();

        if (isResult) {
            const auto inputNameForResult =
                ngraph::op::util::create_ie_output_name(node->input_value(0));
            inputNameForResultSerializer.append(inputNameForResult.size());
            for (const auto ch : inputNameForResult) {
                inputNameForResultSerializer.append(ch);
            }
            std::copy_n(inputNameForResultSerializer.data(),
                        inputNameForResultSerializer.size(),
                        blob.data() + networkInfoOffset);
            networkInfoOffset += inputNameForResultSerializer.size();
        }
    };

    BlobSerializer networkInfoSerializer;
    network_info_header nih;
    nih.parameters_size = networkParams.size();
    nih.results_size = networkResults.size();
    blob.resize(networkInfoOffset + sizeof(nih));
    networkInfoSerializer.append(nih);
    std::copy_n(networkInfoSerializer.data(), sizeof(nih), blob.data() + networkInfoOffset);
    networkInfoOffset += sizeof(nih);

    for (const auto& param : networkParams) {
        serializeParameters(param);
    }

    for (const auto& result : networkResults) {
        serializeParameters(result);
    }
}

void BackEnd::serialize(
        const Model& model,
        std::vector<char>& blob,
        std::pair<char*, size_t>& blobHeader,
        int& numActiveStages) {
    VPU_PROFILE(serialize);

    const auto& env = CompileEnv::get();

    BlobSerializer inputInfoSerializer;
    BlobSerializer outputInfoSerializer;
    BlobSerializer stagesSerializer;

    const auto getExecStages = [&model]() {
        StageVector execStages;
        execStages.reserve(model->numStages());

        for (const auto& stage : model->getStages()) {
            if (stage->category() == StageCategory::Special) {
                continue;
            }

            execStages.emplace_back(stage);
        }

        return execStages;
    };

    const auto getModelStagesStat = [&model]() {
        ModelStagesStat modelStagesStat{false, false, false};

        for (const auto& stage : model->getStages()) {
            if (stage->category() == StageCategory::Special) {
                continue;
            }

            if (stage->category() == StageCategory::HW) {
                modelStagesStat.hasHwStage = true;
            } else if (stage->category() == StageCategory::SHAVE) {
                modelStagesStat.hasShaveStage = true;
            } else if (stage->category() == StageCategory::DMA) {
                modelStagesStat.hasDmaStage = true;
            }
        }

        return modelStagesStat;
    };

    const auto createBlobHeader = [&env, &model, &inputInfoSerializer, &outputInfoSerializer, &stagesSerializer]
            (int numInputs, int numOutputs, const StageVector& execStages, const ModelStagesStat& modelStagesStat) {
        const auto batchSize = model->batchSize();
        const auto usedMemory = model->attrs().get<UsedMemory>("usedMemory");

        const auto hdrSize = alignVal<int>(sizeof(ElfN_Ehdr) + sizeof(mv_blob_header), 64);
        const auto inputInfoSecSize = alignVal(inputInfoSerializer.size(), 64);
        const auto outputInfoSecSize = alignVal(outputInfoSerializer.size(), 64);
        const auto stagesSecSize = alignVal(stagesSerializer.size(), 64);
        const auto constDataSecSize = alignVal(usedMemory.blob, 64);

        mv_blob_header blobHdr = {};
        blobHdr.magic_number = BLOB_MAGIC_NUMBER;
        blobHdr.file_size = checked_cast<uint32_t>(hdrSize + inputInfoSecSize + outputInfoSecSize + stagesSecSize + constDataSecSize);
        blobHdr.blob_ver_major = BLOB_VERSION_MAJOR;
        blobHdr.blob_ver_minor = BLOB_VERSION_MINOR;
        blobHdr.inputs_count = checked_cast<uint32_t>(numInputs);
        blobHdr.outputs_count = checked_cast<uint32_t>(numOutputs);
        blobHdr.stages_count = checked_cast<uint32_t>(execStages.size());
        blobHdr.inputs_size = checked_cast<uint32_t>(usedMemory.input);
        blobHdr.outputs_size = checked_cast<uint32_t>(usedMemory.output);
        blobHdr.batch_size = checked_cast<uint32_t>(batchSize);
        blobHdr.bss_mem_size = checked_cast<uint32_t>(usedMemory.BSS);
        blobHdr.number_of_cmx_slices = checked_cast<uint32_t>(env.resources.numCMXSlices);
        blobHdr.number_of_shaves = checked_cast<uint32_t>(env.resources.numSHAVEs);
        blobHdr.has_hw_stage = static_cast<uint32_t>(modelStagesStat.hasHwStage);
        blobHdr.has_shave_stage = static_cast<uint32_t>(modelStagesStat.hasShaveStage);
        blobHdr.has_dma_stage = static_cast<uint32_t>(modelStagesStat.hasDmaStage);
        blobHdr.input_info_section_offset = checked_cast<uint32_t>(hdrSize);
        blobHdr.output_info_section_offset = checked_cast<uint32_t>(blobHdr.input_info_section_offset + inputInfoSecSize);
        blobHdr.stage_section_offset = checked_cast<uint32_t>(blobHdr.output_info_section_offset + outputInfoSecSize);
        blobHdr.const_data_section_offset = checked_cast<uint32_t>(blobHdr.stage_section_offset + stagesSecSize);

        return blobHdr;
    };

    const int numInputs = serializeIOInfoSection(model, DataUsage::Input, inputInfoSerializer);
    const int numOutputs = serializeIOInfoSection(model, DataUsage::Output, outputInfoSerializer);

    const auto& execStages = getExecStages();
    numActiveStages = checked_cast<int>(execStages.size());

    for (const auto& stage : execStages) {
        stage->serialize(stagesSerializer);
    }

    const auto modelStagesStat = getModelStagesStat();

    const auto elfHdr = createElfHeader();
    const auto blobHdr = createBlobHeader(numInputs, numOutputs, execStages, modelStagesStat);

    blob.clear();
    blob.resize(blobHdr.file_size, 0);

    std::copy_n(&elfHdr, 1, reinterpret_cast<ElfN_Ehdr*>(blob.data()));
    std::copy_n(&blobHdr, 1, reinterpret_cast<mv_blob_header*>(blob.data() + sizeof(elfHdr)));
    std::copy_n(inputInfoSerializer.data(), inputInfoSerializer.size(), blob.data() + blobHdr.input_info_section_offset);
    std::copy_n(outputInfoSerializer.data(), outputInfoSerializer.size(), blob.data() + blobHdr.output_info_section_offset);
    std::copy_n(stagesSerializer.data(), stagesSerializer.size(), blob.data() + blobHdr.stage_section_offset);

    serializeConstData(model, blobHdr, blob);
    serializeConstShapes(model, blobHdr, blob);
    const auto networkParams = model->attrs().getOrDefault<ov::ParameterVector>("networkParameters");
    const auto networkResults = model->attrs().getOrDefault<ov::ResultVector>("networkResults");
    // To avoid constant network case
    if (!networkParams.empty() && !networkResults.empty()) {
        serializeParamsAndResults(model, blobHdr, blob);
    }

    blobHeader.first = blob.data();
    blobHeader.second = sizeof(ElfN_Ehdr) + sizeof(mv_blob_header);
}

}  // namespace vpu
