// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <utility>

#include <ie_input_info.hpp>

#include <vpu/backend/blob_format.hpp>
#include <vpu/model/data_desc.hpp>
#include <vpu/graph_transformer.hpp>

namespace vpu {

namespace ie = InferenceEngine;

class BlobReader {
public:
    BlobReader() = default;

    void parse(const std::vector<char>& blob);

    const ie::InputsDataMap& getNetworkInputs() const { return _networkInputs; }
    const ie::OutputsDataMap& getNetworkOutputs() const { return _networkOutputs; }

    uint32_t getStageCount() const { return _blobHeader.stages_count; }

    uint32_t getMagicNumber() const { return _blobHeader.magic_number; }

    uint32_t getVersionMajor() const { return _blobHeader.blob_ver_major; }
    uint32_t getVersionMinor() const { return _blobHeader.blob_ver_minor; }

    uint32_t getNumberOfShaves() const { return _blobHeader.number_of_shaves; }
    uint32_t getNumberOfSlices() const { return _blobHeader.number_of_cmx_slices; }

    const DataInfo& getInputInfo()  const { return _inputInfo; }
    const DataInfo& getOutputInfo() const { return _outputInfo; }

    std::pair<const char*, size_t> getHeader() const { return {_pBlob, sizeof(ElfN_Ehdr) + sizeof(mv_blob_header)};}

private:
    const char* _pBlob = nullptr;

    mv_blob_header _blobHeader = {};

    ie::InputsDataMap  _networkInputs;
    ie::OutputsDataMap _networkOutputs;

    DataInfo _inputInfo;
    DataInfo _outputInfo;
};

}  // namespace vpu
