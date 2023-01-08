// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "vpu_layers_tests.hpp"

#include <vpu/middleend/pass_manager.hpp>
#include <vpu/frontend/frontend.hpp>

class graphTransformerFunctionalTests : public vpuLayersTests {
protected:
    void SetUp() override;

    void CreateModel();
    void PrepareGraphCompilation();
    void InitializeInputData(const vpu::DataDesc& inputDataDesc);

    vpu::Data InitializeOutputData(const vpu::DataDesc& outputDataDesc);

    /// @returns execution time in microseconds
    int64_t CompileAndInfer(InferenceEngine::Blob::Ptr& inputBlob,
                            InferenceEngine::Blob::Ptr& outputBlob,
                            bool lockLayout = false);

protected:
   vpu::ModelPtr            _gtModel;
   vpu::PluginConfiguration _configuration;
   vpu::StageBuilder::Ptr   _stageBuilder;
   vpu::Data                _dataIntermediate;

private:
   InferenceEngine::ExecutableNetwork _executableNetwork;
};
