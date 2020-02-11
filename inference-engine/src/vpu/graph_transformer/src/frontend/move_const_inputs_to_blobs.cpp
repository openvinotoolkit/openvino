// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <low_precision_transformations/blob_transformation.hpp>

#include <vpu/compile_env.hpp>

namespace vpu {

void FrontEnd::moveConstInputsToBlobs(ie::ICNNNetwork& network) {
    VPU_PROFILE(moveConstInputsToBlobs);

    const auto& env = CompileEnv::get();

    env.log->trace("Move const inputs to blobs");
    VPU_LOGGER_SECTION(env.log);

    ie::details::BlobTransformation blobsTransformation;
    blobsTransformation.transform(network, true);
}

}  // namespace vpu
