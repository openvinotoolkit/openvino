// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_builders.hpp>
#include <ie_icnn_network.hpp>

#include "single_layer_common.hpp"
#include "tests_vpu_common.hpp"

using namespace InferenceEngine;

/* this function assumes that the precision of a generated network is FP16 */
std::shared_ptr<InferenceEngine::ICNNNetwork> createNetworkWithDesiredSize(std::size_t sizeInMB) {

    Builder::Network builder("network");
    Builder::FullyConnectedLayer fcBuilder("FullyConnected");

    SizeVector inputDims = {1, 2, 16, 16}; // 1 KB

    auto generateBlob = [](Precision precision,
                           SizeVector dims, Layout layout) {
        IE_ASSERT(precision == Precision::FP16);
        Blob::Ptr blob = make_shared_blob<ie_fp16>(TensorDesc(precision, dims, layout));
        blob->allocate();
        GenRandomDataCommon(blob);
        return blob;
    };

    idx_t layerId = builder.addLayer(Builder::InputLayer("input").setPort(Port(inputDims)));

    idx_t weightsId = builder.addLayer(Builder::ConstLayer("weights").setData(generateBlob(Precision::FP16,
                                                                                           {sizeInMB * 1024, 2, 16, 16}, Layout::OIHW)));

    layerId = builder.addLayer({{layerId}, {weightsId}}, Builder::FullyConnectedLayer("FullyConnected").setOutputNum(1024 * sizeInMB));

    builder.addLayer({PortInfo(layerId)}, Builder::OutputLayer("output"));

    INetwork::CPtr network = builder.build();
    std::shared_ptr<ICNNNetwork> cnnNetwork = Builder::convertToICNNNetwork(network);

    return cnnNetwork;
}
