// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/function.hpp>

#include <memory>
#include <string>
#include <vector>

#include <legacy/cnn_network_impl.hpp>
#include <ie_ngraph_utils.hpp>
#include "blob_factory.hpp"
#include <ngraph/op/constant.hpp>

namespace InferenceEngine {
namespace details {

INFERENCE_ENGINE_API_CPP(std::shared_ptr<CNNNetworkImpl>)
convertFunctionToICNNNetwork(const std::shared_ptr<const ::ngraph::Function>& graph,
                             const CNNNetwork &network, bool keep_constant_inputs = false);

INFERENCE_ENGINE_API_CPP(void)
convertFunctionToICNNNetwork(const std::shared_ptr<const ::ngraph::Function>& graph,
                             const CNNNetwork &ngraphNetwork,
                             CNNNetworkImpl* cnnNetworkImpl,
                             bool keep_constant_inputs = false);

// TODO: move ConstAllocatorWrapper class, shareWeights add addBlob into CNNLayerCreator when NodeConverter class is removed
class ConstAllocatorWrapper : public IAllocator {
public:
    explicit ConstAllocatorWrapper(std::shared_ptr<ngraph::op::Constant> constOp): _constOp(std::move(constOp)) {}

    void* lock(void* handle, LockOp) noexcept override {
        return handle;
    }

    void unlock(void*) noexcept override {}  // NOLINT

    void* alloc(size_t) noexcept override {
        return const_cast<void*>(_constOp->get_data_ptr());
    }

    bool free(void*) noexcept override {  // NOLINT
        return true;
    }

private:
    std::shared_ptr<ngraph::op::Constant> _constOp;
};

enum BlobType {
    weights,
    biases
};

inline Blob::Ptr shareWeights(const std::shared_ptr<ngraph::op::Constant>& constLayer) {
    if (!constLayer) IE_THROW() << "Cannot share weights! Constant operation is empty!";
    auto dataPrecision = convertPrecision(constLayer->get_element_type());

    size_t shapeSize = ngraph::shape_size(constLayer->get_shape());
    constexpr size_t byte_size{8};
    if (dataPrecision == Precision::BIN) {
        shapeSize = (shapeSize + (byte_size - 1)) / byte_size;
    }

    TensorDesc td(dataPrecision, {shapeSize}, Layout::C);

    auto blob = make_blob_with_precision(td, std::make_shared<ConstAllocatorWrapper>(constLayer));
    blob->allocate();

    return blob;
}

template <class T>
bool addBlob(const std::shared_ptr<ngraph::Node>& weightsNode, std::shared_ptr<T>& res, BlobType type) {
    auto constWeights = ngraph::as_type_ptr<ngraph::op::Constant>(weightsNode);
    if (constWeights) {
        Blob::Ptr dataBlob = shareWeights(constWeights);
        if (type == weights) {
            res->blobs["weights"] = dataBlob;
            res->_weights = dataBlob;
        } else if (type == biases) {
            res->blobs["biases"] = dataBlob;
            res->_biases = dataBlob;
        } else {
            return false;
        }
        return true;
    } else {
        return false;
    }
}

}  // namespace details
}  // namespace InferenceEngine
