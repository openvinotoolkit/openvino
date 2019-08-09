// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <tuple>
#include "adult_test.hpp"
#include "adult_test_utils.hpp"


using namespace InferenceEngine;
using namespace details;
using namespace ShapeInfer;

void BaseMatcher::compareWithRef(const std::vector<InferenceEngine::Blob::Ptr>& outBlobs,
                                 const std::vector<std::vector<float>>& refData,
                                 float tolerance) {
    for (int outIdx = 0; outIdx < outBlobs.size(); outIdx++) {
        auto* data = outBlobs[outIdx]->buffer().as<float*>();
        for (int elemIdx = 0; elemIdx < refData[outIdx].size(); elemIdx++) {
            ASSERT_NEAR(data[elemIdx], refData[outIdx][elemIdx], tolerance);
        }
    }
}

std::vector<IE::Blob::Ptr>
BaseMatcher::createBlobs(const std::vector<IE::SizeVector>& shapes, const std::vector<IE::Precision>& precisions) {
    if (shapes.size() != precisions.size())
        THROW_IE_EXCEPTION << "Vectors of shapes and precisions can't have different sizes";
    std::vector<Blob::Ptr> blobs;
    int i = 0;
    for (const auto& dims : shapes) {
        // it's assumed that empty dims = empty data = no blob
        if (!dims.empty()) {
            TensorDesc inDesc(precisions[i++], dims, TensorDesc::getLayoutByDims(dims));
            auto blob = make_blob_with_precision(inDesc);
            blob->allocate();
            blobs.push_back(blob);
        }
    }
    return blobs;
}

void BaseMatcher::fillBlobs(const std::vector<IE::Blob::Ptr>& blobs, const std::vector<std::vector<float>>& data) {
    if (!data.empty()) {
        for (int blobIdx = 0; blobIdx < blobs.size(); blobIdx++) {
            auto blob = blobs[blobIdx];
            // it's assumed that empty dims = empty data = no blob
            if (!data[blobIdx].empty()) {
                switch (blob->getTensorDesc().getPrecision()) {
                    case Precision::FP32: {
                        auto* buffer = blob->buffer().as<float*>();
                        for (int dataIdx = 0; dataIdx < blob->size(); dataIdx++) {
                            buffer[dataIdx] = data[blobIdx][dataIdx];
                        }
                    }
                        break;
                    case Precision::I32: {
                        auto* buffer = blob->buffer().as<int32_t*>();
                        for (int dataIdx = 0; dataIdx < blob->size(); dataIdx++) {
                            buffer[dataIdx] = static_cast<int32_t>(data[blobIdx][dataIdx]);
                        }
                    }
                        break;
                    default:
                        THROW_IE_EXCEPTION << "Unsupported precision " << blob->getTensorDesc().getPrecision() << " to fill blobs";
                }
            }
        }
    }
}

void ConstInferMatcher::toData(const std::vector<std::vector<float>>& refData) {
    auto impl = holder->getConstInferImpl(config.type);
    ASSERT_NE(nullptr, impl);
    auto outBlobs = createBlobs(config.inOutData.inOutShapes.outDims, config.outPrecisions);
    auto inBlobs = createBlobs(config.inOutData.inOutShapes.inDims, config.inPrecisions);
    fillBlobs(inBlobs, config.inOutData.inData);
    auto blobs = config.initBlobs(config.floatBlobData);
    std::vector<Blob::CPtr> inCBlobs;
    std::copy(inBlobs.begin(), inBlobs.end(), back_inserter(inCBlobs));
    ASSERT_NO_THROW(impl->infer(inCBlobs, config.strParams, blobs, outBlobs));
    compareWithRef(outBlobs, refData);
}

void ShapeInferMatcher::toShapes(const std::vector<IE::SizeVector>& refShape) {
    siHolder.reset(new IE::ShapeInfer::BuiltInShapeInferHolder());
    IE::IShapeInferImpl::Ptr impl;
    std::vector<IE::SizeVector> outShapes;
    sts = siHolder->getShapeInferImpl(impl, config.type.c_str(), &desc);
    ASSERT_NE(nullptr, impl);
    auto inBlobs = createBlobs(config.inOutData.inOutShapes.inDims, config.inPrecisions);
    fillBlobs(inBlobs, config.inOutData.inData);
    std::vector<Blob::CPtr> inCBlobs;
    std::copy(inBlobs.begin(), inBlobs.end(), back_inserter(inCBlobs));
    auto blobs = config.initBlobs(config.floatBlobData);
    sts = impl->inferShapes(inCBlobs, config.strParams, blobs, outShapes, &desc);
    ASSERT_EQ(sts, IE::OK) << desc.msg;
    ASSERT_EQ(config.inOutData.inOutShapes.outDims, outShapes);
}

InitBlobsFunc ASITestBuilder::defaultBlobInit() {
    return [](const FloatMap& blobDataMap) -> BlobMap {
        BlobMap blobs;
        for (const auto& it : blobDataMap) {
            std::string blobName;
            std::vector<float> data;
            std::tie(blobName, data) = it;
            SizeVector blobDims = {data.size()};
            auto blob = make_shared_blob<float>(Precision::FP32, TensorDesc::getLayoutByDims(blobDims), blobDims,
                                                data);
            blobs[blobName] = blob;
        }
        return blobs;
    };
}

MatcherConfigurator<ConstInferMatcher> ASITestBuilder::constInferResultFor() {
    return MatcherConfigurator<ConstInferMatcher>(config);
}

MatcherConfigurator<ShapeInferMatcher> ASITestBuilder::shapeInferResultFor() {
    return MatcherConfigurator<ShapeInferMatcher>(config);
}
