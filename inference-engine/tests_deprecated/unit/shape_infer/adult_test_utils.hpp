// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>

#include <utility>

#pragma once

#include <gtest/gtest.h>
#include <shape_infer/const_infer/ie_const_infer_holder.hpp>
#include "built_in_shape_infer_general_test.hpp"

namespace IE = InferenceEngine;

struct InOutData {
    testing::InOutShapes inOutShapes;
    std::vector<std::vector<float>> inData;
    std::vector<std::vector<float>> outData;
};

using FloatMap = std::map<std::string, std::vector<float>>;
using InitBlobsFunc = std::function<IE::BlobMap(const FloatMap& inOutData)>;

struct ASIConfig {
    InOutData inOutData;
    std::string type;
    FloatMap floatBlobData;
    std::map<std::string, std::string> strParams;
    InitBlobsFunc initBlobs;
    std::vector<IE::Precision> inPrecisions;
    std::vector<IE::Precision> outPrecisions;
};

class BaseMatcher {
public:
    explicit BaseMatcher(ASIConfig config) : config(std::move(config)) {}

protected:
    void compareWithRef(const std::vector<IE::Blob::Ptr>& outBlobs,
                        const std::vector<std::vector<float>>& refData,
                        float tolerance = 0.0001);

    std::vector<IE::Blob::Ptr>
    createBlobs(const std::vector<IE::SizeVector>& shapes, const std::vector<IE::Precision>& precisions);

    void fillBlobs(const std::vector<IE::Blob::Ptr>& blobs, const std::vector<std::vector<float>>& data);

    ASIConfig config;
};

class ConstInferMatcher : public BaseMatcher {
public:
    explicit ConstInferMatcher(const ASIConfig& config) : BaseMatcher(config) {}

    void toData(const std::vector<std::vector<float>>& refData);

private:
    std::shared_ptr<IE::ShapeInfer::ConstInferHolder> holder;
};

class ShapeInferMatcher : public BaseMatcher {
public:
    explicit ShapeInferMatcher(const ASIConfig& config) : BaseMatcher(config) {}

    void toShapes(const std::vector<IE::SizeVector>& refShape);

private:
    std::unique_ptr<IE::ShapeInfer::BuiltInShapeInferHolder> siHolder;
    IE::StatusCode sts;
    IE::ResponseDesc desc;
};

template<typename M>
class MatcherConfigurator {
public:
    explicit MatcherConfigurator(ASIConfig config) : config(std::move(config)) {}

    MatcherConfigurator& withParams(const std::map<std::string, std::string>& params) {
        config.strParams = params;
        return *this;
    }

    MatcherConfigurator& withInputPrecisions(const std::vector<IE::Precision>& inputPrecisions) {
        config.inPrecisions = inputPrecisions;
        return *this;
    }

    MatcherConfigurator& withOutputPrecisions(const std::vector<IE::Precision>& outputPrecisions) {
        config.outPrecisions = outputPrecisions;
        return *this;
    }

    MatcherConfigurator& withBlobs(const FloatMap& blobDataMap) {
        config.floatBlobData = blobDataMap;
        return *this;
    }

    M equals() {
        return M(config);
    }

private:
    ASIConfig config;
};

class ASITestBuilder {
    ASIConfig config;
public:
    ASITestBuilder() {
        config.initBlobs = defaultBlobInit();
    }

    ASITestBuilder& withData(const InOutData& data) {
        config.inOutData = data;
        config.inPrecisions = {data.inOutShapes.inDims.size(), IE::Precision::FP32};
        config.outPrecisions = {data.inOutShapes.outDims.size(), IE::Precision::FP32};
        return *this;
    }

    ASITestBuilder& withType(const std::string& type) {
        config.type = type;
        return *this;
    }

    MatcherConfigurator<ConstInferMatcher> constInferResultFor();

    MatcherConfigurator<ShapeInferMatcher> shapeInferResultFor();

private:
    InitBlobsFunc defaultBlobInit();
};

PRETTY_PARAM(BlobsParam, FloatMap)

PRETTY_PARAM(InOutDataParam, InOutData)
