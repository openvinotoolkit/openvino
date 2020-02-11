// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <gtest/gtest.h>
#include <tests_common.hpp>
#include <memory>
#include "parameters.h"
#include "shapes.h"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

class LayerBuilder {
private:
    CNNLayerPtr layer;
    std::string dataName = "data";
    std::unique_ptr<Parameters> parameters;
public:
    explicit LayerBuilder (InferenceEngine::CNNLayer::Ptr createdLayer) : layer(std::move(createdLayer)) {
        parameters = std::unique_ptr<Parameters>(new Parameters(layer->type));
    }

    LayerBuilder&  setParams(bool valid) {
        if (valid) {
            layer->params = parameters->getValidParameters();
        } else {
            layer->params = parameters->getInvalidParameters();
        }
        return *this;
    }

    LayerBuilder&  setShapes(std::vector<DataPtr>& spData, bool valid_input) {
        testing::InOutShapes shapes;
        LayersWithNotEqualIO layersWithNotEqualIO;
        LayersWithEqualIO layersWithEqualIO;
        LayersWithNIO layersWithNIO;
        std::vector<Layers*> layers{&layersWithNotEqualIO, &layersWithEqualIO, &layersWithNIO};
        ShapesHelper* shapesHelper = nullptr;
        for(const auto& layer : layers) {
            if (layer->containLayer(this->layer->type)) {
                shapesHelper = layer->factoryShape();
                break;
            }
        }
        if (valid_input) {
            shapes = shapesHelper->getValidShapes();
        } else {
            shapes = shapesHelper->getInvalidInputShapes();
        }
        for (const auto& inData : shapes.inDims) {
            DataPtr data = std::make_shared<Data>(dataName,
                                                  InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32,
                                                                              SizeVector(inData.rbegin(), inData.rend()),
                                                                              TensorDesc::getLayoutByDims(inData)));
            spData.push_back(data);
            layer->insData.push_back(data);
        }
        for (const auto& outData : shapes.outDims) {
            layer->outData.push_back(std::make_shared<Data>(dataName,
                                                            InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32,
                                                            SizeVector(outData.rbegin(), outData.rend()),
                                                            TensorDesc::getLayoutByDims(outData))));
        }
        delete shapesHelper;
        return *this;
    }

    CNNLayerPtr get() {
        return layer;
    }

    int getNumOfParams() {
        return parameters->getNumOfParameters();
    }

    int getNumOfLayerVariant() {
        LayersWithNotEqualIO layersWithNotEqualIO;
        LayersWithEqualIO layersWithEqualIO;
        LayersWithNIO layersWithNIO;
        Layers* layers[] = {&layersWithNotEqualIO, &layersWithEqualIO, &layersWithNIO};
        int cnt = 0;
        for(const auto& layer : layers) {
            if (layer->containLayer(this->layer->type)) {
                cnt++;
            }
        }
        return cnt;
    }
};

class CNNLayerValidationTests : public testing::TestWithParam<std::string>{
public:
    void SetUp() override {
        auto params = GetParam();
        type = params;
    }

    std::shared_ptr<LayerBuilder>& createConcreteLayer(const std::string& type) {
        layer = std::make_shared<LayerBuilder>(TestsCommon::createLayer(type));
        return layer;
    }

    std::shared_ptr<LayerBuilder>&  getBuilder() {
        return layer;
    }

    CNNLayerPtr getLayer() {
        return layer.get()->get();
    }

    int getNumOfParams() {
        return layer.get()->getNumOfParams();
    }

    int getNumOfLayerVariant() {
        return layer.get()->getNumOfLayerVariant();
    }
protected:
    std::string type;
    bool valid_params = true;
    bool valid_input = true;
    std::shared_ptr<LayerBuilder> layer;
};

#define assertThat(type) SCOPED_TRACE("");createConcreteLayer(type)
