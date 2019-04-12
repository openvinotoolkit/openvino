#include <utility>

/*
* INTEL CONFIDENTIAL
* Copyright (C) 2018-2019 Intel Corporation.
*
* The source code contained or described herein and all documents
* related to the source code ("Material") are owned by Intel Corporation
* or its suppliers or licensors. Title to the Material remains with
* Intel Corporation or its suppliers and licensors. The Material may
* contain trade secrets and proprietary and confidential information
* of Intel Corporation and its suppliers and licensors, and is protected
* by worldwide copyright and trade secret laws and treaty provisions.
* No part of the Material may be used, copied, reproduced, modified,
* published, uploaded, posted, transmitted, distributed, or disclosed
* in any way without Intel's prior express written permission.
*
* No license under any patent, copyright, trade secret or other
* intellectual property right is granted to or conferred upon you by
* disclosure or delivery of the Materials, either expressly, by implication,
* inducement, estoppel or otherwise. Any license under such intellectual
* property rights must be express and approved by Intel in writing.
*
* Include any supplier copyright notices as supplier requires Intel to use.
*
* Include supplier trademarks or logos as supplier requires Intel to use,
* preceded by an asterisk. An asterisked footnote can be added as follows:
* *Third Party trademarks are the property of their respective owners.
*
* Unless otherwise agreed by Intel in writing, you may not remove or alter
* this notice or any other notice embedded in Materials by Intel or Intel's
* suppliers or licensors in any way.
*/

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
            DataPtr data = std::make_shared<Data>(dataName, inData, InferenceEngine::Precision::FP32);
            spData.push_back(data);
            layer->insData.push_back(data);
        }
        for (const auto& outData : shapes.outDims) {
            layer->outData.push_back(std::make_shared<Data>(dataName, outData, InferenceEngine::Precision::FP32));
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