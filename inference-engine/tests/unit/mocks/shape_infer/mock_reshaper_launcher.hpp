// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gmock/gmock.h>

#include <inference_engine/shape_infer/ie_reshape_launcher.hpp>
#include <inference_engine/shape_infer/ie_reshape_io_controllers.hpp>
#include <shape_infer/mock_ishape_infer_impl.hpp>
#include <shape_infer/mock_input_controller.hpp>
#include <shape_infer/mock_output_controller.hpp>


using namespace InferenceEngine;
using namespace ShapeInfer;

class MockReshapeLauncher : public ReshapeLauncher {
public:
    using Ptr = std::shared_ptr<MockReshapeLauncher>;
    class TestLauncherInitializer : public DefaultInitializer {
    public:
        void check(const CNNLayer* layer, const IShapeInferImpl::Ptr& impl) override {}

        InputController* createInputController(const CNNLayer* layer) override {
            if (!_iController) {
                std::vector<DataPtr> data;
                if (layer) {
                    for (auto const& insData: layer->insData) {
                        data.push_back(insData.lock());
                    }
                }
                _iController = new MockInputController(data);
            }
            return _iController;
        }

        OutputController* createOutputController(const CNNLayer* layer) override {
            if (!_oController) {
                std::vector<DataPtr> data;
                if (layer) data = layer->outData;;
                _oController = new MockOutputController(data);
            }
            return _oController;
        }

        MockInputController* getInputController() {
            return _iController;
        }

        MockOutputController* getOutputController() {
            return _oController;
        }

    private:
        MockInputController* _iController;
        MockOutputController* _oController;
    };

    MockReshapeLauncher(const DefaultInitializer::Ptr& initializer = std::make_shared<TestLauncherInitializer>(),
                        const CNNLayer* layer = nullptr,
                        const IShapeInferImpl::Ptr& impl = std::make_shared<MockIShapeInferImpl>())
            : ReshapeLauncher(layer, impl, initializer) {}

    MOCK_METHOD2(setShapeByName, void(const SizeVector&, const std::string&));

    MOCK_METHOD1(reshape, void(const std::set<ReshapeLauncher::Ptr>&));

    MOCK_METHOD1(applyChanges, void(CNNLayer*));

    MOCK_METHOD0(reset, void());

    MOCK_QUALIFIED_METHOD0(getLayerName, const, std::string());

    MOCK_METHOD1(setShapeInferImpl, void(const IShapeInferImpl::Ptr& ));

    void realReset() {
        ReshapeLauncher::reset();
    }

    void realReshape() {
        ReshapeLauncher::reshape({});
    }

    std::string realGetLayerName() {
        return ReshapeLauncher::getLayerName();
    }
};

