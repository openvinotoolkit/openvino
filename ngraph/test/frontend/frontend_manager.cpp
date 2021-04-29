// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <frontend_manager/frontend_manager.hpp>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ngraph;
using namespace ngraph::frontend;

TEST(FrontEndManagerTest, testAvailableFrontEnds)
{
    FrontEndManager fem;
    ASSERT_NO_THROW(fem.registerFrontEnd("mock", [](FrontEndCapabilities fec) {
        return std::make_shared<FrontEnd>();
    }));
    auto frontends = fem.availableFrontEnds();
    ASSERT_NE(std::find(frontends.begin(), frontends.end(), "mock"), frontends.end());
    FrontEnd::Ptr fe;
    ASSERT_NO_THROW(fe = fem.loadByFramework("mock"));
}

TEST(FrontEndManagerTest, testMockPluginFrontEnd)
{
    FrontEndManager fem("."); // specify current lib dir
    auto frontends = fem.availableFrontEnds();
    ASSERT_NE(std::find(frontends.begin(), frontends.end(), "mock1"), frontends.end());
}

TEST(FrontEndManagerTest, testDefaultFrontEnd)
{
    FrontEndManager fem;
    ASSERT_ANY_THROW(fem.loadByModel(""));

    std::unique_ptr<FrontEnd> fePtr (new FrontEnd()); // to verify base destructor
    FrontEnd::Ptr fe = std::make_shared<FrontEnd>();
    ASSERT_ANY_THROW(fe->loadFromFile(""));
    ASSERT_ANY_THROW(fe->loadFromFiles({"", ""}));
    ASSERT_ANY_THROW(fe->loadFromMemory(nullptr));
    ASSERT_ANY_THROW(fe->loadFromMemoryFragments({nullptr, nullptr}));
    std::stringstream str;
    ASSERT_ANY_THROW(fe->loadFromStream(str));
    ASSERT_ANY_THROW(fe->loadFromStreams({&str, &str}));
    ASSERT_ANY_THROW(fe->convert(std::shared_ptr<Function>(nullptr)));
    ASSERT_ANY_THROW(fe->convert(InputModel::Ptr(nullptr)));
    ASSERT_ANY_THROW(fe->convertPartially(nullptr));
    ASSERT_ANY_THROW(fe->decode(nullptr));
    ASSERT_ANY_THROW(fe->normalize(nullptr));
}

TEST(FrontEndManagerTest, testDefaultInputModel)
{
    std::unique_ptr<InputModel> imPtr (new InputModel()); // to verify base destructor
    InputModel::Ptr im = std::make_shared<InputModel>();
    ASSERT_ANY_THROW(im->getInputs());
    ASSERT_ANY_THROW(im->getOutputs());
    ASSERT_ANY_THROW(im->overrideAllInputs({nullptr}));
    ASSERT_ANY_THROW(im->overrideAllOutputs({nullptr}));
    ASSERT_ANY_THROW(im->extractSubgraph({nullptr}, {nullptr}));
    ASSERT_ANY_THROW(im->getPlaceByTensorName(""));
    ASSERT_ANY_THROW(im->getPlaceByOperationName(""));
    ASSERT_ANY_THROW(im->getPlaceByOperationAndInputPort("", 0));
    ASSERT_ANY_THROW(im->getPlaceByOperationAndOutputPort("", 0));
    ASSERT_ANY_THROW(im->setNameForTensor(nullptr, ""));
    ASSERT_ANY_THROW(im->addNameForTensor(nullptr, ""));
    ASSERT_ANY_THROW(im->setNameForOperation(nullptr, ""));
    ASSERT_ANY_THROW(im->freeNameForTensor(""));
    ASSERT_ANY_THROW(im->freeNameForOperation(""));
    ASSERT_ANY_THROW(im->setNameForDimension(nullptr, 0, ""));
    ASSERT_ANY_THROW(im->cutAndAddNewInput(nullptr, ""));
    ASSERT_ANY_THROW(im->cutAndAddNewOutput(nullptr, ""));
    ASSERT_ANY_THROW(im->addOutput(nullptr));
    ASSERT_ANY_THROW(im->removeOutput(nullptr));
    ASSERT_ANY_THROW(im->removeInput(nullptr));
    ASSERT_ANY_THROW(im->setDefaultShape(nullptr, ngraph::Shape{}));
    ASSERT_ANY_THROW(im->setPartialShape(nullptr, ngraph::Shape{}));
    ASSERT_ANY_THROW(im->getPartialShape(nullptr));
    ASSERT_ANY_THROW(im->setElementType(nullptr, ngraph::element::Type{}));
    ASSERT_ANY_THROW(im->setTensorValue(nullptr, nullptr));
    ASSERT_ANY_THROW(im->setTensorPartialValue(nullptr, nullptr, nullptr));
}

TEST(FrontEndManagerTest, testDefaultPlace)
{
    std::unique_ptr<Place> placePtr (new Place()); // to verify base destructor
    Place::Ptr place = std::make_shared<Place>();
    ASSERT_ANY_THROW(place->getNames());
    ASSERT_ANY_THROW(place->getConsumingOperations());
    ASSERT_ANY_THROW(place->getTargetTensor());
    ASSERT_ANY_THROW(place->getSourceTensor());
    ASSERT_ANY_THROW(place->getProducingOperation());
    ASSERT_ANY_THROW(place->getProducingPort());
    ASSERT_ANY_THROW(place->getInputPort());
    ASSERT_ANY_THROW(place->getInputPort(""));
    ASSERT_ANY_THROW(place->getOutputPort());
    ASSERT_ANY_THROW(place->getOutputPort(""));
    ASSERT_ANY_THROW(place->getConsumingPorts());
    ASSERT_ANY_THROW(place->isInput());
    ASSERT_ANY_THROW(place->isOutput());
    ASSERT_ANY_THROW(place->isEqual(nullptr));
    ASSERT_ANY_THROW(place->isEqualData(nullptr));
}
