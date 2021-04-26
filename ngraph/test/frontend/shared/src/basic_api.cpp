// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <regex>
#include "../include/basic_api.hpp"
#include "../include/utils.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

std::string FrontEndBasicTest::getTestCaseName(const testing::TestParamInfo<BasicTestParam> &obj) {
    std::string fe, path, fileName;
    std::tie(fe, path, fileName) = obj.param;
    return fe + "_" + FrontEndTestUtils::fileToTestName(fileName);
}

void FrontEndBasicTest::SetUp() {
    initParamTest();
}

void FrontEndBasicTest::initParamTest() {
    std::tie(m_feName, m_pathToModels, m_modelFile) = GetParam();
    m_modelFile = std::string(TEST_FILES) + m_pathToModels + m_modelFile;
    std::cout << "Model: " << m_modelFile << std::endl;
}

void FrontEndBasicTest::doLoadFromFile() {
    std::vector<std::string> frontends;
    ASSERT_NO_THROW(frontends = m_fem.availableFrontEnds());
    ASSERT_NO_THROW(m_frontEnd = m_fem.loadByFramework(m_feName));
    ASSERT_NO_THROW(m_inputModel = m_frontEnd.loadFromFile(m_modelFile));
}

TEST_P(FrontEndBasicTest, testLoadFromFile) {
    ASSERT_NO_THROW(doLoadFromFile());
    std::shared_ptr<ngraph::Function> function;
    ASSERT_NO_THROW(function = m_frontEnd.convert(m_inputModel));
    ASSERT_NE(function, nullptr);
//    std::cout << "Ordered ops names\n";
//    for (const auto &n : function->get_ordered_ops()) {
//        std::cout << "----" << n->get_friendly_name() << "---\n";
//    }
}

TEST_P(FrontEndBasicTest, testInputModel_getInputsOutputs)
{
    ASSERT_NO_THROW(doLoadFromFile());

    using CustomCheck = std::function<void(const Place&)>;
    auto checkPlaces = [&](const std::vector<Place>& places, CustomCheck cb) {
        EXPECT_GT(places.size(), 0);
        std::for_each(places.begin(), places.end(), [&](const Place& place) {
            std::vector<std::string> names;
            ASSERT_NO_THROW(names = place.getNames());
            EXPECT_GT(names.size(), 0);
            cb(place);
        });
    };
    std::vector<Place> inputs;
    ASSERT_NO_THROW(inputs = m_inputModel.getInputs());
    checkPlaces(inputs, [&](const Place& place) {
        EXPECT_TRUE(place.isInput());
    });

    std::vector<Place> outputs;
    ASSERT_NO_THROW(outputs = m_inputModel.getOutputs());
    checkPlaces(outputs, [&](const Place& place) {
        EXPECT_TRUE(place.isOutput());
    });
}

TEST_P(FrontEndBasicTest, testInputModel_getPlaceByTensorName)
{
    ASSERT_NO_THROW(doLoadFromFile());

    auto testGetPlaceByTensorName = [&](const std::vector<Place>& places) {
        EXPECT_GT(places.size(), 0);
        for (const auto& place : places) {
            std::vector<std::string> names;
            ASSERT_NO_THROW(names = place.getNames());
            for (auto name : names) {
                EXPECT_NE(name, std::string());
                Place placeByName;
                ASSERT_NO_THROW(placeByName = m_inputModel.getPlaceByTensorName(name));
                EXPECT_TRUE(placeByName.isEqual(place));
            }
        }
    };

    std::vector<Place> outputs;
    ASSERT_NO_THROW(outputs = m_inputModel.getOutputs());
    testGetPlaceByTensorName(outputs);

    std::vector<Place> inputs;
    ASSERT_NO_THROW(inputs = m_inputModel.getInputs());
    testGetPlaceByTensorName(inputs);
}

TEST_P(FrontEndBasicTest, testInputModel_overrideAll)
{
    ASSERT_NO_THROW(doLoadFromFile());

    using GetPlaces = std::function<std::vector<Place>()>;
    using OverridePlaces = std::function<void(const std::vector<Place>&)>;
    auto verifyOverride = [](GetPlaces getCB, OverridePlaces overrideCB) {
        std::vector<Place> places, placesAfter;
        ASSERT_NO_THROW(places = getCB());
        auto placesReversed = getCB();
        std::reverse(placesReversed.begin(), placesReversed.end());
        ASSERT_NO_THROW(overrideCB(placesReversed));
        ASSERT_NO_THROW(placesAfter = getCB());
        EXPECT_EQ(places.size(), placesAfter.size());
        std::for_each(places.begin(), places.end(), [&](const Place& place) {
            EXPECT_NE(std::find_if(placesAfter.begin(), placesAfter.end(),
                                   [&place](const Place& other) {
                return place.isEqual(other);
            }), placesAfter.end());
        });
    };
    verifyOverride([&]() { return m_inputModel.getInputs(); },
                   [&](const std::vector<Place>& p) { m_inputModel.overrideAllInputs(p); });

    verifyOverride([&]() { return m_inputModel.getOutputs(); },
                   [&](const std::vector<Place>& p) { m_inputModel.overrideAllOutputs(p); });
}

TEST_P(FrontEndBasicTest, testInputModel_overrideAll_empty)
{
    ASSERT_NO_THROW(doLoadFromFile());
    using GetPlaces = std::function<std::vector<Place>()>;
    using OverrideEmpty = std::function<void(void)>;
    using CustomCheck = std::function<void(std::string)>;
    auto verifyOverride = [](GetPlaces getCB, OverrideEmpty overrideCB, CustomCheck customCB) {
        std::vector<Place> places;
        std::vector<Place> newPlaces;
        ASSERT_NO_THROW(places = getCB());
        ASSERT_NO_THROW(overrideCB());
        ASSERT_NO_THROW(newPlaces = getCB());
        ASSERT_EQ(newPlaces.size(), 0);
        std::for_each(places.begin(), places.end(), [&](const Place& place) {
            std::vector<std::string> names;
            ASSERT_NO_THROW(names = place.getNames());
            for (const auto& name : names) {
                customCB(name);
            }
        });
    };
    verifyOverride([&]() { return m_inputModel.getOutputs(); },
                   [&]() { m_inputModel.overrideAllOutputs({}); },
                   [&](const std::string &name) {
                       EXPECT_FALSE(m_inputModel.getPlaceByTensorName(name).isOutput());
                   });

    verifyOverride([&]() { return m_inputModel.getInputs(); },
                   [&]() { m_inputModel.overrideAllInputs({}); },
                   [&](const std::string &name) {
                       EXPECT_FALSE(m_inputModel.getPlaceByTensorName(name).isInput());
                   });
}

TEST_P(FrontEndBasicTest, DISABLED_testInputModel_extractSubgraph)
{
    ASSERT_NO_THROW(doLoadFromFile());

    // TODO: not clear now
}

TEST_P(FrontEndBasicTest, DISABLED_testInputModel_setPartialShape)
{
    ASSERT_NO_THROW(doLoadFromFile());

    // TODO: not clear now
}
