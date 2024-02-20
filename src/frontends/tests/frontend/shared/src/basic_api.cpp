// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "basic_api.hpp"

#include "utils.hpp"

using namespace ov::frontend;

std::string FrontEndBasicTest::getTestCaseName(const testing::TestParamInfo<BasicTestParam>& obj) {
    std::string fe, path, fileName;
    std::tie(fe, path, fileName) = obj.param;
    return fe + "_" + FrontEndTestUtils::fileToTestName(fileName);
}

void FrontEndBasicTest::SetUp() {
    m_fem = FrontEndManager();  // re-initialize after setting up environment
    initParamTest();
}

void FrontEndBasicTest::initParamTest() {
    std::tie(m_feName, m_pathToModels, m_modelFile) = GetParam();
    m_modelFile = FrontEndTestUtils::make_model_path(m_pathToModels + m_modelFile);
}

void FrontEndBasicTest::doLoadFromFile() {
    std::tie(m_frontEnd, m_inputModel) = FrontEndTestUtils::load_from_file(m_fem, m_feName, m_modelFile);
}

TEST_P(FrontEndBasicTest, testLoadFromFile) {
    ASSERT_NO_THROW(doLoadFromFile());
    ASSERT_EQ(m_frontEnd->get_name(), m_feName);
    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = m_frontEnd->convert(m_inputModel));
    ASSERT_NE(model, nullptr);
}

TEST_P(FrontEndBasicTest, testInputModel_getInputsOutputs) {
    ASSERT_NO_THROW(doLoadFromFile());

    using CustomCheck = std::function<void(Place::Ptr)>;
    auto checkPlaces = [&](const std::vector<Place::Ptr>& places, CustomCheck cb) {
        EXPECT_GT(places.size(), 0);
        std::set<Place::Ptr> placesSet(places.begin(), places.end());
        EXPECT_EQ(placesSet.size(), places.size());
        std::for_each(places.begin(), places.end(), [&](Place::Ptr place) {
            ASSERT_NE(place, nullptr);
            std::vector<std::string> names;
            ASSERT_NO_THROW(names = place->get_names());
            EXPECT_GT(names.size(), 0);
            cb(place);
        });
    };
    std::vector<Place::Ptr> inputs;
    ASSERT_NO_THROW(inputs = m_inputModel->get_inputs());
    checkPlaces(inputs, [&](Place::Ptr place) {
        EXPECT_TRUE(place->is_input());
    });

    std::vector<Place::Ptr> outputs;
    ASSERT_NO_THROW(outputs = m_inputModel->get_outputs());
    checkPlaces(outputs, [&](Place::Ptr place) {
        EXPECT_TRUE(place->is_output());
    });
}

TEST_P(FrontEndBasicTest, testInputModel_getPlaceByTensorName) {
    ASSERT_NO_THROW(doLoadFromFile());

    auto testGetPlaceByTensorName = [&](const std::vector<Place::Ptr>& places) {
        EXPECT_GT(places.size(), 0);
        for (auto place : places) {
            ASSERT_NE(place, nullptr);
            std::vector<std::string> names;
            ASSERT_NO_THROW(names = place->get_names());
            for (auto name : names) {
                EXPECT_NE(name, std::string());
                Place::Ptr placeByName;
                ASSERT_NO_THROW(placeByName = m_inputModel->get_place_by_tensor_name(name));
                ASSERT_NE(placeByName, nullptr);
                EXPECT_TRUE(placeByName->is_equal(place));
            }
        }
    };

    std::vector<Place::Ptr> outputs;
    ASSERT_NO_THROW(outputs = m_inputModel->get_outputs());
    testGetPlaceByTensorName(outputs);

    std::vector<Place::Ptr> inputs;
    ASSERT_NO_THROW(inputs = m_inputModel->get_inputs());
    testGetPlaceByTensorName(inputs);
}

TEST_P(FrontEndBasicTest, testInputModel_overrideAll) {
    ASSERT_NO_THROW(doLoadFromFile());

    using GetPlaces = std::function<std::vector<Place::Ptr>()>;
    using OverridePlaces = std::function<void(const std::vector<Place::Ptr>&)>;
    auto verifyOverride = [](GetPlaces getCB, OverridePlaces overrideCB) {
        std::vector<Place::Ptr> places;
        ASSERT_NO_THROW(places = getCB());
        std::set<Place::Ptr> placesSet(places.begin(), places.end());

        auto placesReversed = places;
        std::reverse(placesReversed.begin(), placesReversed.end());
        ASSERT_NO_THROW(overrideCB(placesReversed));
        ASSERT_NO_THROW(places = getCB());
        EXPECT_GT(places.size(), 0);
        std::set<Place::Ptr> placesSetAfter(places.begin(), places.end());
        EXPECT_EQ(placesSet.size(), placesSet.size());
        std::for_each(places.begin(), places.end(), [&](Place::Ptr place) {
            EXPECT_GT(placesSet.count(place), 0);
        });
    };
    verifyOverride(
        [&]() {
            return m_inputModel->get_inputs();
        },
        [&](const std::vector<Place::Ptr>& p) {
            m_inputModel->override_all_inputs(p);
        });

    verifyOverride(
        [&]() {
            return m_inputModel->get_outputs();
        },
        [&](const std::vector<Place::Ptr>& p) {
            m_inputModel->override_all_outputs(p);
        });
}

TEST_P(FrontEndBasicTest, testInputModel_overrideAll_empty) {
    ASSERT_NO_THROW(doLoadFromFile());
    using GetPlaces = std::function<std::vector<Place::Ptr>()>;
    using OverrideEmpty = std::function<void(void)>;
    using CustomCheck = std::function<void(std::string)>;
    auto verifyOverride = [](GetPlaces getCB, OverrideEmpty overrideCB, CustomCheck customCB) {
        std::vector<Place::Ptr> places;
        std::vector<Place::Ptr> newPlaces;
        ASSERT_NO_THROW(places = getCB());
        ASSERT_NO_THROW(overrideCB());
        ASSERT_NO_THROW(newPlaces = getCB());
        ASSERT_EQ(newPlaces.size(), 0);
        std::for_each(places.begin(), places.end(), [&](Place::Ptr place) {
            std::vector<std::string> names;
            ASSERT_NO_THROW(names = place->get_names());
            for (auto name : names) {
                customCB(name);
            }
        });
    };
    verifyOverride(
        [&]() {
            return m_inputModel->get_outputs();
        },
        [&]() {
            m_inputModel->override_all_outputs({});
        },
        [&](const std::string& name) {
            EXPECT_FALSE(m_inputModel->get_place_by_tensor_name(name)->is_output());
        });

    verifyOverride(
        [&]() {
            return m_inputModel->get_inputs();
        },
        [&]() {
            m_inputModel->override_all_inputs({});
        },
        [&](const std::string& name) {
            EXPECT_FALSE(m_inputModel->get_place_by_tensor_name(name)->is_input());
        });
}
