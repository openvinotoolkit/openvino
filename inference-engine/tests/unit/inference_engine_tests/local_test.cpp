// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <single_layer_common.hpp>

#include <cpp/ie_cnn_net_reader.h>

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;

class LocaleTests : public ::testing::Test {
    std::string _model = R"V0G0N(
<net name="Power_Only" version="3" precision="FP32" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="power" type="Power" precision="FP32" id="1">
            <data scale="0.75"
                  shift="0.35"
                  power="0.5"/>

            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
        <layer name="sum" type="Eltwise" precision="FP32" id="2">
            <data coeff="0.77,0.33"/>
            <input>
                <port id="1">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
                <port id="2">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </input>
            <output>
                <port id="3">
                    <dim>_IN_</dim>
                    <dim>_IC_</dim>
                    <dim>_IH_</dim>
                    <dim>_IW_</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="1"/>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="2"/>
    </edges>
    <pre-process reference-layer-name="data">
        <channel id="0">
            <mean value="104.006"/>
            <scale value="0.1"/>
        </channel>
        <channel id="1">
            <mean value="116.668"/>
            <scale value="0.2"/>
        </channel>
        <channel id="2">
            <mean value="122.678"/>
            <scale value="0.3"/>
        </channel>
    </pre-process>

</net>
)V0G0N";

protected:
    std::string getModel() const {
        std::string model = _model;

        REPLACE_WITH_NUM(model, "_IN_", 2);
        REPLACE_WITH_NUM(model, "_IC_", 3);
        REPLACE_WITH_NUM(model, "_IH_", 4);
        REPLACE_WITH_NUM(model, "_IW_", 5);

        return model;
    }

    void testBody() const {
        CNNNetReader reader;

        // This model contains layers with float attributes.
        // Conversion from string may be affected by locale.
        auto model = getModel();
        reader.ReadNetwork(model.data(), model.length());
        auto net = reader.getNetwork();

        auto power_layer = dynamic_pointer_cast<PowerLayer>(net.getLayerByName("power"));
        ASSERT_EQ(power_layer->scale, 0.75f);
        ASSERT_EQ(power_layer->offset, 0.35f);
        ASSERT_EQ(power_layer->power, 0.5f);

        auto sum_layer = dynamic_pointer_cast<EltwiseLayer>(net.getLayerByName("sum"));
        std::vector<float> ref_coeff {0.77f, 0.33f};
        ASSERT_EQ(sum_layer->coeff, ref_coeff);

        auto info = net.getInputsInfo();
        auto preproc = info.begin()->second->getPreProcess();
        ASSERT_EQ(preproc[0]->stdScale, 0.1f);
        ASSERT_EQ(preproc[0]->meanValue, 104.006f);
    }
};

TEST_F(LocaleTests, WithRULocale) {
    setlocale(LC_ALL, "ru_RU.UTF-8");
    testBody();
    setlocale(LC_ALL, "");
}

TEST_F(LocaleTests, WithUSLocale) {
    setlocale(LC_ALL, "en_US.UTF-8");
    testBody();
    setlocale(LC_ALL, "");
}

TEST_F(LocaleTests, DISABLED_WithRULocaleCPP) {
    auto prev = std::locale();
    std::locale::global(std::locale("ru_RU.UTF-8"));
    testBody();
    std::locale::global(prev);
}

TEST_F(LocaleTests, DISABLED_WithUSLocaleCPP) {
    auto prev = std::locale();
    std::locale::global(std::locale("en_US.UTF-8"));
    testBody();
    std::locale::global(prev);
}
