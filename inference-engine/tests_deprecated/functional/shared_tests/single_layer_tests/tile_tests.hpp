// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "tests_common.hpp"
#include "single_layer_common.hpp"

#include <ie_core.hpp>
#include "ir_gen_helper.hpp"
#include "common_test_utils/data_utils.hpp"

using namespace ::testing;
using namespace InferenceEngine;
using namespace single_layer_tests;

struct tile_test_base_params {
    SizeVector shape_1;
    SizeVector shape_2;
    int axis;
    int tiles;
};

struct tile_test_params : public tile_test_base_params {
    std::string device_name;

    tile_test_params(std::string name, tile_test_base_params params)
        : device_name(name), tile_test_base_params(params) {}
};

class TileTest: public TestsCommon, public WithParamInterface<tile_test_params> {
    std::string model_t = R"V0G0N(
<net batch="1" name="tile_net" version="5">
	<layers>
		<layer id="0" name="data" precision="FP32" type="Input">
			<output>
				<port id="0">
                    _SHAPE_1_
				</port>
			</output>
		</layer>
		<layer id="1" name="tile" precision="FP32" type="Tile">
            <data axis="_AXIS_" tiles="_TILES_" />
            <input>
                <port id="0">
                    _SHAPE_1_
                </port>
            </input>
            <output>
				<port id="1">
                    _SHAPE_2_
				</port>
			</output>
		</layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::string shape_xml(const SizeVector shape) {
        std::string res;
        for (auto dim : shape)
            res += "<dim>" + std::to_string(dim) + "</dim>";
        return res;
    }

    std::string getModel() {
        auto p = ::testing::WithParamInterface<tile_test_params>::GetParam();


        auto shape_1_xml = shape_xml(p.shape_1);
        auto shape_2_xml = shape_xml(p.shape_2);

        std::string model = model_t;
        REPLACE_WITH_STR(model, "_SHAPE_1_", shape_1_xml);
        REPLACE_WITH_STR(model, "_SHAPE_2_", shape_2_xml);
        REPLACE_WITH_NUM(model, "_AXIS_", p.axis);
        REPLACE_WITH_NUM(model, "_TILES_", p.tiles);

        return model;
    }

protected:
    virtual void SetUp() {
        try {
            auto p = GetParam();
            std::string model = getModel();

            InferenceEngine::Core ie;
            auto network = ie.ReadNetwork(model, Blob::CPtr());
            auto exec = ie.LoadNetwork(network, p.device_name);
            auto req = exec.CreateInferRequest();

            auto in_blob = req.GetBlob("data");
            CommonTestUtils::fill_data_const(in_blob, 7);

            req.Infer();

            TensorDesc desc {Precision::FP32, p.shape_2, TensorDesc::getLayoutByDims(p.shape_2)};
            Blob::Ptr out_ref = make_shared_blob<float>(desc);
            out_ref->allocate();

            CommonTestUtils::fill_data_const(out_ref, 7);
            compare(*out_ref, *req.GetBlob("tile"));
        } catch (const InferenceEngine::details::InferenceEngineException &e) {
            FAIL() << e.what();
        }
    }
};

#define case_1  tile_test_base_params{ {1}, {5}, 0, 5 }
#define case_2  tile_test_base_params{ {2}, {6}, 0, 3 }
#define case_3  tile_test_base_params{ {1, 3}, {5, 3}, 0, 5 }
#define case_4  tile_test_base_params{ {1, 3}, {1, 6}, 1, 2 }
#define case_5  tile_test_base_params{ {1, 2, 3}, {5, 2, 3}, 0, 5 }
#define case_6  tile_test_base_params{ {1, 2, 3}, {1, 4, 3}, 1, 2 }
#define case_7  tile_test_base_params{ {1, 2, 3}, {1, 2, 6}, 2, 2 }
#define case_8  tile_test_base_params{ {1, 2, 3, 4}, {5, 2, 3, 4}, 0, 5 }
#define case_9  tile_test_base_params{ {1, 2, 3, 4}, {1, 4, 3, 4}, 1, 2 }
#define case_10 tile_test_base_params{ {1, 2, 3, 4}, {1, 2, 6, 4}, 2, 2 }
#define case_11 tile_test_base_params{ {1, 2, 3, 4}, {1, 2, 3, 8}, 3, 2 }
#define case_12 tile_test_base_params{ {1, 2, 3, 4, 2}, {5, 2, 3, 4, 2}, 0, 5 }
#define case_13 tile_test_base_params{ {1, 2, 3, 4, 2}, {1, 4, 3, 4, 2}, 1, 2 }
#define case_14 tile_test_base_params{ {1, 2, 3, 4, 2}, {1, 2, 6, 4, 2}, 2, 2 }
#define case_15 tile_test_base_params{ {1, 2, 3, 4, 2}, {1, 2, 3, 8, 2}, 3, 2 }
#define case_16 tile_test_base_params{ {1, 2, 3, 4, 2}, {1, 2, 3, 4, 4}, 4, 2 }

TEST_P(TileTest, TestsGeneralTile) {}
