// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "ir_gen_helper.hpp"

namespace single_layer_tests {

    std::string IRTemplateGenerator::getIRTemplate(const std::string& name,
        const std::vector<size_t>& input_shape,
        const std::string& precision,
        const std::string& layers,
        const std::string& edges,
        const unsigned ir_version,
        const std::string& metadata) {
        const std::vector< std::vector<size_t>> input_shape_vector = { input_shape };
        return getIRTemplate(name, input_shape_vector, precision, layers, edges, ir_version, metadata);
    }
    std::string IRTemplateGenerator::getIRTemplate(const std::string& name,
        const std::vector<std::vector<size_t>>& input_shape,
        const std::string& precision,
        const std::string& layers,
        const std::string& edges,
        const unsigned ir_version,
        const std::string& metadata) {
        std::string model = model_t;
        REPLACE_WITH_STR(model, "_NAME_", name);
        REPLACE_WITH_NUM(model, "_IRv_", ir_version);
        std::string input_layers;
        for (int i = 0; i < input_shape.size(); i++) {
            std::string model_input = model_input_t;
            std::string s_dims;
            for (auto& dim : input_shape[0]) {
                s_dims += "\n\t                    <dim>";
                s_dims += std::to_string(dim) + "</dim>";
            }
            REPLACE_WITH_STR(model_input, "_ID_", std::to_string(i));
            std::string input_name = "in" + std::to_string(i + 1);
            REPLACE_WITH_STR(model_input, "_input_name_", input_name);
            REPLACE_WITH_STR(model_input, "__SRC_DIMS__", s_dims);
            input_layers += model_input;
        }
        REPLACE_WITH_STR(model, "__INPUT_LAYERS_", input_layers);
        REPLACE_WITH_STR(model, "_PR_", precision);
        REPLACE_WITH_STR(model, "_LAYERS_", layers);
        REPLACE_WITH_STR(model, "_EDGES_", edges);
        REPLACE_WITH_STR(model, "_META_DATA_", metadata);

        return model;
    }

    std::string IRTemplateGenerator::model_input_t = R"V0G0N(
                <layer name="_input_name_" type="Input" precision="_PR_" id="_ID_">
                    <output>
                        <port id="0">__SRC_DIMS__
                        </port>
                    </output>
                </layer>

        )V0G0N";

    std::string IRTemplateGenerator::model_t = R"V0G0N(
        <net name="_NAME_" version="_IRv_" precision="_PR_" batch="1">
            <layers>
                __INPUT_LAYERS_
                _LAYERS_
            </layers>
            <edges>
                _EDGES_
            </edges>

            <meta_data>
                _META_DATA_
            </meta_data>

        </net>
        )V0G0N";
}  // namespace single_layer_tests
