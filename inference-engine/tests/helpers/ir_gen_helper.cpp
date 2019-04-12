//
// Copyright 2017-2018 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "ir_gen_helper.hpp"

namespace single_layer_tests {

    std::string IRTemplateGenerator::getIRTemplate(const std::string& name,
                                  const std::vector<size_t>& input_shape,
                                  const std::string& precision,
                                  const std::string& layers, 
                                  const std::string& edges,
                                  const unsigned ir_version) {
        std::string model = model_t;
        REPLACE_WITH_STR(model, "_NAME_", name);
        REPLACE_WITH_NUM(model, "_IRv_", ir_version);
        REPLACE_WITH_STR(model, "_PR_", precision);

        std::string s_dims;
        for (auto& dim : input_shape) {
            s_dims += "\n                    <dim>";
            s_dims += std::to_string(dim) + "</dim>";
        }
        REPLACE_WITH_STR(model, "__SRC_DIMS__", s_dims);
        REPLACE_WITH_STR(model, "_LAYERS_", layers);
        REPLACE_WITH_STR(model, "_EDGES_", edges);

        return model;
    }

    std::string IRTemplateGenerator::model_t = R"V0G0N(
        <net name="_NAME_" version="_IRv_" precision="_PR_" batch="1">
            <layers>
                <layer name="in1" type="Input" precision="_PR_" id="0">
                    <output>
                        <port id="0">__SRC_DIMS__
                        </port>
                    </output>
                </layer>
                _LAYERS_
            </layers>
            <edges>
                _EDGES_
            </edges>
        </net>
        )V0G0N";
}