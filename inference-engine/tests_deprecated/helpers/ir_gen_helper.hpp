// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef IR_GEN_HELPER_HPP
#define IR_GEN_HELPER_HPP

#include "single_layer_common.hpp"

namespace single_layer_tests {

    class IRTemplateGenerator {
        IRTemplateGenerator() = default;
    public:
        static std::string model_t;
        static std::string model_input_t;

        static std::string getIRTemplate(const std::string& name,
                                  const std::vector<size_t>& input_shape,
                                  const std::string& precision,
                                  const std::string& layers,
                                  const std::string& edges,
                                  const unsigned ir_version = 5u,
                                  const std::string& metadata = "");



        static std::string getIRTemplate(const std::string& name,
            const std::vector<std::vector<size_t>>& input_shape,
            const std::string& precision,
            const std::string& layers,
            const std::string& edges,
            const unsigned ir_version = 5u,
            const std::string& metadata = "");

    };

} // namespace single_layer_tests
#endif /* IR_GEN_HELPER_HPP */

