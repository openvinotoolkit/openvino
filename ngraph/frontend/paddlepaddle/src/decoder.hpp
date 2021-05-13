//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "framework.pb.h"

#include <paddlepaddle_frontend/frontend.hpp>
#include <paddlepaddle_frontend/place.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset6.hpp>

namespace ngraph
{
    namespace frontend
    {
        extern std::map<paddle::framework::proto::VarType_Type, ngraph::element::Type> TYPE_MAP;

        // TODO: Inherit from one of the ngraph classes
        class AttributeNotFound : public std::exception
        {
        };

        class DecoderPDPDProto
        {
            std::shared_ptr<OpPlacePDPD> op_place;

        public:
            explicit DecoderPDPDProto(const std::shared_ptr<OpPlacePDPD>& op)
                : op_place(op)
            {
            }

            std::vector<int32_t> get_ints(const std::string& name,
                                          const std::vector<int32_t>& def = {}) const;
            int get_int(const std::string& name, int def = 0) const;
            std::vector<float> get_floats(const std::string& name,
                                          const std::vector<float>& def = {}) const;
            float get_float(const std::string& name, float def = 0.) const;
            std::string get_str(const std::string& name, const std::string& def = "") const;
            bool get_bool(const std::string& name, bool def = false) const;
            std::vector<int64_t> get_longs(const std::string& name,
                                           const std::vector<int64_t>& def = {}) const;
            int64_t get_long(const std::string& name, const int64_t& def = {}) const;

            // TODO: Further populate get_XXX methods on demand
            ngraph::element::Type get_dtype(const std::string& name,
                                            ngraph::element::Type def) const;

            std::vector<std::string> get_output_names() const;
            std::vector<element::Type> get_out_port_types(const std::string& port_name) const;
        };

    } // namespace frontend
} // namespace ngraph
