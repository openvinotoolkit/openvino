// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

        class DecoderPDPDProto
        {
        public:
            explicit DecoderPDPDProto(const std::shared_ptr<OpPlacePDPD>& op)
                : op_place(op)
            {
            }

            // TODO: Further populate get_XXX methods on demand
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

            ngraph::element::Type get_dtype(const std::string& name,
                                            ngraph::element::Type def) const;

            const std::string& get_op_type() const { return op_place->getDesc()->type(); }
            std::vector<std::string> get_output_names() const;
            std::vector<element::Type> get_out_port_types(const std::string& port_name) const;

        private:
            std::vector<paddle::framework::proto::OpDesc_Attr>
                decode_attribute_helper(const std::string& name) const;
            std::shared_ptr<OpPlacePDPD> op_place;
        };

    } // namespace frontend
} // namespace ngraph
