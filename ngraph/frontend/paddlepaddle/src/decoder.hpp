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
#include "node_context.hpp"

#include <ngraph/ngraph.hpp>

namespace ngraph
{
    namespace frontend
    {
        extern std::map<paddle::framework::proto::VarType_Type, ngraph::element::Type> TYPE_MAP;

        class DecoderPDPDProto : public pdpd::DecoderBase
        {
        public:
            explicit DecoderPDPDProto(const std::shared_ptr<OpPlacePDPD>& op)
                : op_place(op)
            {
            }

            std::shared_ptr<Variant> get_attribute(const std::string& name,
                                                   const VariantTypeInfo& type_info) const override;

            std::vector<pdpd::OutPortName> get_output_names() const override;

            size_t get_output_size() const override;

            ngraph::element::Type get_out_port_type(const std::string& port_name) const override;

            std::string get_op_type() const override;

            std::map<std::string, std::vector<ngraph::element::Type>> get_output_type_map() const;

            std::map<std::string, OutputVector> map_for_each_input(
                const std::function<Output<Node>(const std::string&, size_t)>& func) const;

            std::map<std::string, OutputVector> map_for_each_output(
                const std::function<Output<Node>(const std::string&, size_t)>& func) const;

        private:
            std::vector<paddle::framework::proto::OpDesc_Attr>
                decode_attribute_helper(const std::string& name) const;
            std::shared_ptr<OpPlacePDPD> op_place;
        };

    } // namespace frontend
} // namespace ngraph
