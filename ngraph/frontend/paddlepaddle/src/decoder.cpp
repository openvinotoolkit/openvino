// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "framework.pb.h"

#include "decoder.hpp"

namespace ngraph
{
    namespace frontend
    {
        using namespace paddle::framework;

        std::map<paddle::framework::proto::VarType_Type, ngraph::element::Type> TYPE_MAP{
            {proto::VarType_Type::VarType_Type_BOOL, ngraph::element::boolean},
            {proto::VarType_Type::VarType_Type_INT16, ngraph::element::i16},
            {proto::VarType_Type::VarType_Type_INT32, ngraph::element::i32},
            {proto::VarType_Type::VarType_Type_INT64, ngraph::element::i64},
            {proto::VarType_Type::VarType_Type_FP16, ngraph::element::f16},
            {proto::VarType_Type::VarType_Type_FP32, ngraph::element::f32},
            {proto::VarType_Type::VarType_Type_FP64, ngraph::element::f64},
            {proto::VarType_Type::VarType_Type_UINT8, ngraph::element::u8},
            {proto::VarType_Type::VarType_Type_INT8, ngraph::element::i8},
            {proto::VarType_Type::VarType_Type_BF16, ngraph::element::bf16}};

        std::shared_ptr<Variant>
            DecoderPDPDProto::get_attribute(const std::string& name,
                                            const VariantTypeInfo& type_info) const
        {
            auto attrs = decode_attribute_helper(name);
            if (attrs.empty())
            {
                return nullptr;
            }

            if (type_info == VariantWrapper<std::string>::type_info)
            {
                return std::make_shared<VariantWrapper<std::string>>(attrs[0].s());
            }
            else if (type_info == VariantWrapper<int64_t>::type_info)
            {
                return std::make_shared<VariantWrapper<int64_t>>(attrs[0].l());
            }
            else if (type_info == VariantWrapper<std::vector<int64_t>>::type_info)
            {
                auto longs = std::vector<int64_t>(attrs[0].longs().begin(), attrs[0].longs().end());
                return std::make_shared<VariantWrapper<std::vector<int64_t>>>(longs);
            }
            else if (type_info == VariantWrapper<int32_t>::type_info)
            {
                return std::make_shared<VariantWrapper<int32_t>>(attrs[0].i());
            }
            else if (type_info == VariantWrapper<std::vector<int32_t>>::type_info)
            {
                auto ints = std::vector<int32_t>(attrs[0].ints().begin(), attrs[0].ints().end());
                return std::make_shared<VariantWrapper<std::vector<int32_t>>>(ints);
            }
            else if (type_info == VariantWrapper<float>::type_info)
            {
                return std::make_shared<VariantWrapper<float>>(attrs[0].f());
            }
            else if (type_info == VariantWrapper<std::vector<float>>::type_info)
            {
                auto floats =
                    std::vector<float>(attrs[0].floats().begin(), attrs[0].floats().end());
                return std::make_shared<VariantWrapper<std::vector<float>>>(floats);
            }
            else if (type_info == VariantWrapper<ngraph::element::Type>::type_info)
            {
                auto data_type = (paddle::framework::proto::VarType_Type)attrs[0].i();
                return std::make_shared<VariantWrapper<ngraph::element::Type>>(TYPE_MAP[data_type]);
            }
            else if (type_info == VariantWrapper<bool>::type_info)
            {
                return std::make_shared<VariantWrapper<bool>>(attrs[0].b());
            }

            // Type is not supported by decoder
            return nullptr;
        }

        std::vector<pdpd::OutPortName> DecoderPDPDProto::get_output_names() const
        {
            std::vector<std::string> output_names;
            for (const auto& output : op_place->getDesc()->outputs())
            {
                output_names.push_back(output.parameter());
            }
            return output_names;
        }

        ngraph::element::Type
            DecoderPDPDProto::get_out_port_type(const std::string& port_name) const
        {
            std::vector<ngraph::element::Type> output_types;
            for (const auto& out_port : op_place->getOutputPorts().at(port_name))
            {
                output_types.push_back(out_port->getTargetTensorPDPD()->getElementType());
            }
            FRONT_END_GENERAL_CHECK(output_types.size() > 0, "Port has no tensors connected.");
            FRONT_END_GENERAL_CHECK(
                std::equal(output_types.begin() + 1, output_types.end(), output_types.begin()),
                "Port has tensors with different types connected.");
            return output_types[0];
        }

        std::string DecoderPDPDProto::get_op_type() const { return op_place->getDesc()->type(); }

        std::vector<proto::OpDesc_Attr>
            DecoderPDPDProto::decode_attribute_helper(const std::string& name) const
        {
            std::vector<proto::OpDesc_Attr> attrs;
            for (const auto& attr : op_place->getDesc()->attrs())
            {
                if (attr.name() == name)
                    attrs.push_back(attr);
            }
            FRONT_END_GENERAL_CHECK(attrs.size() <= 1,
                                    "An error occurred while parsing the ",
                                    name,
                                    " attribute of ",
                                    op_place->getDesc()->type(),
                                    "node. Unsupported number of attributes. Current number: ",
                                    attrs.size(),
                                    " Expected number: 0 or 1");
            return attrs;
        }
    } // namespace frontend
} // namespace ngraph