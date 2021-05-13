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

        ngraph::element::Type DecoderPDPDProto::get_dtype(const std::string& name,
                                                          ngraph::element::Type def) const
        {
            auto dtype = (paddle::framework::proto::VarType_Type)get_int(name);
            return TYPE_MAP[dtype];
        }

        std::vector<int32_t> DecoderPDPDProto::get_ints(const std::string& name,
                                                        const std::vector<int32_t>& def) const
        {
            std::cout << "Running get_ints" << std::endl;
            std::vector<proto::OpDesc_Attr> attrs;
            for (const auto& attr : op_place->getDesc()->attrs())
            {
                if (attr.name() == name)
                    attrs.push_back(attr);
            }
            if (attrs.size() == 0)
            {
                return def;
            }
            else if (attrs.size() > 1)
            {
                // TODO: raise exception here
                return def;
            }
            else
            {
                return std::vector<int32_t>(attrs[0].ints().begin(), attrs[0].ints().end());
            }
        }

        int DecoderPDPDProto::get_int(const std::string& name, int def) const
        {
            std::vector<proto::OpDesc_Attr> attrs;
            for (const auto& attr : op_place->getDesc()->attrs())
            {
                if (attr.name() == name)
                    attrs.push_back(attr);
            }
            if (attrs.size() == 0)
            {
                return def;
            }
            else if (attrs.size() > 1)
            {
                // TODO: raise exception here
                return def;
            }
            else
            {
                return attrs[0].i();
            }
        }

        std::vector<float> DecoderPDPDProto::get_floats(const std::string& name,
                                                        const std::vector<float>& def) const
        {
            std::vector<proto::OpDesc_Attr> attrs;
            for (const auto& attr : op_place->getDesc()->attrs())
            {
                if (attr.name() == name)
                {
                    attrs.push_back(attr);
                    std::cout << attr.type() << std::endl;
                }
            }
            if (attrs.size() == 0)
            {
                return def;
            }
            else if (attrs.size() > 1)
            {
                // TODO: raise exception here
                return def;
            }
            else
            {
                return std::vector<float>(attrs[0].floats().begin(), attrs[0].floats().end());
            }
        }

        float DecoderPDPDProto::get_float(const std::string& name, float def) const
        {
            std::vector<proto::OpDesc_Attr> attrs;
            for (const auto& attr : op_place->getDesc()->attrs())
            {
                if (attr.name() == name)
                    attrs.push_back(attr);
            }
            if (attrs.size() == 0)
            {
                return def;
            }
            else if (attrs.size() > 1)
            {
                // TODO: raise exception here
                return def;
            }
            else
            {
                return attrs[0].f();
            }
        }

        std::string DecoderPDPDProto::get_str(const std::string& name, const std::string& def) const
        {
            std::vector<proto::OpDesc_Attr> attrs;
            for (const auto& attr : op_place->getDesc()->attrs())
            {
                if (attr.name() == name)
                    attrs.push_back(attr);
            }
            if (attrs.size() == 0)
            {
                return def;
            }
            else if (attrs.size() > 1)
            {
                // TODO: raise exception here
                return def;
            }
            else
            {
                return attrs[0].s();
            }
        }

        bool DecoderPDPDProto::get_bool(const std::string& name, bool def) const
        {
            std::vector<proto::OpDesc_Attr> attrs;
            for (const auto& attr : op_place->getDesc()->attrs())
            {
                if (attr.name() == name)
                    attrs.push_back(attr);
            }
            if (attrs.size() == 0)
            {
                return def;
            }
            else if (attrs.size() > 1)
            {
                // TODO: raise exception here
                return def;
            }
            else
            {
                return attrs[0].b();
            }
        }

        std::vector<int64_t> DecoderPDPDProto::get_longs(const std::string& name,
                                                         const std::vector<int64_t>& def) const
        {
            std::cout << "Running get_longs" << std::endl;
            std::vector<proto::OpDesc_Attr> attrs;
            for (const auto& attr : op_place->getDesc()->attrs())
            {
                if (attr.name() == name)
                    attrs.push_back(attr);
            }
            if (attrs.empty())
            {
                return def;
            }
            else if (attrs.size() > 1)
            {
                // TODO: raise exception here
                return def;
            }
            else
            {
                return std::vector<int64_t>(attrs[0].longs().begin(), attrs[0].longs().end());
            }
        }

        int64_t DecoderPDPDProto::get_long(const std::string& name, const int64_t& def) const
        {
            std::cout << "Running get_long" << std::endl;
            std::vector<proto::OpDesc_Attr> attrs;
            for (const auto& attr : op_place->getDesc()->attrs())
            {
                if (attr.name() == name)
                    attrs.push_back(attr);
            }
            if (attrs.empty())
            {
                return def;
            }
            else if (attrs.size() > 1)
            {
                // TODO: raise exception here
                return def;
            }
            else
            {
                return attrs[0].l();
            }
        }

        std::vector<std::string> DecoderPDPDProto::get_output_names() const
        {
            std::vector<std::string> output_names;
            for (const auto& output : op_place->getDesc()->outputs())
            {
                output_names.push_back(output.parameter());
            }
            return output_names;
        }

        std::vector<ngraph::element::Type>
            DecoderPDPDProto::get_out_port_types(const std::string& port_name) const
        {
            std::vector<ngraph::element::Type> output_types;
            for (const auto& out_port : op_place->getOutputPorts().at(port_name))
            {
                output_types.push_back(out_port->getTargetTensorPDPD()->getElementType());
            }
            return output_types;
        }

    } // namespace frontend
} // namespace ngraph