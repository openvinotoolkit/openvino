// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <fstream>

#include "framework.pb.h"

#include "decoder.hpp"


namespace ngraph {
namespace frontend {

std::map<paddle::framework::proto::VarType_Type, ngraph::element::Type> TYPE_MAP{
        {proto::VarType_Type::VarType_Type_BOOL,  ngraph::element::boolean},
        {proto::VarType_Type::VarType_Type_INT16, ngraph::element::i16},
        {proto::VarType_Type::VarType_Type_INT32, ngraph::element::i32},
        {proto::VarType_Type::VarType_Type_INT64, ngraph::element::i64},
        {proto::VarType_Type::VarType_Type_FP16,  ngraph::element::f16},
        {proto::VarType_Type::VarType_Type_FP32,  ngraph::element::f32},
        {proto::VarType_Type::VarType_Type_FP64,  ngraph::element::f64},
        {proto::VarType_Type::VarType_Type_UINT8, ngraph::element::u8},
        {proto::VarType_Type::VarType_Type_INT8,  ngraph::element::i8},
        {proto::VarType_Type::VarType_Type_BF16,  ngraph::element::bf16}
};

ngraph::element::Type DecoderPDPDProto::get_dtype(const std::string& name, ngraph::element::Type def) const
{
    auto dtype = (paddle::framework::proto::VarType_Type)get_int(name);
    return TYPE_MAP[dtype];
}

std::vector<int32_t> DecoderPDPDProto::get_ints(const std::string& name, const std::vector<int32_t>& def) const
{
    std::cout << "Running get_ints" << std::endl;
    std::vector<proto::OpDesc_Attr> attrs;
    for (const auto &attr : op.attrs()) {
        if (attr.name() == name)
            attrs.push_back(attr);
    }
    if (attrs.size() == 0) {
        return def;
    } else if (attrs.size() > 1) {
        // TODO: raise exception here
        return def;
    } else {
        std::vector<int32_t> res;
        std::copy(attrs[0].ints().begin(), attrs[0].ints().end(), std::back_inserter(res));
        return res;
    }
}

int DecoderPDPDProto::get_int(const std::string& name, int def) const
{
    std::vector<proto::OpDesc_Attr> attrs;
    for (const auto &attr : op.attrs()) {
        if (attr.name() == name)
            attrs.push_back(attr);
    }
    if (attrs.size() == 0) {
        return def;
    } else if (attrs.size() > 1) {
        // TODO: raise exception here
        return def;
    } else {
        return attrs[0].i();
    }
}

std::vector<float> DecoderPDPDProto::get_floats(const std::string& name, const std::vector<float>& def) const
{
    std::vector<proto::OpDesc_Attr> attrs;
    for (const auto &attr : op.attrs()) {
        if (attr.name() == name) {
            attrs.push_back(attr);
            std::cout << attr.type() << std::endl;
        }
    }
    if (attrs.size() == 0) {
        return def;
    } else if (attrs.size() > 1) {
        // TODO: raise exception here
        return def;
    } else {
        std::vector<float> res;
        std::copy(attrs[0].floats().begin(), attrs[0].floats().end(), std::back_inserter(res));
        return res;
    }
}

float DecoderPDPDProto::get_float(const std::string& name, float def) const
{
    std::vector<proto::OpDesc_Attr> attrs;
    for (const auto &attr : op.attrs()) {
        if (attr.name() == name)
            attrs.push_back(attr);
    }
    if (attrs.size() == 0) {
        return def;
    } else if (attrs.size() > 1) {
        // TODO: raise exception here
        return def;
    } else {
        return attrs[0].f();
    }
}

std::string DecoderPDPDProto::get_str(const std::string& name, const std::string& def) const
{
    std::vector<proto::OpDesc_Attr> attrs;
    for (const auto &attr : op.attrs()) {
        if (attr.name() == name)
            attrs.push_back(attr);
    }
    if (attrs.size() == 0) {
        return def;
    } else if (attrs.size() > 1) {
        // TODO: raise exception here
        return def;
    } else {
        return attrs[0].s();
    }
}

bool DecoderPDPDProto::get_bool(const std::string& name, bool def) const
{
    std::vector<proto::OpDesc_Attr> attrs;
    for (const auto &attr : op.attrs()) {
        if (attr.name() == name)
            attrs.push_back(attr);
    }
    if (attrs.size() == 0) {
        return def;
    } else if (attrs.size() > 1) {
        // TODO: raise exception here
        return def;
    } else {
        return attrs[0].b();
    }
}

}
}