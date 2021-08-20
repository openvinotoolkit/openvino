// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <frontend_manager/frontend.hpp>

#define NGRAPH_HELPER_DLL_EXPORT __declspec(dllexport)

#define TF_API NGRAPH_HELPER_DLL_EXPORT


namespace ngraph {
namespace frontend {
class PlaceTensorflow : public Place {
public:
    std::string name;
    enum Kind { PORT_INPUT, PORT_OUTPUT, TENSOR, OP } kind;
    size_t port;

    PlaceTensorflow(const std::string& _name, Kind _kind = OP, size_t _port = 0)
        : name(_name),
          kind(_kind),
          port(_port) {}

    virtual std::vector<std::string> get_names() const override {
        return {name};
    }

    virtual bool is_equal(Ptr another) const override {
        auto another_tf = std::dynamic_pointer_cast<PlaceTensorflow>(another);
        return another_tf && name == another_tf->name && kind == another_tf->kind && port == another_tf->port;
    }
};

}  // namespace frontend

}  // namespace ngraph
