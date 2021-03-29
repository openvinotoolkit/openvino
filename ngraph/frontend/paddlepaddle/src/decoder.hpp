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

#include "../include/paddlepaddle_frontend/frontend.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset6.hpp>

namespace ngraph {
namespace frontend {

using namespace google;
using namespace paddle::framework;

extern std::map<paddle::framework::proto::VarType_Type, ngraph::element::Type> TYPE_MAP;

class DecoderPDPDProto
{
    proto::OpDesc op;

public:

    DecoderPDPDProto (proto::OpDesc _op) : op(_op) {}

    std::vector<int32_t> get_ints(const std::string& name, const std::vector<int32_t>& def = {}) const;
    int get_int(const std::string& name, int def = 0) const;
    float get_float(const std::string& name, float def = 0.) const;
    std::string get_str(const std::string& name, const std::string& def = "") const;
    bool get_bool (const std::string& name, bool def = false) const;

    // TODO: Further populate get_XXX methods on demand
};

}
}
