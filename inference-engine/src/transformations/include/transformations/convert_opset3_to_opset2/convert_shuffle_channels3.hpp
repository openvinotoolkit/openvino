// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ie_api.h>

#include <ngraph/pass/graph_rewrite.hpp>

namespace ngraph {
namespace pass {

    class INFERENCE_ENGINE_API_CLASS(ConvertShuffleChannels3);

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::ConvertShuffleChannels3: public ngraph::pass::GraphRewrite {
public:
    ConvertShuffleChannels3() : GraphRewrite() {
        convert_shuffle_channels3();
    }

private:
    void convert_shuffle_channels3();
};
