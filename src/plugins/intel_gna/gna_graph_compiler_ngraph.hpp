// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gna2-model-api.h"
#include "backend/am_intel_dnn.hpp"
#include "backend/dnn_components.hpp"
#include "gna_data_types.hpp"

namespace ngraph {
class GNAFullyConnected;
class GNAElementWise;
class GNAConvolution;
class GNATraspose;
class GNACopy;
class GNAPWL;
}  // namespace ngraph

namespace GNAPluginNS {

    struct Gna2ModelBuilder {
        std::shared_ptr<GNAPluginNS::backend::AMIntelDNN> dnn;
        GNAPluginNS::backend::DnnComponents dnnComponents;
        std::shared_ptr<GNAPluginNS::gna_memory_type> gnamem;

        // Append new GNA Layer to Gna2Model
        void Append(ngraph::GNAFullyConnected& op);
        void Append(ngraph::GNAElementWise& op);
        void Append(ngraph::GNAConvolution& op);
        void Append(ngraph::GNATraspose& op);
        void Append(ngraph::GNACopy& op);
        void Append(ngraph::GNAPWL& op);


        Gna2Model Build();
    };

}  // namespace GNAPluginNS


/*
    for(auto& x: serializedNgraphModel) {
        gna2ModelBuilder.Append(x);

    }

    return gna2ModelBuilder.Build()

*/