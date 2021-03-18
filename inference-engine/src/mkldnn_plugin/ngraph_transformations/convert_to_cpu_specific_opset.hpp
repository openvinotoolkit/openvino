// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/pass/constant_folding.hpp>
#include "convert_matmul_to_fc_or_gemm.hpp"
#include "fc_bias_fusion.hpp"
#include "reshape_fc_fusion.hpp"
#include "reshape_fully_connected.hpp"

namespace MKLDNNPlugin {

inline void ConvertToCPUSpecificOpset(std::shared_ptr<ngraph::Function> &nGraphFunc) {
    ngraph::pass::Manager manager;
    manager.register_pass<ConvertMatMulToFC>();
    manager.register_pass<ConvertMatMulToGemm>();
    manager.register_pass<FullyConnectedBiasFusion>();
    manager.register_pass<ReshapeFullyConnected>();
    if (!ngraph::op::util::has_op_with_type<ngraph::op::FakeQuantize>(nGraphFunc)) {
        manager.register_pass<ReshapeFullyConnectedFusion>();
    }
    manager.register_pass<ngraph::pass::ConstantFolding>();
    manager.register_pass<ngraph::pass::ConvertPrecision>(ngraph::element::i64, ngraph::element::i32);
    manager.run_passes(nGraphFunc);
}

}  // namespace MKLDNNPlugin