// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_test_utils.hpp"

#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/subtract.hpp"
#include "plugin/transformations/sdpa_kv_compression_fusion.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov::test::intel_gpu {

// KV cache stored as symmetric int8: Multiply(Convert(quant), scale) on both K and V is folded into a
// compressed op::SDPA that consumes the int8 cache directly plus the dequantization scales.
TEST_F(TransformationTestsF, SDPAKVCompressionFusionQuantizedKVInt8) {
    const bool is_causal = true;
    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto k_quant = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::PartialShape::dynamic(4));
        auto k_scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto v_quant = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::PartialShape::dynamic(4));
        auto v_scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));

        auto k_convert = std::make_shared<ov::op::v0::Convert>(k_quant, ov::element::f16);
        auto k_deq = std::make_shared<ov::op::v1::Multiply>(k_convert, k_scale);
        auto v_convert = std::make_shared<ov::op::v0::Convert>(v_quant, ov::element::f16);
        auto v_deq = std::make_shared<ov::op::v1::Multiply>(v_convert, v_scale);

        const auto order = ov::intel_gpu::op::SDPA::default_order(4);
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(
            ov::OutputVector{q, k_deq, v_deq},
            is_causal,
            order,
            order,
            order,
            order);

        model = std::make_shared<ov::Model>(ov::OutputVector{sdpa},
                                            ov::ParameterVector{q, k_quant, k_scale, v_quant, v_scale});
        manager.register_pass<SDPAKVCompressionFusion>();
    }
    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto k_quant = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::PartialShape::dynamic(4));
        auto k_scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto v_quant = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::PartialShape::dynamic(4));
        auto v_scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));

        const std::vector<int64_t> order = {0, 1, 2, 3};
        ov::intel_gpu::op::SDPA::QuantizationAttribute config;
        config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Symmetric;
        config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
        config.quantization_dt = ov::element::i8;
        config.scale_dt = ov::element::f16;
        config.group_sizes = {1, 1, 1, 1};
        config.scales_zp_output_order = {0, 1, 2, 3};

        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(
            ov::OutputVector{q, k_quant, v_quant, k_scale, v_scale},
            is_causal,
            order,
            order,
            order,
            order,
            config);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{sdpa},
                                                ov::ParameterVector{q, k_quant, k_scale, v_quant, v_scale});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

// KV cache stored as asymmetric int4: Multiply(Subtract(Convert(quant), zp), scale) on both K and V is folded
// into a compressed op::SDPA carrying the int4 cache, scales and zero points.
TEST_F(TransformationTestsF, SDPAKVCompressionFusionQuantizedKVInt4Asymmetric) {
    const bool is_causal = true;
    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto k_quant = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::PartialShape::dynamic(4));
        auto k_zp = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto k_scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto v_quant = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::PartialShape::dynamic(4));
        auto v_zp = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto v_scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));

        auto k_convert = std::make_shared<ov::op::v0::Convert>(k_quant, ov::element::f16);
        auto k_sub = std::make_shared<ov::op::v1::Subtract>(k_convert, k_zp);
        auto k_deq = std::make_shared<ov::op::v1::Multiply>(k_sub, k_scale);
        auto v_convert = std::make_shared<ov::op::v0::Convert>(v_quant, ov::element::f16);
        auto v_sub = std::make_shared<ov::op::v1::Subtract>(v_convert, v_zp);
        auto v_deq = std::make_shared<ov::op::v1::Multiply>(v_sub, v_scale);

        const auto order = ov::intel_gpu::op::SDPA::default_order(4);
        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{q, k_deq, v_deq},
                                                              is_causal,
                                                              order,
                                                              order,
                                                              order,
                                                              order);

        model = std::make_shared<ov::Model>(
            ov::OutputVector{sdpa},
            ov::ParameterVector{q, k_quant, k_zp, k_scale, v_quant, v_zp, v_scale});
        manager.register_pass<SDPAKVCompressionFusion>();
    }
    {
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto k_quant = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::PartialShape::dynamic(4));
        auto k_zp = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto k_scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto v_quant = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::PartialShape::dynamic(4));
        auto v_zp = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));
        auto v_scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape::dynamic(4));

        const std::vector<int64_t> order = {0, 1, 2, 3};
        ov::intel_gpu::op::SDPA::QuantizationAttribute config;
        config.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
        config.quantization_dt = ov::element::u4;
        config.scale_dt = ov::element::f16;
        config.zp_dt = ov::element::f16;
        config.group_sizes = {1, 1, 1, 1};
        config.scales_zp_output_order = {0, 1, 2, 3};

        auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(
            ov::OutputVector{q, k_quant, v_quant, k_scale, v_scale, k_zp, v_zp},
            is_causal,
            order,
            order,
            order,
            order,
            config);

        model_ref = std::make_shared<ov::Model>(
            ov::OutputVector{sdpa},
            ov::ParameterVector{q, k_quant, k_zp, k_scale, v_quant, v_zp, v_scale});
        comparator.enable(FunctionsComparator::ATTRIBUTES);
    }
}

}  // namespace ov::test::intel_gpu
