// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mha_inst.h"
#include "primitive_onednn_graph_base.hpp"
#include "implementation_map.hpp"

#include "impls/ocl/kernel_selector_helper.h"

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.h>
#include <oneapi/dnnl/dnnl_ocl.hpp>

#include <algorithm>
#include <memory>
#include <cmath>
#include <ostream>

using namespace dnnl::graph;

namespace cldnn {
namespace onednn {

struct mha_graph_onednn : typed_primitive_onednn_graph_impl<mha> {
    using parent = typed_primitive_onednn_graph_impl<mha>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::mha_graph_onednn)

    mha_graph_onednn(const engine& engine,
            const ExecutionConfig& config,
            const std::vector<cldnn::layout>& input_layouts,
            const cldnn::layout& output_layout)
        : parent(engine, config, input_layouts, output_layout) {
            build_graph(config);
    }

    void build_graph(const ExecutionConfig& config) {
        auto q_dims = get_logical_tensor_dims(_input_layouts[0]);
        auto k_dims = get_logical_tensor_dims(_input_layouts[1]);
        auto v_dims = get_logical_tensor_dims(_input_layouts[2]);
        auto sv_dims = get_logical_tensor_dims(_output_layout);

        /**
         * Matching pattern : (Matrices Q,K,V)
         *                       Q(1)   K(2)
         *                       |  /
         *                     MatMul
         *                       |
         *                    Softmax  V(3)
         *                       |    /
         *                       |   /
         *                     Matmul
         */

        logical_tensor q_desc {0, logical_tensor::data_type::f16, q_dims, logical_tensor::layout_type::strided};
        logical_tensor k_desc {1, logical_tensor::data_type::f16, v_dims, logical_tensor::layout_type::strided};
        logical_tensor v_desc {2, logical_tensor::data_type::f16, k_dims, logical_tensor::layout_type::strided};
        logical_tensor qk_matmul_desc {3, logical_tensor::data_type::f16};
        logical_tensor softmax_desc {4, logical_tensor::data_type::f16};
        logical_tensor sv_desc {1000000, logical_tensor::data_type::f16, q_dims, logical_tensor::layout_type::strided};

        op qk_matmul{0, op::kind::MatMul, {q_desc, k_desc}, {qk_matmul_desc}};
        op softmax {1, op::kind::SoftMax, {qk_matmul_desc}, {softmax_desc}};
        softmax.set_attr<int64_t>(op::attr::axis, -1);
        op sv_matmul{2, op::kind::MatMul, {softmax_desc, v_desc}, {sv_desc}};

        auto g = std::make_shared<dnnl::graph::graph>(dnnl::engine::kind::gpu);
        g->add_op(qk_matmul);
        g->add_op(softmax);
        g->add_op(sv_matmul);
        g->finalize();

        _graph = g;

        _concrete_tensors = {
            {0, {0, q_desc.get_data_type(), q_dims, logical_tensor::layout_type::strided}},
            {1, {1, k_desc.get_data_type(), k_dims, logical_tensor::layout_type::strided}},
            {2, {2, v_desc.get_data_type(), v_dims, logical_tensor::layout_type::strided}},
        };

        auto dnnl_engine = _engine->get_onednn_engine();
        auto partitions = g->get_partitions();

        std::unordered_set<size_t> ids_with_any_layout;
        set_any_layout(partitions, ids_with_any_layout);

        std::unordered_map<size_t, logical_tensor> id_to_queried_logical_tensors;
        for (auto& p : partitions) {
            if (!p.is_supported()) {
                continue;
            }

            auto inputs = p.get_input_ports();
            auto outputs = p.get_output_ports();

            // Update input logical tensors with concrete logical tensors
            for (auto &input : inputs) {
                const auto id = input.get_id();
                // If the tensor is an output of another partition, use the cached
                // logical tensor
                if (id_to_queried_logical_tensors.find(id)
                        != id_to_queried_logical_tensors.end()) {
                    input = id_to_queried_logical_tensors[id];
                } else {
                    // Create logical tensor with strided layout
                    input = _concrete_tensors[id];
                }
            }

            // Update output logical tensors with concrete shape and layout
            for (auto &output : outputs) {
                const auto id = output.get_id();
                output = logical_tensor {id, output.get_data_type(),
                        DNNL_GRAPH_UNKNOWN_NDIMS, // set output dims to unknown
                        ids_with_any_layout.count(id) ? logical_tensor::layout_type::any
                                                    : logical_tensor::layout_type::strided};
            }

            compiled_partition cp = p.compile(inputs, outputs, dnnl_engine);

            // Update output logical tensors with queried one
            for (auto &output : outputs) {
                const auto id = output.get_id();
                output = cp.query_logical_tensor(id);
                id_to_queried_logical_tensors[id] = output;
            }

            _compiled_partitions.push_back({p, cp, inputs, outputs});
        }

        std::vector<size_t> queried_tensor_ids;
        for (auto& kvp : id_to_queried_logical_tensors) {
            if (kvp.first == 1000000) // id of output
                continue;
            queried_tensor_ids.push_back(kvp.first);
        }
        std::sort(queried_tensor_ids.begin(), queried_tensor_ids.end());

        _id_to_queried_logical_tensors = std::move(id_to_queried_logical_tensors);
        _intermediate_to_ids = std::move(queried_tensor_ids);
    }

    static std::unique_ptr<primitive_impl> create(const mha_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();

        return cldnn::make_unique<mha_graph_onednn>(engine, config,
            impl_params.input_layouts, impl_params.output_layouts[0]);
    }

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<mha_graph_onednn>(*this);
    }
};

namespace detail {

attach_mha_graph_onednn::attach_mha_graph_onednn() {
    std::vector<data_types> dt = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
    };
    std::vector<format::type> fmt = {
        format::bfyx,
    };
    implementation_map<mha>::add(impl_types::onednn, mha_graph_onednn::create, dt, fmt);
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::mha_graph_onednn)
