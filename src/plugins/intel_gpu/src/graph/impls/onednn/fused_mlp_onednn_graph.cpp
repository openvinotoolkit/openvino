// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fused_mlp_onednn_graph.hpp"

#include "primitive_inst.h"
#include "runtime/ocl/ocl_common.hpp"

#include <oneapi/dnnl/dnnl_graph.hpp>
#include "openvino/core/except.hpp"

#include <memory>
#include <utility>
#include <vector>

namespace cldnn {
namespace onednn {

namespace dg = dnnl::graph;

namespace {

static dg::logical_tensor::data_type to_graph_dt(data_types dt) {
    if (dt == data_types::f16)
        return dg::logical_tensor::data_type::f16;
    OPENVINO_THROW("[GPU] FusedMLP oneDNN Graph supports only f16");
}

static void extract_dims_from_layouts(const layout& x_layout,
                                      const layout& w_gate_layout,
                                      const layout& w_up_layout,
                                      const layout& w_down_layout,
                                      int64_t& mb,
                                      int64_t& ic,
                                      int64_t& oc) {
    auto x = x_layout.get_tensor().sizes(format::bfyx);          // [b, f, y, x] in bfyx order
    auto w_gate = w_gate_layout.get_tensor().sizes(format::bfyx);
    auto w_up = w_up_layout.get_tensor().sizes(format::bfyx);
    auto w_down = w_down_layout.get_tensor().sizes(format::bfyx);

    OPENVINO_ASSERT(x[3] == 1, "[GPU] fused_mlp: X last dimension must be 1 (bfyx x=1)");
    if (x[2] == 1) {
        mb = static_cast<int64_t>(x[0]);
        ic = static_cast<int64_t>(x[1]);
    } else {
        mb = static_cast<int64_t>(x[0]) * static_cast<int64_t>(x[1]);
        ic = static_cast<int64_t>(x[2]);
    }

    OPENVINO_ASSERT(w_gate[2] == 1 && w_gate[3] == 1, "[GPU] fused_mlp: W_gate must be 2D matrix encoded as [IC, OC, 1, 1]");
    OPENVINO_ASSERT(w_up[2] == 1 && w_up[3] == 1, "[GPU] fused_mlp: W_up must be 2D matrix encoded as [IC, OC, 1, 1]");
    OPENVINO_ASSERT(w_down[2] == 1 && w_down[3] == 1, "[GPU] fused_mlp: W_down must be 2D matrix encoded as [OC, IC, 1, 1]");

    const int64_t w_gate_ic = static_cast<int64_t>(w_gate[0]);
    const int64_t w_gate_oc = static_cast<int64_t>(w_gate[1]);
    const int64_t w_up_ic = static_cast<int64_t>(w_up[0]);
    const int64_t w_up_oc = static_cast<int64_t>(w_up[1]);
    const int64_t w_down_oc = static_cast<int64_t>(w_down[0]);
    const int64_t w_down_ic = static_cast<int64_t>(w_down[1]);

    OPENVINO_ASSERT(w_gate_ic == ic && w_up_ic == ic, "[GPU] fused_mlp: weights IC mismatch");
    OPENVINO_ASSERT(w_gate_oc == w_up_oc, "[GPU] fused_mlp: W_gate/W_up OC mismatch");
    OPENVINO_ASSERT(w_down_oc == w_gate_oc && w_down_ic == ic, "[GPU] fused_mlp: W_down dims mismatch");

    oc = w_gate_oc;
}

}  // namespace

struct fused_mlp_onednn_graph : public typed_primitive_impl<fused_mlp> {
    using parent = typed_primitive_impl<fused_mlp>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::fused_mlp_onednn_graph)

    fused_mlp_onednn_graph() : parent("onednn_graph::fused_mlp") {}

    fused_mlp_onednn_graph(const kernel_impl_params& impl_params, cldnn::engine& engine)
        : parent("onednn_graph::fused_mlp") {
        build(impl_params, engine);
    }

    bool is_cpu() const override { return false; }
    bool is_onednn() const override { return true; }

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<fused_mlp_onednn_graph>(*this);
    }

    void init_kernels(const kernels_cache&, const kernel_impl_params&) override {}

    event::ptr execute_impl(const std::vector<event::ptr>& /*events*/, fused_mlp_inst& instance) override {
        auto& network = instance.get_network();
        auto& stream = network.get_stream();
        auto& engine = network.get_engine();
        auto& dnnl_engine = engine.get_onednn_engine();

        const auto& params = *instance.get_impl_params();
        auto x_layout = params.get_input_layout(0);
        auto w_gate_layout = instance.dep_memory_ptr(1)->get_layout();
        auto w_up_layout = instance.dep_memory_ptr(2)->get_layout();
        auto w_down_layout = instance.dep_memory_ptr(3)->get_layout();

        int64_t mb = 0, ic = 0, oc = 0;
        extract_dims_from_layouts(x_layout, w_gate_layout, w_up_layout, w_down_layout, mb, ic, oc);

        const auto dt = to_graph_dt(x_layout.data_type);
        dg::logical_tensor::dims src_sz = {mb, ic};
        dg::logical_tensor::dims wei0_sz = {ic, oc};
        dg::logical_tensor::dims hd_sz = {mb, oc};
        dg::logical_tensor::dims wei2_sz = {oc, ic};
        dg::logical_tensor::dims out_sz = {mb, ic};

        size_t id = 0;
        auto src = dg::logical_tensor(id++, dt, src_sz, dg::logical_tensor::layout_type::strided);
        auto wei0 = dg::logical_tensor(id++, dt, wei0_sz, dg::logical_tensor::layout_type::strided, dg::logical_tensor::property_type::constant);
        auto wei1 = dg::logical_tensor(id++, dt, wei0_sz, dg::logical_tensor::layout_type::strided, dg::logical_tensor::property_type::constant);
        auto wei2 = dg::logical_tensor(id++, dt, wei2_sz, dg::logical_tensor::layout_type::strided, dg::logical_tensor::property_type::constant);

        auto out0 = dg::logical_tensor(id++, dt, hd_sz, dg::logical_tensor::layout_type::strided);
        auto out1 = dg::logical_tensor(id++, dt, hd_sz, dg::logical_tensor::layout_type::strided);
        auto out2 = dg::logical_tensor(id++, dt, hd_sz, dg::logical_tensor::layout_type::strided);
        auto out3 = dg::logical_tensor(id++, dt, hd_sz, dg::logical_tensor::layout_type::strided);
        auto out4 = dg::logical_tensor(id++, dt, hd_sz, dg::logical_tensor::layout_type::strided);

        auto dst = dg::logical_tensor(id++, dt, out_sz, dg::logical_tensor::layout_type::strided);

        auto fc_gate = dg::op(id++, dg::op::kind::MatMul, "fc_gate");
        fc_gate.add_inputs({src, wei0});
        fc_gate.add_outputs({out0});

        auto fc_up = dg::op(id++, dg::op::kind::MatMul, "fc_up");
        fc_up.add_inputs({src, wei1});
        fc_up.add_outputs({out1});

        auto swi_sig = dg::op(id++, dg::op::kind::Sigmoid, "swish/sigmoid");
        swi_sig.add_inputs({out0});
        swi_sig.add_outputs({out2});

        auto swi_mul = dg::op(id++, dg::op::kind::Multiply, "swish/multiply");
        swi_mul.add_inputs({out0, out2});
        swi_mul.add_outputs({out3});

        auto mul = dg::op(id++, dg::op::kind::Multiply, "mul");
        mul.add_inputs({out3, out1});
        mul.add_outputs({out4});

        auto fc_down = dg::op(id++, dg::op::kind::MatMul, "fc_down");
        fc_down.add_inputs({out4, wei2});
        fc_down.add_outputs({dst});

        dg::graph g(dnnl_engine.get_kind());
        g.add_op(fc_gate);
        g.add_op(fc_up);
        g.add_op(swi_sig);
        g.add_op(swi_mul);
        g.add_op(mul);
        g.add_op(fc_down);
        g.finalize();

        auto partitions = g.get_partitions();
        OPENVINO_ASSERT(partitions.size() == 1, "[GPU] fused_mlp: expected a single oneDNN Graph partition, got ", partitions.size());

        _cp = partitions[0].compile({src, wei0, wei1, wei2}, {dst}, dnnl_engine);

        _src_lt = _cp.query_logical_tensor(src.get_id());
        _w_gate_lt = _cp.query_logical_tensor(wei0.get_id());
        _w_up_lt = _cp.query_logical_tensor(wei1.get_id());
        _w_down_lt = _cp.query_logical_tensor(wei2.get_id());
        _dst_lt = _cp.query_logical_tensor(dst.get_id());

        // Create tensor wrappers for current memory handles.
        std::vector<dg::tensor> inputs;
        inputs.reserve(4);
        inputs.emplace_back(_src_lt, dnnl_engine, instance.dep_memory_ptr(0)->buffer_ptr());
        inputs.emplace_back(_w_gate_lt, dnnl_engine, instance.dep_memory_ptr(1)->buffer_ptr());
        inputs.emplace_back(_w_up_lt, dnnl_engine, instance.dep_memory_ptr(2)->buffer_ptr());
        inputs.emplace_back(_w_down_lt, dnnl_engine, instance.dep_memory_ptr(3)->buffer_ptr());

        std::vector<dg::tensor> outputs;
        outputs.reserve(1);
        outputs.emplace_back(_dst_lt, dnnl_engine, instance.output_memory_ptr()->buffer_ptr());

        try {
            _cp.execute(stream.get_onednn_stream(), inputs, outputs);
        } catch (dnnl::error& err) {
            auto err_code = err.status == dnnl_status_t::dnnl_out_of_memory ? CL_OUT_OF_RESOURCES : CL_INVALID_OPERATION;
            ocl::rethrow(err.what(), err_code, engine.get_device_info());
        }

        if (instance.needs_completion_event())
            return stream.enqueue_marker({});

        return nullptr;
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());
        OPENVINO_ASSERT(impl_params != nullptr, "[GPU] Missing kernel_impl_params for fused_mlp deserialization");
        build(*impl_params, ib.get_engine());
    }

private:
    dg::compiled_partition _cp;
    dg::logical_tensor _src_lt;
    dg::logical_tensor _w_gate_lt;
    dg::logical_tensor _w_up_lt;
    dg::logical_tensor _w_down_lt;
    dg::logical_tensor _dst_lt;

    void build(const kernel_impl_params& impl_params, cldnn::engine& engine) {
        OPENVINO_ASSERT(impl_params.input_layouts.size() == 4, "[GPU] fused_mlp expects 4 inputs");

        const auto& x_layout = impl_params.input_layouts[0];
        const auto& w_gate_layout = impl_params.input_layouts[1];
        const auto& w_up_layout = impl_params.input_layouts[2];
        const auto& w_down_layout = impl_params.input_layouts[3];
        const auto& out_layout = impl_params.output_layouts[0];

        
        // OPENVINO_ASSERT(!x_layout.is_dynamic() && !w_gate_layout.is_dynamic() && !w_up_layout.is_dynamic() && !w_down_layout.is_dynamic() && !out_layout.is_dynamic(),
        //                 "[GPU] fused_mlp supports only static layouts (POC)");

        OPENVINO_ASSERT(x_layout.data_type == data_types::f16 && w_gate_layout.data_type == data_types::f16 && w_up_layout.data_type == data_types::f16 &&
                            w_down_layout.data_type == data_types::f16 && out_layout.data_type == data_types::f16,
                        "[GPU] fused_mlp supports only f16 (POC)");

        // int64_t mb = 0, ic = 0, oc = 0;
        // extract_dims_from_layouts(x_layout, w_gate_layout, w_up_layout, w_down_layout, mb, ic, oc);

        // const auto dt = to_graph_dt(x_layout.data_type);
        // dg::logical_tensor::dims src_sz = {mb, ic};
        // dg::logical_tensor::dims wei0_sz = {ic, oc};
        // dg::logical_tensor::dims hd_sz = {mb, oc};
        // dg::logical_tensor::dims wei2_sz = {oc, ic};
        // dg::logical_tensor::dims out_sz = {mb, ic};

        // size_t id = 0;
        // auto src = dg::logical_tensor(id++, dt, src_sz, dg::logical_tensor::layout_type::strided);
        // auto wei0 = dg::logical_tensor(id++, dt, wei0_sz, dg::logical_tensor::layout_type::strided, dg::logical_tensor::property_type::constant);
        // auto wei1 = dg::logical_tensor(id++, dt, wei0_sz, dg::logical_tensor::layout_type::strided, dg::logical_tensor::property_type::constant);
        // auto wei2 = dg::logical_tensor(id++, dt, wei2_sz, dg::logical_tensor::layout_type::strided, dg::logical_tensor::property_type::constant);

        // auto out0 = dg::logical_tensor(id++, dt, hd_sz, dg::logical_tensor::layout_type::strided);
        // auto out1 = dg::logical_tensor(id++, dt, hd_sz, dg::logical_tensor::layout_type::strided);
        // auto out2 = dg::logical_tensor(id++, dt, hd_sz, dg::logical_tensor::layout_type::strided);
        // auto out3 = dg::logical_tensor(id++, dt, hd_sz, dg::logical_tensor::layout_type::strided);
        // auto out4 = dg::logical_tensor(id++, dt, hd_sz, dg::logical_tensor::layout_type::strided);

        // auto dst = dg::logical_tensor(id++, dt, out_sz, dg::logical_tensor::layout_type::strided);

        // auto fc_gate = dg::op(id++, dg::op::kind::MatMul, "fc_gate");
        // fc_gate.add_inputs({src, wei0});
        // fc_gate.add_outputs({out0});

        // auto fc_up = dg::op(id++, dg::op::kind::MatMul, "fc_up");
        // fc_up.add_inputs({src, wei1});
        // fc_up.add_outputs({out1});

        // auto swi_sig = dg::op(id++, dg::op::kind::Sigmoid, "swish/sigmoid");
        // swi_sig.add_inputs({out0});
        // swi_sig.add_outputs({out2});

        // auto swi_mul = dg::op(id++, dg::op::kind::Multiply, "swish/multiply");
        // swi_mul.add_inputs({out0, out2});
        // swi_mul.add_outputs({out3});

        // auto mul = dg::op(id++, dg::op::kind::Multiply, "mul");
        // mul.add_inputs({out3, out1});
        // mul.add_outputs({out4});

        // auto fc_down = dg::op(id++, dg::op::kind::MatMul, "fc_down");
        // fc_down.add_inputs({out4, wei2});
        // fc_down.add_outputs({dst});

        // auto& dnnl_engine = engine.get_onednn_engine();
        // dg::graph g(dnnl_engine.get_kind());
        // g.add_op(fc_gate);
        // g.add_op(fc_up);
        // g.add_op(swi_sig);
        // g.add_op(swi_mul);
        // g.add_op(mul);
        // g.add_op(fc_down);
        // g.finalize();

        // auto partitions = g.get_partitions();
        // OPENVINO_ASSERT(partitions.size() == 1, "[GPU] fused_mlp: expected a single oneDNN Graph partition, got ", partitions.size());

        // _cp = partitions[0].compile({src, wei0, wei1, wei2}, {dst}, dnnl_engine);

        // _src_lt = _cp.query_logical_tensor(src.get_id());
        // _w_gate_lt = _cp.query_logical_tensor(wei0.get_id());
        // _w_up_lt = _cp.query_logical_tensor(wei1.get_id());
        // _w_down_lt = _cp.query_logical_tensor(wei2.get_id());
        // _dst_lt = _cp.query_logical_tensor(dst.get_id());
    }
};

std::unique_ptr<primitive_impl> FusedMLPImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<fused_mlp>());
    auto& engine = params.prog->get_engine();
    return std::make_unique<fused_mlp_onednn_graph>(params, engine);
}

}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::fused_mlp_onednn_graph)
