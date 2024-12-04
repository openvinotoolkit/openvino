// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "common.hpp"
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "partitioning/partitioning.hpp"
#include "spatial.hpp"
#include "weights_bank.hpp"

namespace intel_npu {
class Plugin;
}

namespace ov {
namespace npuw {
class ICompiledModel : public ov::ICompiledModel {
public:
    static std::shared_ptr<ov::npuw::ICompiledModel> create(const std::shared_ptr<ov::Model>& model,
                                                            const std::shared_ptr<const ov::IPlugin>& plugin,
                                                            const ov::AnyMap& properties);
    ICompiledModel(const std::shared_ptr<ov::Model>& model, const std::shared_ptr<const ov::IPlugin>& plugin);
};

class InferRequest;
class CompiledModel : public ov::npuw::ICompiledModel {
    using DevList = std::vector<std::string>;
    using GetPropertiesMap =
        std::map<std::string, std::tuple<ov::PropertyMutability, std::function<ov::Any(const ::intel_npu::Config&)>>>;

public:
    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  const ov::AnyMap& properties);

    void export_model(std::ostream& model) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

private:
    // FIXME: This class has many friends..
    friend class IBaseInferRequest;
    friend class JustInferRequest;
    friend class UnfoldInferRequest;
    friend class MemAccessSim;
    friend class FuncMemMgr;
    friend class LLMCompiledModel;

    bool compile_for_success(std::size_t id);
    bool compile_for_device(std::size_t id, const std::string& device_to_try);
    ov::SoPtr<ov::ICompiledModel> compile_submodel(const std::shared_ptr<ov::Model>& submodel,
                                                   const std::string& device);

    void dump_on_fail(std::size_t id, const std::string& device_to_stry, const char* extra);

    void report_io() const;

    void serialize(const std::string& path) const;
    void deserialize(const std::string& path);

    // This is used for removing too long output tensor names to fix some compilation issues
    // NB: These two methods has nothing to do with this particular class and should be
    // moved elsewhere
    void remove_long_output_names(const std::shared_ptr<ov::Model>& model);
    void fill_empty_tensor_names(const std::shared_ptr<ov::Model>& model);

    std::shared_ptr<const ::intel_npu::Plugin> get_npuw_plugin() const;
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    std::string submodel_device(const std::size_t idx) const;
    bool is_gather_closure(const std::size_t idx, const std::size_t cidx) const;
    bool unpack_required(const std::size_t idx) const;
    bool unpack_required(const std::size_t idx, const std::size_t cidx) const;

    void log_device_dist() const;
    void implement_properties();

    void finalize_weights_bank();
    void detach_memory();
    std::string global_mem_device() const;
    std::string funcall_mem_device(const std::size_t idx) const;

    std::shared_ptr<::intel_npu::OptionsDesc> m_options_desc; // no ser
    ::intel_npu::Config m_cfg; // no ser ?
    GetPropertiesMap m_prop_to_opt; // no ser ?
    std::vector<ov::PropertyName> m_all_supported_props; // no ser
    ov::AnyMap m_non_npuw_props; // no ser ?

    std::string m_name; // yes ser
    const bool m_loaded_from_cache; // no ser

    using ToSubmodel = std::pair<size_t /* submodel_idx */
                                 ,
                                 size_t /* port_idx     */
                                 >;
    static const constexpr auto NO_LINK = ToSubmodel{-1, -1};

    // In the below vector, index == compiled model's input/output port idex.
    std::vector<ToSubmodel> m_inputs_to_submodels_inputs; // yes ser
    std::vector<ToSubmodel> m_outputs_to_submodels_outputs; // yes ser

    std::map<std::size_t, std::vector<ToSubmodel>> m_param_subscribers; // yes ser

    std::map<std::pair<size_t /*submodel_idx*/, size_t /*node_idx*/>,  // input ("to")
             std::pair<size_t /*submodel_idx*/, size_t /*node_idx*/>>  // output ("from")
        m_submodels_input_to_prev_output; // yes ser

    DeviceProperties m_meta_devices; // no ser

    DevList m_dev_list; // no ser

    struct execution_stats {
        float gflops{};
        std::size_t ops{};
    };

    struct CompiledModelDesc {
        DevList::const_iterator device_it; // yes ser
        std::set<std::string> devices_to_avoid; // no ser
        std::shared_ptr<ov::Model> model; // no ser
        ov::SoPtr<ov::ICompiledModel> compiled_model; // yes ser

        std::optional<std::size_t> replaced_by; // yes ser

        Subgraph::Gather host_gather; // yes ser
        std::optional<ov::npuw::compiled::Spatial> spatial; // yes ser

        // FIXME: This is a 1:1 copy of the ov::npuw::Subgraph structure
        // w.r.t. function calls
        std::size_t param_base = 0; // yes ser
        // NB: closure and lazy_closure are of the same size - to preserve proper indexing.
        //     closure is responsible for host-side tensors (DCOFF, Gather, etc) while
        //     lazy_closure is used for weights sharing and allocating device memory.
        std::vector<ov::Tensor> closure; // yes ser
        std::vector<weights::LazyTensor> lazy_closure; // yes ser, weightless
        std::vector<ov::Tensor> scales; // yes ser
        std::vector<ov::Tensor> zerops; // yes ser
        std::vector<bool> is_remote; // yes ser

        bool forced_to_fcall = false; // yes ser

        // FIXME: Take it out of structure
        ov::SoPtr<ov::ICompiledModel> ref_compiled_model; // no ser
        bool switched_to_ref = false; // no ser

        // Metrics
        execution_stats stat; // no ser
    };
    std::vector<CompiledModelDesc> m_compiled_submodels; // yes ser

    std::function<bool(const ov::SoPtr<ov::ITensor>&, const ov::SoPtr<ov::ITensor>&)> m_acc_check; // no ser
    std::string m_ref_device; // no ser

    execution_stats m_total_stat; // no ser

    std::shared_ptr<weights::Bank> m_weights_bank = nullptr; // no ser - inderectly instead
};
}  // namespace npuw
}  // namespace ov
