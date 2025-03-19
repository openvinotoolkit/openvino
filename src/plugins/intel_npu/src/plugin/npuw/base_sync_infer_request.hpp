// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "perf.hpp"
#include "spatial.hpp"

namespace ov {
namespace npuw {

using TensorPtr = ov::SoPtr<ov::ITensor>;

class CompiledModel;

using LinkFrom = std::pair<std::size_t /* Subrequest index */
                           ,
                           std::size_t /* Subrequest output index */
                           >;          // FIXME: This is a third, if not fourth, definitiion of such structure

// This interface is provided to npuw::AsyncInferRequest to manage the
// individual subrequests' execution
class IBaseInferRequest : public ov::ISyncInferRequest {
public:
    explicit IBaseInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>&);

    // Execution API - explicitly "finalize" the infer() here
    void infer() override;  // final - not final yet

    // I/O APIs - supply default implementations
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;

    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const override;
    void set_tensors(const ov::Output<const ov::Node>& port,
                     const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

    void check_tensors() const override;

    // Query APIs - some default implementations here
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    using sptr = std::shared_ptr<IBaseInferRequest>;
    using Completed = std::function<void(std::exception_ptr)>;

    virtual void prepare_for_infer() = 0;
    virtual bool valid_subrequest(std::size_t idx) const = 0;  // FIXME: Get rid of this!
    virtual void start_subrequest(std::size_t idx) = 0;
    virtual void subscribe_subrequest(std::size_t idx, Completed cb) = 0;
    virtual void run_subrequest_for_success(std::size_t idx, bool& failover) = 0;
    virtual void complete_subrequest(std::size_t idx) = 0;
    virtual void cancel_subrequest(std::size_t idx) = 0;
    virtual std::size_t total_subrequests() const;
    virtual bool supports_async_pipeline() const = 0;

protected:
    using RqPtr = ov::SoPtr<ov::IAsyncInferRequest>;
    using RqPtrs = std::vector<RqPtr>;

    // This method can only be called for regular subgraphs or
    // function bodies. Function calls are not allowed to have
    // their inference requests anymore - they must be stored
    // only once in the subrequests list
    RqPtrs create_infer_requests(std::size_t id, size_t nireq = 1, bool* recompiled = nullptr);
    void ensure_subrequest_is_accurate(std::size_t idx, bool& failover);
    virtual void update_subrequest_links(std::size_t idx) = 0;

    std::shared_ptr<ov::npuw::CompiledModel> m_npuw_model;
    std::vector<IBaseInferRequest::Completed> m_completion_cbs;
    RqPtrs m_subrequests;

    // This vector is used to track devices for individual subrequests
    // here locally. Note that the models can be recompiled in
    // contexts of other requests (if multiple of those are created)
    // so this cached information is used to detect these situations.
    std::vector<std::string> m_subrequest_devices;

    // Permanent storage for input & output tensors
    // FIXME: Currently is initialized in subclasses. Likely this
    // initialization should be moved here, to the base class?
    std::vector<ov::SoPtr<ov::ITensor>> m_input_tensors;
    std::vector<ov::SoPtr<ov::ITensor>> m_output_tensors;

    struct TensorStorage {
        ov::SoPtr<ov::ITensor> tensor;
        bool persistent = false;       // true for the parent I/O tensors
        std::size_t num_readers = 0u;  // fixed during execution
        std::size_t num_reads = 0u;    // changes during execution (ref-counter-like).
                                       // reset to 0 before every new execution
    };
    // FROM(Every subrequests' output port) TO(Its output tensor)
    std::map<ov::Output<const ov::Node>, TensorStorage> m_port_to_tensor;

    // FIXME: Currently is initialized/managed by subclass as well.
    // Moved here dumping purposes only
    // Another sparse vector. Represents populated spatial I/O parameters
    // which can should be read/written by parts in multile submissions.
    // An ugly structure, cries for refactoring
    // See function_prologue for details.
    // Also it contains pre-allocated tensors for tails handling
    struct SpatialIO {
        std::vector<ov::SoPtr<ov::ITensor>> inputs;   // # of elements - # of graph-side inputs
        std::vector<ov::SoPtr<ov::ITensor>> outputs;  // # of elements - # of subgraph outputs

        std::vector<ov::SoPtr<ov::ITensor>> input_tails;   // temporary buffers for input tails
        std::vector<ov::SoPtr<ov::ITensor>> output_tails;  // temporary buffers for output tails
    };
    std::vector<SpatialIO> m_spatial_io;

    // FIXME: Currently is initialized/managed by subclass as well.
    // Moved here dumping purposes only
    // Represents spatial run-time info
    runtime::spatial::Selector::Ptr m_spatial_selector;

    // This structure tracks how every individual subrequest
    // access the model's top-level (global, public, etc) parameters
    // and results. Again, is managed by subclasses
    struct GlobalIO {
        using map_t = std::map<std::size_t, std::size_t>;
        map_t global_params;   // param idx -> input idx
        map_t global_results;  // result idx -> output idx
    };
    std::vector<GlobalIO> m_subrequests_gio;

    // Tracks tensors we allocated on our own - to recognize and avoid copies
    std::unordered_set<void*> m_input_allocated;

    // Common functionality - shared for subclasses
    const std::size_t m_num_submodels;

    TensorPtr allocMem(const ov::element::Type type, const ov::Shape& shape, const std::string& device);
    TensorPtr allocOut(const ov::Output<const ov::Node>& node, const std::string& device);
    virtual void alloc_io();
    virtual TensorPtr alloc_global_out(std::size_t out_idx);

    virtual void init_gio();
    void unpack_closure(std::size_t idx, RqPtr request);
    virtual void bind_global_params(std::size_t idx, RqPtr request);
    virtual void bind_global_results(std::size_t idx, RqPtr request);

    void dump_input_tensors(std::size_t idx);
    void dump_output_tensors(std::size_t idx);

    // Quick-and-dirty profiling
    ov::npuw::perf::metric<float, ov::npuw::perf::MSec> m_ms_unpack;

    // Various name/dump formatting methods
    // TODO: These methods should probably go to CompiledModel
    std::string subgr_name(std::size_t idx) const;
    std::string subgr_path_suffix(std::size_t idx) const;

    // TODO: And this should probably go to some TensorDumper, etc
    // if we go over-designing the things.
    std::string iter_path_suffix(std::size_t idx) const;
    mutable std::optional<bool> m_iter_suffix_required;
    std::size_t m_run_iter = 0u;

    bool needs_copy(std::size_t idx) const;
    bool needs_copy(std::size_t idx, std::size_t cidx) const;
    std::size_t next(std::size_t idx_base) const;
    std::size_t real(std::size_t idx) const;

    RqPtrs m_ref_subrequests;

    using now_t = std::optional<std::size_t>;
    now_t now_idx() const;

private:
    now_t m_now_idx;
};

}  // namespace npuw
}  // namespace ov
