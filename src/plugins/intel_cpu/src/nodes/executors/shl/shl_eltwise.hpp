#include "shl.hpp"
#include "cpu_memory.h"
#include "nodes/executors/eltwise.hpp"
#include <functional>
#include <tuple>
#include <utility>
#include <type_traits>

namespace ov {
namespace intel_cpu {

class ShlEltwiseExecutor : public EltwiseExecutor {
public:
    explicit ShlEltwiseExecutor(const ExecutorContext::CPtr context);
    static bool isEltwiseAlgorithmSupported(Algorithm algorithm);

    bool init(const EltwiseAttrs& eltwiseAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const std::vector<EltwisePostOp>& postOps) override;

    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              const void *post_ops_data_) override;

    impl_desc_type getImplType() const override {
        return impl_desc_type::shl;
    }

private:
    EltwiseAttrs shlEltwiseAttrs{};
    ShlSession sess = {};
    std::vector<ShlTensor> srcTensors, dstTensors;
    std::unique_ptr<IShlParams> params;
    std::function<int()> exec_func;

    template<typename ExecFunc, typename... Args>
    void setExecFunc(ExecFunc&& execFunc, Args&&... args) {
        exec_func = [this, execFunc, args...]()->int { return callFunc(execFunc, std::make_tuple(args...)); };
    }

    template<typename Func, typename Tuple, size_t... Index>
    int callFunc(Func&& func, Tuple&& tuple, index_sequence<Index...>) const {
        return func(std::get<Index>(std::forward<Tuple>(tuple))...);
    }

    template<typename Func, typename Tuple>
    int callFunc(Func&& func, Tuple&& tuple) const {
        return callFunc(std::forward<Func>(func), std::forward<Tuple>(tuple),
                        make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{});
    }
};

class ShlEltwiseExecutorBuilder : public EltwiseExecutorBuilder {
public:
    bool isSupported(const EltwiseAttrs& eltwiseAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override;

    EltwiseExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ShlEltwiseExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov