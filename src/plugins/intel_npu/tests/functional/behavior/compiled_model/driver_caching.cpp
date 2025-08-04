#include "internal/overload/compiled_model/driver_caching.hpp"
#include "common/npu_test_env_cfg.hpp"
#include "common/utils.hpp"
namespace {

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> Config = {
                                        {}
                                        // {ov::cache_dir("deadbeef"), ov::log::level(ov::log::Level::DEBUG)},
                                        // {ov::intel_npu::bypass_umd_caching(false)},
                                        // {ov::intel_npu::bypass_umd_caching(true)}
                                    };

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         CompileAndDriverCaching,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(Config)),
                         ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

}  // namespace
