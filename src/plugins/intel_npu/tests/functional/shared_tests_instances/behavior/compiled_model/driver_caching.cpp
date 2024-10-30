#include "overload/driver_caching.hpp"
#include "common/npu_test_env_cfg.hpp"
//#include "iostream"

namespace {

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> configEmpty = {{}};
const std::vector<ov::AnyMap> configOvCache = {{ov::cache_dir.name(), "./testCacheDir"}};
const std::vector<ov::AnyMap> configBypass = {{ov::intel_npu::bypass_umd_caching.name(), true}};

INSTANTIATE_TEST_SUITE_P(smoke_DriverCaching_BehaviorTests,
                         CompileAndDriverCaching,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configEmpty)),
                         ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);



INSTANTIATE_TEST_SUITE_P(smoke_DriverCaching_BehaviorTests,
                         CompileAndDriverCaching,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configOvCache)),
                         ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);


INSTANTIATE_TEST_SUITE_P(smoke_DriverCaching_BehaviorTests,
                         CompileAndDriverCaching,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configBypass)),
                        ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);
}  // namespace


///这边应该传三个，分别是空，cachedir 和udmcache