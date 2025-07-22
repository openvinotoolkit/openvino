#include "overload/driver_caching.hpp"
#include "common/npu_test_env_cfg.hpp"
//#include "iostream"

namespace {

using namespace ov::test::behavior;

const std::vector<ov::AnyMap> Config = {{}};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         CompileAndDriverCaching,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(Config)),
                         ov::test::utils::appendPlatformTypeTestName<CompileAndDriverCaching>);

}  // namespace
