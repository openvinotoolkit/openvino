#include <string>
#include <iostream>

extern "C" __declspec(dllexport) void core_get_property_test(std::string target_device);
extern "C" __declspec(dllexport) void core_infer_test(std::string target_device);
