#include "../common/tests_utils.h"

#include <pugixml.hpp>

class MemCheckEnvironment {
private:
    pugi::xml_document _refs_config;
    MemCheckEnvironment() = default;
    MemCheckEnvironment(const MemCheckEnvironment&) = delete;
    MemCheckEnvironment& operator=(const MemCheckEnvironment&) = delete;
public:
    static MemCheckEnvironment& Instance(){
        static MemCheckEnvironment env;
        return env;
    }

    const pugi::xml_document & getRefsConfig() {
        return _refs_config;
    }

    void setRefsConfig(const pugi::xml_document &refs_config) {
        _refs_config.reset(refs_config);
    }
};

class TestReferences {
private:
    std::vector<std::string> model_path_v, test_name_v, device_v;
    std::vector<long> vmsize_v, vmpeak_v, vmrss_v, vmhwm_v;
public:
    long ref_vmsize = -1, ref_vmpeak = -1, ref_vmrss = -1, ref_vmhwm = -1;

    TestReferences () {
        // Parse RefsConfig from MemCheckEnvironment
        std::string models_path = Environment::Instance().getEnvConfig()
                .child("attributes").child("irs_path").child("value").text().as_string();

        const pugi::xml_document &refs_config = MemCheckEnvironment::Instance().getRefsConfig();
        auto values = refs_config.child("attributes").child("models");
        for (pugi::xml_node node = values.first_child(); node; node = node.next_sibling()) {
            for (pugi::xml_attribute_iterator ait = node.attributes_begin(); ait != node.attributes_end(); ait++) {
                if (strncmp(ait->name(), "path", strlen(ait->name())) == 0) {
                    model_path_v.push_back(OS_PATH_JOIN({models_path, ait->value()}));
                } else if (strncmp(ait->name(), "test", strlen(ait->name())) == 0) {
                    test_name_v.push_back(ait->value());
                } else if (strncmp(ait->name(), "device", strlen(ait->name())) == 0) {
                    device_v.push_back(ait->value());
                } else if (strncmp(ait->name(), "vmsize", strlen(ait->name())) == 0) {
                    vmsize_v.push_back(std::atoi(ait->value()));
                } else if (strncmp(ait->name(), "vmpeak", strlen(ait->name())) == 0) {
                    vmpeak_v.push_back(std::atoi(ait->value()));
                } else if (strncmp(ait->name(), "vmrss", strlen(ait->name())) == 0) {
                    vmrss_v.push_back(std::atoi(ait->value()));
                } else if (strncmp(ait->name(), "vmhwm", strlen(ait->name())) == 0) {
                    vmhwm_v.push_back(std::atoi(ait->value()));
                }
            }
        }
    }

    void collect_vm_values_for_test(std::string test_name, TestCase test_params) {
        for (int i = 0; i < test_name_v.size(); i++)
            if (test_name_v[i] == test_name)
                if (model_path_v[i] == test_params.model)
                    if (device_v[i] == test_params.device) {
                        ref_vmsize = vmsize_v[i];
                        ref_vmpeak = vmpeak_v[i];
                        ref_vmrss = vmrss_v[i];
                        ref_vmhwm = vmhwm_v[i];
                    }
    }
};