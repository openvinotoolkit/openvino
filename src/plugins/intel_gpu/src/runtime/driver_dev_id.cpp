// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "driver_dev_id.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "openvino/util/log.hpp"

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
// The include order is significant: devguid.h needs DEFINE_GUID from windows.h, and initguid.h makes
// devguid.h define the GUID_DEVCLASS_* constants in this translation unit.
// clang-format off
#    include <windows.h>
#    include <initguid.h>
#    include <devguid.h>
#    include <setupapi.h>
// clang-format on
#endif

namespace cldnn {

namespace {

constexpr int intel_vendor_id = 0x8086;

bool all_equal(const std::vector<int>& ids) {
    return std::all_of(ids.begin(), ids.end(), [&ids](int id) {
        return id == ids.front();
    });
}

void remove_unused_ids(std::vector<int>& ids) {
    const std::vector<int> unused_ids = {0x4905, 0x4906, 0x4907, 0x4908};
    ids.erase(std::remove_if(ids.begin(),
                             ids.end(),
                             [&unused_ids](int id) {
                                 return std::find(unused_ids.begin(), unused_ids.end(), id) != unused_ids.end();
                             }),
              ids.end());
}

#ifdef _WIN32
// Collects Intel device IDs of all present adapters into `all` and of those located at `pci` into `matched`.
// Display adapters are enumerated last, so `all.back()` is the same ID as before MCDM adapters were supported.
void collect_ids(const std::optional<pci_bus_info>& pci, std::vector<int>& all, std::vector<int>& matched) {
    // SetupAPI exposes bus, device and function only, so a non-zero PCI domain cannot be matched
    const bool match_location = pci.has_value() && pci->pci_domain == 0;
    const GUID* device_classes[] = {&GUID_DEVCLASS_COMPUTEACCELERATOR, &GUID_DEVCLASS_DISPLAY};

    for (const GUID* device_class : device_classes) {
        HDEVINFO device_info_set = SetupDiGetClassDevsA(device_class, NULL, NULL, DIGCF_PRESENT);
        if (device_info_set == INVALID_HANDLE_VALUE) {
            continue;
        }

        SP_DEVINFO_DATA devinfo_data;
        std::memset(&devinfo_data, 0, sizeof(devinfo_data));
        devinfo_data.cbSize = sizeof(devinfo_data);

        for (DWORD dev_idx = 0; SetupDiEnumDeviceInfo(device_info_set, dev_idx, &devinfo_data); dev_idx++) {
            const size_t kBufSize = 512;
            char buf[kBufSize];
            if (!SetupDiGetDeviceInstanceIdA(device_info_set, &devinfo_data, buf, kBufSize, NULL)) {
                continue;
            }

            char* vendor_pos = std::strstr(buf, "VEN_");
            if (vendor_pos == NULL || std::stoi(vendor_pos + 4, NULL, 16) != intel_vendor_id) {
                continue;
            }
            char* device_pos = std::strstr(vendor_pos, "DEV_");
            if (device_pos == NULL) {
                continue;
            }
            const int device_id = std::stoi(device_pos + 4, NULL, 16);
            all.push_back(device_id);

            if (match_location) {
                DWORD bus = 0;
                DWORD address = 0;
                DWORD property_type = 0;
                if (SetupDiGetDeviceRegistryPropertyA(device_info_set,
                                                      &devinfo_data,
                                                      SPDRP_BUSNUMBER,
                                                      &property_type,
                                                      reinterpret_cast<BYTE*>(&bus),
                                                      sizeof(bus),
                                                      nullptr) &&
                    property_type == REG_DWORD &&
                    SetupDiGetDeviceRegistryPropertyA(device_info_set,
                                                      &devinfo_data,
                                                      SPDRP_ADDRESS,
                                                      &property_type,
                                                      reinterpret_cast<BYTE*>(&address),
                                                      sizeof(address),
                                                      nullptr) &&
                    property_type == REG_DWORD && bus == pci->pci_bus && HIWORD(address) == pci->pci_device &&
                    LOWORD(address) == pci->pci_function) {
                    matched.push_back(device_id);
                }
            }
        }

        SetupDiDestroyDeviceInfoList(device_info_set);
    }
}
#elif defined(__linux__)
bool read_intel_device_id(const std::string& dev_base, int& device_id) {
    std::ifstream vendor_ifs(dev_base + "vendor");
    int ven_id = 0;
    if (!(vendor_ifs >> std::hex >> ven_id) || ven_id != intel_vendor_id) {
        return false;
    }
    std::ifstream device_ifs(dev_base + "device");
    return static_cast<bool>(device_ifs >> std::hex >> device_id);
}

// Collects the ID of the device located at `pci` into `matched` and the ID of the integrated GPU at the
// legacy fixed location into `all`.
void collect_ids(const std::optional<pci_bus_info>& pci, std::vector<int>& all, std::vector<int>& matched) {
    int device_id = 0;
    if (pci.has_value()) {
        char location[64];
        std::snprintf(location,
                      sizeof(location),
                      "/sys/bus/pci/devices/%04x:%02x:%02x.%x/",
                      pci->pci_domain,
                      pci->pci_bus,
                      pci->pci_device,
                      pci->pci_function);
        if (read_intel_device_id(location, device_id)) {
            matched.push_back(device_id);
        }
    }
    if (read_intel_device_id("/sys/devices/pci0000:00/0000:00:02.0/", device_id)) {
        all.push_back(device_id);
    }
}
#endif

}  // namespace

int driver_dev_id(const std::optional<pci_bus_info>& pci) {
    std::vector<int> all;
    std::vector<int> matched;
#if defined(_WIN32) || defined(__linux__)
    collect_ids(pci, all, matched);
#endif
    remove_unused_ids(all);
    remove_unused_ids(matched);

    if (!matched.empty() && all_equal(matched)) {
        return matched.front();
    }
    if (all.empty()) {
        return 0;
    }
    if (all_equal(all)) {
        return all.front();
    }

    char last_id[16];
    std::snprintf(last_id, sizeof(last_id), "0x%X", all.back());
    OPENVINO_WARN("[GPU] Unable to match the device to a single Intel PCI device ID among ",
                  all.size(),
                  " adapters; using the last enumerated one: ",
                  last_id);
    return all.back();
}

}  // namespace cldnn
