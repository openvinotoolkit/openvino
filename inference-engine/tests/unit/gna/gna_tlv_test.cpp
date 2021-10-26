// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

#include "gna2-tlv-reader.h"

struct LD {
    uint8_t type;
    uint8_t pad[0x1F];
    uint32_t in;
    uint32_t out;
    uint32_t sum;
    uint32_t fb;
    uint32_t weight;
    uint32_t bias;
    uint32_t actlist;
    uint32_t activation;
    uint8_t pad2[0x40];
    static std::string barName(uint32_t offset) {
        offset &= 0x3;
        if (offset == 0) {
            return "GNADSCBAR";
        } else if (offset == 1) {
            return "MBAR0";
        } else if (offset == 2) {
            return "MBAR1";
        }
        return "MBAR2";
    }
    static std::string offset(uint32_t offset) {
        auto barN = barName(offset);
        offset -= offset & 0x3;
        std::ostringstream s;
        s << barN << "[0x" << std::hex << offset << ", "<< std::dec << offset << "]";
        return s.str();
    }
    std::string GetNnopName() {
        static const std::map<uint8_t, std::string> nnopMap = {
            {0x00, "[AFFINE] Standard fully connected layer"                            },
            {0x01, "[AFFINE_AL] Affine layer with pruned outputs (Active-List)"         },
            {0x02, "[DIAGONAL] Affine layer with a diagonal weights matrix"             },
            {0x03, "[AFFINE_TH] Affine layer with Threshold"                            },
            {0x04, "[RNN] Recurrent layer (Non-Interleaved)"                            },
            {0x08, "[1DCNN] 1D Convolutional Layer {Deprecated}"                        },
            {0x09, "[AFFINE_MBG] Affine with Multi-BIAS Interleaved array"              },
            {0x10, "[DE_INTERLEAVE] Transforms Interleaved to Non-Interleaved"          },
            {0x11, "[INTERLEAVE] Transforms Non-Interleaved to Interleaved"             },
            {0x12, "[COPY] Manipulation layer, copy operation for 16-bits arrays"       },
            {0x20, "[GMM] GMM Layer {Deprecated}"                                       },
            {0x21, "[GMM_AL] GMM Layer with Active List {Deprecated}"                   },
            {0x30, "[2DCNNc] 2D Convolutional Fused Layer"                              },
        };
        auto f = nnopMap.find(type);
        if (f != nnopMap.end()) {
            return f->second;
        }
        return "[UNKNOWN_NNOP] (NNOP=" + std::to_string(type) + ")";
    }
    void dumpLayer() {
        std::cout << "type:  " << GetNnopName() << "\n";
        std::cout << "in:    " << offset(in) << "\n";
        std::cout << "out:   " << offset(out) << "\n";
        if (type == 0x12) return;
        std::cout << "sum:   " << offset(sum) << "\n";
        if (type == 0x04 || type == 0x30)
            std::cout << "fb/ada:" << offset(fb) << "\n";
        std::cout << "w/k:   " << offset(weight) << "\n";
        std::cout << "bias:  " << offset(bias) << "\n";
        if (activation == 0) return;
        std::cout << "pwl:   " << offset(activation) << "\n";
    }
};
static_assert(sizeof(LD) == 128, "sizeof(LD) != 128");

class GNATlvTest : public ::testing::Test {
protected:
    void SetUp() override {
    }
};

TEST_F(GNATlvTest, DISABLED_tlvLDParse) {
    std::ifstream tlvFile("C:\\WD\\OV\\211019_TLV\\wov_zrzuty_ww43.3\\wov_en811c_d.fixed.trimmed_ww43.3_qb8.tlv", std::ios::binary | std::ios::ate);
    std::streamsize size = tlvFile.tellg();
    tlvFile.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (tlvFile.read(buffer.data(), size)) {
        uint32_t len;
        void* ptr = nullptr;
        LD* ld = nullptr;
        Gna2TlvFindInArray(buffer.data(), size, Gna2TlvTypeLayerNumber, &len, &ptr);
        const auto numOfLayers = *static_cast<uint32_t*>(ptr);
        Gna2TlvFindInArray(buffer.data(), size, Gna2TlvTypeLayerDescriptorAndRoArrayData, &len, reinterpret_cast<void**>(&ld));
        for (int i = 0; i < numOfLayers; i++, ld++) {
            std::cout << "Layer " << i << "\n";
            ld->dumpLayer();
        }
    }
}
