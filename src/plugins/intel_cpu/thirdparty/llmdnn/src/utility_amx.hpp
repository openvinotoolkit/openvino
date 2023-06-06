// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdio.h>
#include <stdint.h>
#include <initializer_list>
#include <vector>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

struct tileconfig_t {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
    tileconfig_t() = default;

    tileconfig_t(int palette, int _startRow, const std::initializer_list<std::pair<int, int>> &_rows_columnsBytes) {
        palette_id = palette;
        startRow = _startRow;
        int i;
        for(i = 0; i < 14; i++) {
            reserved[i] = 0;
        }
        i = 0;
        for (const auto& ele : _rows_columnsBytes) {
            rows[i] = ele.first;
            cols[i] = ele.second;
            i++;
        }
        for(; i < 16; i++) {
            cols[i] = 0;
            rows[i] = 0;
        }
        load();
    }

    tileconfig_t(int palette, int _startRow, const std::initializer_list<int> &_rows, int columnsBytes) {
        palette_id = palette;
        startRow = _startRow;
        int i;
        for(i = 0; i < 14; i++) {
            reserved[i] = 0;
        }
        i = 0;
        for (const auto ele : _rows) {
            rows[i] = ele;
            cols[i] = columnsBytes;
            i++;
        }
        for(; i < 16; i++) {
            cols[i] = 0;
            rows[i] = 0;
        }
        load();
    }
    tileconfig_t(int palette, int _startRow, int numTiles, int _rows, int columnsBytes) {
        palette_id = palette;
        startRow = _startRow;
        int i;
        for(i = 0; i < 14; i++) {
            reserved[i] = 0;
        }
        for(i = 0; i < numTiles; i++) {
            rows[i] = _rows;
            cols[i] = columnsBytes;
        }
        for(; i < 16; i++) {
            cols[i] = 0;
            rows[i] = 0;
        }
        load();
    }

    ~tileconfig_t() {
        _tile_release();
    }
    void __attribute__((noinline)) load() {
        //std::cout << "\ttile load config ... " << std::flush;
        _tile_loadconfig(this);
        //std::cout << *this << std::flush << std::endl;
    }
    void store() {
        _tile_storeconfig(this);
    }
    friend std::ostream& operator<<(std::ostream& out, const tileconfig_t& cfg) {
        out << " palette_id=" << static_cast<int>(cfg.palette_id);
        out << " startRow=" << static_cast<int>(cfg.startRow);
        out << " row x colsb=(";
        for (int i = 0; i < 16;i++) {
            if (cfg.rows[i] == 0 && cfg.cols[i] == 0)
                continue;
            if (i > 0) out << ",";
            out << static_cast<int>(cfg.rows[i]) << "x" << static_cast<int>(cfg.cols[i]);
        }
        out << ")";
        return out;
    }
} __attribute__ ((__packed__));
