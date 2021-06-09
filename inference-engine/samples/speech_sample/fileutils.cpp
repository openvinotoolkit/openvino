// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fileutils.hpp"

void ArkFile::GetFileInfo(const char* fileName, uint32_t numArrayToFindSize, uint32_t* ptrNumArrays, uint32_t* ptrNumMemoryBytes) {
    uint32_t numArrays = 0;
    uint32_t numMemoryBytes = 0;

    std::ifstream in_file(fileName, std::ios::binary);
    if (in_file.good()) {
        while (!in_file.eof()) {
            std::string line;
            uint32_t numRows = 0u, numCols = 0u, num_bytes = 0u;
            std::getline(in_file, line, '\0');  // read variable length name followed by space and NUL
            std::getline(in_file, line, '\4');  // read "BFM" followed by space and control-D
            if (line.compare("BFM ") != 0) {
                break;
            }
            in_file.read(reinterpret_cast<char*>(&numRows), sizeof(uint32_t));  // read number of rows
            std::getline(in_file, line, '\4');                                  // read control-D
            in_file.read(reinterpret_cast<char*>(&numCols), sizeof(uint32_t));  // read number of columns
            num_bytes = numRows * numCols * sizeof(float);
            in_file.seekg(num_bytes, in_file.cur);  // read data

            if (numArrays == numArrayToFindSize) {
                numMemoryBytes += num_bytes;
            }
            numArrays++;
        }
        in_file.close();
    } else {
        throw std::runtime_error(std::string("Failed to open %s for reading in GetFileInfo()!\n") + fileName);
    }

    if (ptrNumArrays != NULL)
        *ptrNumArrays = numArrays;
    if (ptrNumMemoryBytes != NULL)
        *ptrNumMemoryBytes = numMemoryBytes;
}

void ArkFile::LoadFile(const char* fileName, uint32_t arrayIndex, std::string& ptrName, std::vector<uint8_t>& memory, uint32_t* ptrNumRows,
                       uint32_t* ptrNumColumns, uint32_t* ptrNumBytesPerElement) {
    std::ifstream in_file(fileName, std::ios::binary);
    if (in_file.good()) {
        uint32_t i = 0;
        while (i < arrayIndex) {
            std::string line;
            uint32_t numRows = 0u, numCols = 0u;
            std::getline(in_file, line, '\0');  // read variable length name followed by space and NUL
            std::getline(in_file, line, '\4');  // read "BFM" followed by space and control-D
            if (line.compare("BFM ") != 0) {
                break;
            }
            in_file.read(reinterpret_cast<char*>(&numRows), sizeof(uint32_t));  // read number of rows
            std::getline(in_file, line, '\4');                                  // read control-D
            in_file.read(reinterpret_cast<char*>(&numCols), sizeof(uint32_t));  // read number of columns
            in_file.seekg(numRows * numCols * sizeof(float), in_file.cur);      // read data
            i++;
        }
        if (!in_file.eof()) {
            std::string line;
            std::getline(in_file, ptrName, '\0');  // read variable length name followed by space and NUL
            std::getline(in_file, line, '\4');     // read "BFM" followed by space and control-D
            if (line.compare("BFM ") != 0) {
                throw std::runtime_error(std::string("Cannot find array specifier in file %s in LoadFile()!\n") + fileName);
            }
            in_file.read(reinterpret_cast<char*>(ptrNumRows), sizeof(uint32_t));     // read number of rows
            std::getline(in_file, line, '\4');                                       // read control-D
            in_file.read(reinterpret_cast<char*>(ptrNumColumns), sizeof(uint32_t));  // read number of columns
            in_file.read(reinterpret_cast<char*>(&memory.front()),
                         *ptrNumRows * *ptrNumColumns * sizeof(float));  // read array data
        }
        in_file.close();
    } else {
        throw std::runtime_error(std::string("Failed to open %s for reading in LoadFile()!\n") + fileName);
    }

    *ptrNumBytesPerElement = sizeof(float);
}

void ArkFile::SaveFile(const char* fileName, bool shouldAppend, std::string name, void* ptrMemory, uint32_t numRows, uint32_t numColumns) {
    std::ios_base::openmode mode = std::ios::binary;
    if (shouldAppend) {
        mode |= std::ios::app;
    }
    std::ofstream out_file(fileName, mode);
    if (out_file.good()) {
        out_file.write(name.c_str(), name.length());  // write name
        out_file.write("\0", 1);
        out_file.write("BFM ", 4);
        out_file.write("\4", 1);
        out_file.write(reinterpret_cast<char*>(&numRows), sizeof(uint32_t));
        out_file.write("\4", 1);
        out_file.write(reinterpret_cast<char*>(&numColumns), sizeof(uint32_t));
        out_file.write(reinterpret_cast<char*>(ptrMemory), numRows * numColumns * sizeof(float));
        out_file.close();
    } else {
        throw std::runtime_error(std::string("Failed to open %s for writing in SaveFile()!\n") + fileName);
    }
}

void NumpyFile::GetFileInfo(const char* fileName, uint32_t numArrayToFindSize, uint32_t* ptrNumArrays, uint32_t* ptrNumMemoryBytes) {
    uint32_t numArrays = 0;
    uint32_t numMemoryBytes = 0;

    cnpy::npz_t my_npz1 = cnpy::npz_load(fileName);
    auto it = my_npz1.begin();
    std::advance(it, numArrayToFindSize);
    if (it != my_npz1.end()) {
        numArrays = my_npz1.size();
        cnpy::NpyArray my_npy = it->second;
        numMemoryBytes = my_npy.data_holder->size();

        if (ptrNumArrays != NULL)
            *ptrNumArrays = numArrays;
        if (ptrNumMemoryBytes != NULL)
            *ptrNumMemoryBytes = numMemoryBytes;
    } else {
        throw std::runtime_error(std::string("Failed to get info %s  GetFileInfo()!\n") + fileName);
    }
}

void NumpyFile::LoadFile(const char* fileName, uint32_t arrayIndex, std::string& ptrName, std::vector<uint8_t>& memory, uint32_t* ptrNumRows,
                         uint32_t* ptrNumColumns, uint32_t* ptrNumBytesPerElement) {
    cnpy::npz_t my_npz1 = cnpy::npz_load(fileName);
    auto it = my_npz1.begin();
    std::advance(it, arrayIndex);
    if (it != my_npz1.end()) {
        ptrName = it->first;
        cnpy::NpyArray my_npy = it->second;
        *ptrNumRows = my_npy.shape[0];
        *ptrNumColumns = my_npy.shape[1];

        for (size_t i = 0; i < my_npy.data_holder->size(); i++) {
            memory.at(i) = my_npy.data_holder->at(i);
        }

        *ptrNumBytesPerElement = sizeof(float);
    } else {
        throw std::runtime_error(std::string("Failed to open %s for reading in LoadFile()!\n") + fileName);
    }
}

void NumpyFile::SaveFile(const char* fileName, bool shouldAppend, std::string name, void* ptrMemory, uint32_t numRows, uint32_t numColumns) {
    std::string mode;
    shouldAppend ? mode = "a" : mode = "w";
    std::vector<size_t> shape {numRows, numColumns};
    cnpy::npz_save(fileName, name, reinterpret_cast<float*>(ptrMemory), shape, mode);
}
