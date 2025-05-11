// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

typedef struct BmpHeaderType {
    unsigned short type; /* Magic identifier            */
    unsigned int size;   /* File size in bytes          */
    unsigned int reserved;
    unsigned int offset; /* Offset to image data, bytes */
} BmpHeader;

typedef struct BmpInfoHeaderType {
    unsigned int size;             /* Header size in bytes      */
    int width, height;             /* Width and height of image */
    unsigned short planes;         /* Number of colour planes   */
    unsigned short bits;           /* Bits per pixel            */
    unsigned int compression;      /* Compression type          */
    unsigned int imagesize;        /* Image size in bytes       */
    int xresolution, yresolution;  /* Pixels per meter          */
    unsigned int ncolours;         /* Number of colours         */
    unsigned int importantcolours; /* Important colours         */
} BmpInfoHeader;

typedef struct BitMapType {
    BmpHeader header;
    BmpInfoHeader infoHeader;
    int width, height;
    unsigned char* data;
} BitMap;

int readBmpImage(const char* fileName, BitMap* image);
