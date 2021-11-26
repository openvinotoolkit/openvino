#include <stdio.h>
#include <stdlib.h>

#include "bmp_reader.h"


int readBmpImage(const char* fileName, BitMap* image) {
    FILE* input;

    if (fopen_s(&input, fileName, "rb") != 0) {
        printf_s("[BMP] file %s is not opened\n", fileName);
        return 1;
    }

    fread_s(&image->header.type, 2, 2, 1, input);

    if (image->header.type != 'M' * 256 + 'B') {
        printf_s("[BMP] file is not bmp type\n");
        return 2;
    }

    fread_s(&image->header.size, 4, 4, 1, input);
    fread_s(&image->header.reserved, 4, 4, 1, input);
    fread_s(&image->header.offset, 4, 4, 1, input);

    fread_s(&image->infoHeader, sizeof(BmpInfoHeader), sizeof(BmpInfoHeader), 1, input);

    image->width = image->infoHeader.width;
    image->height = image->infoHeader.height;

    if (image->infoHeader.bits != 24) {
        printf_s("[BMP] 24bpp only supported. But input has: %d\n", image->infoHeader.bits);
        return 3;
    }

    if (image->infoHeader.compression != 0) {
        printf_s("[BMP] compression not supported\n");
        return 4;
    }

    int padSize = image->width & 3;
    char pad[3];
    size_t size = image->width * image->height * 3;

    image->data = malloc(sizeof(char) * size);

    fseek(input, image->header.offset, 0);

    // reading by rows in invert vertically
    for (int i = 0; i < image->height; i++) {
        unsigned int storeAt = image->infoHeader.height < 0 ? i : (unsigned int)image->height - 1 - i;
        fread_s(image->data + image->width * 3 * storeAt, image->width * 3, image->width * 3, 1, input);
        fread_s(pad, padSize, padSize, 1, input);
    }

    fclose(input);
    return 0;
}
