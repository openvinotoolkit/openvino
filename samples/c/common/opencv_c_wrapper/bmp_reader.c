#include "bmp_reader.h"

#include <stdio.h>
#include <stdlib.h>

int readBmpImage(const char* fileName, BitMap* image) {
    FILE* input = fopen(fileName, "rb");

    if (input == NULL) {
        printf("[BMP] file %s is not opened\n", fileName);
        return 1;
    }

    fread(&image->header.type, 2, 1, input);

    if (image->header.type != 'M' * 256 + 'B') {
        printf("[BMP] file is not bmp type\n");
        return 2;
    }

    fread(&image->header.size, 4, 1, input);
    fread(&image->header.reserved, 4, 1, input);
    fread(&image->header.offset, 4, 1, input);

    fread(&image->infoHeader, sizeof(BmpInfoHeader), 1, input);

    image->width = image->infoHeader.width;
    image->height = image->infoHeader.height;

    if (image->infoHeader.bits != 24) {
        printf("[BMP] 24bpp only supported. But input has: %d\n", image->infoHeader.bits);
        return 3;
    }

    if (image->infoHeader.compression != 0) {
        printf("[BMP] compression not supported\n");
        return 4;
    }

    int padSize = image->width & 3;
    char pad[3];
    size_t size = image->width * image->height * 3;

    image->data = malloc(sizeof(char) * size);

    fseek(input, image->header.offset, 0);

    // reading by rows in invert vertically
    int i;
    for (i = 0; i < image->height; i++) {
        unsigned int storeAt = image->infoHeader.height < 0 ? i : (unsigned int)image->height - 1 - i;
        fread(image->data + image->width * 3 * storeAt, image->width * 3, 1, input);
        fread(pad, padSize, 1, input);
    }

    fclose(input);
    return 0;
}
