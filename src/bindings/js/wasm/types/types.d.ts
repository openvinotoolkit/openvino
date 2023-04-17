import type { TypedArray, PrecisionSupportedType } from 'openvinojs-common';
import type { OriginalModel } from './ov-module.js';
export type OVType = 'uint8_t' | 'int8_t' | 'uint16_t' | 'int16_t' | 'uint32_t' | 'int32_t' | 'float' | 'double';
export type HEAPType = 'HEAP8' | 'HEAPU8' | 'HEAP16' | 'HEAPU16' | 'HEAP32' | 'HEAPU32' | 'HEAPF32' | 'HEAPF64';
interface WASMFilesystem {
    open(filename: string, flags: string): string;
    write(stream: string, data: Uint8Array, position: number, length: number, from: number): void;
    close(stream: string): void;
}
export interface OpenvinoModule {
    FS: WASMFilesystem;
    HEAP8: TypedArray;
    HEAPU8: TypedArray;
    HEAP16: TypedArray;
    HEAPU16: TypedArray;
    HEAP32: TypedArray;
    HEAPU32: TypedArray;
    HEAPF32: TypedArray;
    HEAPF64: TypedArray;
    _malloc: (amount: number) => number;
    _free: (heapPointer: number) => void;
    Shape: new (heapPointer: number, dimensions: number) => OriginalShape;
    Tensor: new (precision: PrecisionSupportedType, heapPointer: number, shape: OriginalShape) => OriginalTensor;
    Session: new (xmlFilename: string, binFilename: string, originalShapeObj: OriginalShape, layout: string) => OriginalModel;
    getVersionString(): string;
    getDescriptionString(): string;
}
export interface OriginalShape {
    getDim(): number;
    getData(): number;
}
export interface OriginalTensor {
    getPrecision(): OVType;
    getShape(): OriginalShape;
    getData(): number;
}
export {};
