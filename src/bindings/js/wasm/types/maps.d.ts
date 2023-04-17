import { PrecisionSupportedType, JSArrayType } from 'openvinojs-common';
import { OVType, HEAPType } from './types.js';
export declare const ovTypesMap: {
    [Type in OVType]: PrecisionSupportedType;
};
export declare const heapLabelByArrayTypeMap: {
    [ArrayType in keyof JSArrayType as string]: HEAPType;
};
