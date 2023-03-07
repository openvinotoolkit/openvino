import { OpenvinoModule, OriginalTensor, OriginalTensorWrapper } from './ov-module';

import { 
  TypedArray,
  PrecisionSupportedType, 
  IShape,
  ITensor,
} from './types';

import {
  jsTypeByPrecisionMap, 
  ovTypesMap, 
  heapLabelByArrayTypeMap, 
} from './maps';

import Shape from './shape';

export default class Tensor implements ITensor {
  #precision: PrecisionSupportedType;
  #data: TypedArray;
  #shape: IShape;

  constructor(precision: PrecisionSupportedType, data: number[] | TypedArray, shapeData: Shape | number[]) {
    this.#precision = precision;
    this.#data = new jsTypeByPrecisionMap[this.#precision](data);
    
    if (shapeData instanceof Shape) this.#shape = shapeData;
    else this.#shape = new Shape(...shapeData);
  }

  get precision(): PrecisionSupportedType {
    return this.#precision;
  }

  get data(): TypedArray {
    return this.#data;
  }

  get shape(): IShape {
    return this.#shape;
  }

  static parse(ov: OpenvinoModule, originalTensor: OriginalTensor): ITensor {
    const precision = ovTypesMap[originalTensor.getPrecision()];
    const shape = Shape.parse(ov, originalTensor.getShape());

    const dataType = jsTypeByPrecisionMap[precision];
    const heapTypeLabel = heapLabelByArrayTypeMap[dataType.name];
    const originalDataPointer = originalTensor.getData();

    const elementsCount = shape.data.reduce((acc, val) => acc*val);
    const data: TypedArray = new dataType(elementsCount);

    for (let i = 0; i < elementsCount; i++) {
      const element = ov[heapTypeLabel][originalDataPointer/dataType.BYTES_PER_ELEMENT + i];
      
      data[i] = element;
    }

    return new Tensor(precision, data, shape);
  }

  static convert(ov: OpenvinoModule, tensor: Tensor): OriginalTensorWrapper {
    const { precision } = tensor;
    const dataType = jsTypeByPrecisionMap[precision];
    const originalShape = tensor.shape.convert(ov);

    const originalData = new dataType(tensor.data);
    const elementSizeInBytes = originalData.BYTES_PER_ELEMENT;
    const heapSpace = ov._malloc(originalData.length*elementSizeInBytes);
    const offset = Math.log2(elementSizeInBytes);
    const waPrecision = heapLabelByArrayTypeMap[dataType.name];

    ov[waPrecision].set(originalData, heapSpace>>offset); 

    return { 
      obj: new ov.Tensor(precision, heapSpace, originalShape.obj),
      free: () => {
        originalShape.free();
        ov._free(heapSpace);
      }
     };
  }

  convert(ov: OpenvinoModule): OriginalTensorWrapper {
    return Tensor.convert(ov, this);
  }
};
