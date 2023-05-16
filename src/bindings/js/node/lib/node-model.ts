import { Tensor, Shape, TypedArray } from 'openvinojs-common';

import type { ovNodeModule, NodeModel, NodeTensor } from './types';
import type {
  ITensor,
  IShape,
  IModel,
  PrecisionSupportedType,
} from 'openvinojs-common';

/* eslint-disable @typescript-eslint/no-var-requires */
const ovNode: ovNodeModule = require('../build/Release/ov_node_addon.node');

const DEFAULT_PRECISION = 'f32';

export default async function loadModel(xmlPath: string, binPath: string)
: Promise<IModel> {
  if (typeof xmlPath !== 'string' || typeof binPath !== 'string')
    throw new Error('Parameters \'xmlPath\' and \'binPath\' should be string');

  const model = new ovNode.Model().read_model(xmlPath).compile('CPU');

  return new CommonModel(ovNode, model);
}

class CommonModel implements IModel {
  #ovNode: ovNodeModule;
  #nodeModel: NodeModel;

  constructor(ovNode: ovNodeModule, nodeModel: NodeModel) {
    this.#ovNode = ovNode;
    this.#nodeModel = nodeModel;
  }

  async infer(
    tensorOrDataArray: ITensor | number[] | TypedArray,
    shapeOrDimensionsArray?: IShape | number[],
  )
  : Promise<ITensor> {
    const shape = shapeOrDimensionsArray instanceof Shape
      ? shapeOrDimensionsArray
      : new Shape(shapeOrDimensionsArray as number[] || []);

    const tensor = tensorOrDataArray instanceof Tensor
      ? tensorOrDataArray
      : new Tensor(DEFAULT_PRECISION, tensorOrDataArray as number[], shape);

    const precision = tensorOrDataArray instanceof Tensor
      ? tensorOrDataArray.precision
      : DEFAULT_PRECISION;

    const nodeTensor = new this.#ovNode.Tensor(
      precision,
      tensor.shape.data,
      tensor.data,
    );
    const output = this.#nodeModel.infer(nodeTensor);

    return parseNodeTensor(output);
  }
}

function parseNodeTensor(nodeTensor: NodeTensor): Tensor {
  const precision = nodeTensor.getPrecision() as PrecisionSupportedType;
  const data = nodeTensor.data;
  const shape = new Shape(nodeTensor.getShape().getData());

  return new Tensor(precision, data, shape);
}
