import { Shape } from 'openvinojs-common';
import type { IModel } from 'openvinojs-common';
export default function loadModel(xmlPath: string, binPath: string, shapeData: Shape | number[], layout: string): Promise<IModel>;
