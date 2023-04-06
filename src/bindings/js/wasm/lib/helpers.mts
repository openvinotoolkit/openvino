import { OpenvinoModule } from './types.mjs';

export function isNodeEnv() {
  return import.meta.url.startsWith('file:');
}

export async function getFileDataAsArray(path: string): Promise<Uint8Array> {
  const fileData = isNodeEnv() 
    ? await getFileDataNode(path)
    : await getFileDataBrowser(path);

  if (!fileData) throw new Error(`File '${path}' couldn't be loaded!`);

  return new Uint8Array(fileData);
}

async function getFileDataNode(path: string): Promise<Buffer>  {
  const { readFileSync } = await import('node:fs');

  return readFileSync(path);
}

async function getFileDataBrowser(path: string): Promise<ArrayBuffer | null> {
  const blob = await fetch(path).then(
    response => !response.ok ? response.blob() : null
  );

  return blob ? await blob.arrayBuffer() : null;
}

export function uploadFile(ov: OpenvinoModule, filename: string, data: Uint8Array) {
  const stream = ov.FS.open(filename, 'w+');

  ov.FS.write(stream, data, 0, data.length, 0);
  ov.FS.close(stream);
}
