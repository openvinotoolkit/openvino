/**
 * Unit tests for the action's main functionality, src/restoreImpl.js
 */
const core = require('@actions/core');
const path = require('path');
const os = require('os');
const fs = require('fs');
const tar = require('tar');
const restoreImpl = require('../src/restoreImpl');

const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'test-restore-'));
const cacheLocalPath = path.join(tempDir, 'cache_local');
const cacheRemotePath = path.join(tempDir, 'cache_remote');
const cacheTmpPath = path.join(tempDir, 'cache_tmp');

const cacheFiles = ['cache_1.cache', 'cache_2.cache', 'cache_3.cache'];
const testFiles = ['file1.txt', 'file2.txt', 'file3.txt'];

// Mock the GitHub Actions core library
const getInputMock = jest.spyOn(core, 'getInput').mockImplementation();
const setOutputMock = jest.spyOn(core, 'setOutput').mockImplementation();
const setFailedMock = jest.spyOn(core, 'setFailed').mockImplementation();

// Clean up mock file system after each test
afterEach(() => {
  fs.rmSync(tempDir, { recursive: true });
});

// Mock the action's main function
const runMock = jest.spyOn(restoreImpl, 'restore');

describe('restore', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Set up mock file system before each test
    fs.mkdirSync(cacheLocalPath, { recursive: true });
    fs.mkdirSync(cacheRemotePath, { recursive: true });
    fs.mkdirSync(cacheTmpPath, { recursive: true });

    // Define test files
    const file1Path = path.join(cacheTmpPath, testFiles[0]);
    fs.writeFileSync(file1Path, 'File 1 contents');
    tar.c(
      {
        gzip: true,
        file: cacheFiles[0],
        cwd: cacheTmpPath,
        sync: true
      },
      ['.']
    );
    fs.renameSync(cacheFiles[0], path.join(cacheRemotePath, cacheFiles[0]));
    fs.utimesSync(
      path.join(cacheRemotePath, cacheFiles[0]),
      new Date(Date.now() - 1500),
      new Date(Date.now() - 1500)
    );

    //
    const file2Path = path.join(cacheTmpPath, testFiles[1]);
    fs.writeFileSync(file2Path, 'File 2 contents');
    tar.c(
      {
        gzip: true,
        file: cacheFiles[1],
        cwd: cacheTmpPath,
        sync: true
      },
      ['.']
    );
    fs.renameSync(cacheFiles[1], path.join(cacheRemotePath, cacheFiles[1]));
    fs.utimesSync(
      path.join(cacheRemotePath, cacheFiles[1]),
      new Date(Date.now() - 1000),
      new Date(Date.now() - 1000)
    );
    //
    const file3Path = path.join(cacheTmpPath, testFiles[2]);
    fs.writeFileSync(file3Path, 'File 3 contents');
    tar.c(
      {
        gzip: true,
        file: cacheFiles[2],
        cwd: cacheTmpPath,
        sync: true
      },
      ['.']
    );
    fs.renameSync(cacheFiles[2], path.join(cacheRemotePath, cacheFiles[2]));
    fs.utimesSync(
      path.join(cacheRemotePath, cacheFiles[2]),
      new Date(Date.now() - 500),
      new Date(Date.now() - 500)
    );
  });

  it('gets the cache file', async () => {
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'key';
        case 'restore-keys':
          return 'cache';
        case 'path':
          return cacheLocalPath;
        default:
          return '';
      }
    });

    await restoreImpl.restore();
    expect(runMock).toHaveReturned();

    // Verify that all of the core library functions were called correctly
    expect(setOutputMock).toHaveBeenNthCalledWith(
      1,
      'cache-file',
      cacheFiles[2]
    );
    expect(setOutputMock).toHaveBeenNthCalledWith(2, 'cache-hit', true);
    let id = 1;
    for (const filename of testFiles) {
      const filePath = path.join(cacheLocalPath, filename);
      expect(fs.existsSync(filePath)).toBe(true);
      const fileContent = fs.readFileSync(filePath, 'utf8');
      expect(fileContent).toBe(`File ${id++} contents`);
    }
  });

  it('gets the updated cache file', async () => {
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'key';
        case 'restore-keys':
          return 'cache';
        case 'path':
          return cacheLocalPath;
        default:
          return '';
      }
    });

    // updated cache 2 access time
    fs.utimesSync(
      path.join(cacheRemotePath, cacheFiles[1]),
      new Date(Date.now() - 100),
      new Date(Date.now() - 100)
    );

    await restoreImpl.restore();
    expect(runMock).toHaveReturned();

    // Verify that all of the core library functions were called correctly
    expect(setOutputMock).toHaveBeenNthCalledWith(
      1,
      'cache-file',
      cacheFiles[1]
    );
    expect(setOutputMock).toHaveBeenNthCalledWith(2, 'cache-hit', true);
    let id = 1;
    for (const filename of testFiles.slice(0, 1)) {
      const filePath = path.join(cacheLocalPath, filename);
      expect(fs.existsSync(filePath)).toBe(true);
      const fileContent = fs.readFileSync(filePath, 'utf8');
      expect(fileContent).toBe(`File ${id++} contents`);
    }
    // check that file3 is absent
    expect(fs.existsSync(path.join(cacheLocalPath, testFiles[2]))).toBe(false);
  });

  it('gets the cache file and extract to absent dir', async () => {
    const cacheAbsentPath = path.join(tempDir, 'cache_absent');
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'key';
        case 'restore-keys':
          return 'cache';
        case 'path':
          return cacheAbsentPath;
        default:
          return '';
      }
    });

    await restoreImpl.restore();
    expect(runMock).toHaveReturned();

    // Verify that all of the core library functions were called correctly
    expect(setOutputMock).toHaveBeenNthCalledWith(
      1,
      'cache-file',
      cacheFiles[2]
    );
    expect(setOutputMock).toHaveBeenNthCalledWith(2, 'cache-hit', true);
    let id = 1;
    for (const filename of testFiles) {
      const filePath = path.join(cacheAbsentPath, filename);
      expect(fs.existsSync(filePath)).toBe(true);
      const fileContent = fs.readFileSync(filePath, 'utf8');
      expect(fileContent).toBe(`File ${id++} contents`);
    }
  });

  it('Test when no cache found', async () => {
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'key';
        case 'path':
          return cacheLocalPath;
        default:
          return '';
      }
    });

    await restoreImpl.restore();
    expect(runMock).toHaveReturned();

    // Verify that all of the core library functions were called correctly
    expect(setOutputMock).toHaveBeenNthCalledWith(1, 'cache-file', '');
    expect(setOutputMock).toHaveBeenNthCalledWith(2, 'cache-hit', false);
    for (const filename of testFiles) {
      const filePath = path.join(cacheLocalPath, filename);
      expect(fs.existsSync(filePath)).toBe(false);
    }
    expect(setFailedMock).not.toHaveBeenCalled();
  });

  it('Unsupported cache found', async () => {
    const cacheFakePath = path.join(cacheRemotePath, 'cache-fake.cache');
    fs.writeFileSync(cacheFakePath, 'Fake content');
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'cache';
        case 'path':
          return cacheLocalPath;
        default:
          return '';
      }
    });

    await restoreImpl.restore();
    expect(runMock).toHaveReturned();
    expect(setOutputMock).not.toHaveBeenCalled();
    expect(setFailedMock).not.toHaveBeenCalled();
  });
});
