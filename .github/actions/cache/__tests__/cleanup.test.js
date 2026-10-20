/**
 * Unit tests for the action's main functionality, src/cleanupImpl.js
 */
import {
  jest,
  describe,
  it,
  beforeEach,
  afterEach,
  expect
} from '@jest/globals';
import path from 'path';
import os from 'os';
import fs from 'fs';

// Mock the GitHub Actions core library
jest.unstable_mockModule('@actions/core', () => ({
  getInput: jest.fn(),
  setOutput: jest.fn(),
  setFailed: jest.fn(),
  debug: jest.fn(),
  info: jest.fn(),
  warning: jest.fn(),
  error: jest.fn()
}));

const core = await import('@actions/core');
const cleanupImpl = await import('../src/cleanupImpl.js');

const getInputMock = core.getInput;
const setFailedMock = core.setFailed;

const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'test-cleanup-'));
const cacheRemotePath = path.join(
  tempDir,
  'subdir_1',
  'subdir_2',
  'cache_remote'
);

const cacheFiles = ['cache_1.cache', 'cache_2.cache', 'cache_3.cache'];
const minAccessTime = 7 * 24 * 60 * 60 * 1000; // 1 week

describe('cleanup', () => {
  beforeEach(() => {
    jest.clearAllMocks();

    // Set up file system before each test
    fs.mkdirSync(cacheRemotePath, { recursive: true });

    const fileSizeInBytes = 1024 * 1024 * 1024; // 1 GB
    const buffer = Buffer.alloc(fileSizeInBytes);
    let id = 1;
    for (const cache of cacheFiles) {
      const cachePath = path.join(cacheRemotePath, cache);
      fs.writeFileSync(cachePath, buffer);
      fs.utimesSync(
        cachePath,
        new Date(Date.now() - minAccessTime * id++),
        new Date()
      );
    }
  });

  // Clean up mock file system after each test
  afterEach(() => {
    fs.rmSync(tempDir, { recursive: true });
  });

  it('Cleanup old files using restore key', async () => {
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'key';
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 1;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    // cache2 and cache3 should be removed
    for (const cache of cacheFiles.slice(1, 2)) {
      expect(fs.existsSync(path.join(cacheRemotePath, cache))).toBe(false);
    }
    // check that file1 exists
    expect(fs.existsSync(path.join(cacheRemotePath, cacheFiles[0]))).toBe(true);
  });

  it('Cleanup old files using key', async () => {
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'cache';
        case 'cache-size':
          return 1;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    // cache2 and cache3 should be removed
    for (const cache of cacheFiles.slice(1, 2)) {
      expect(fs.existsSync(path.join(cacheRemotePath, cache))).toBe(false);
    }
    // check that file1 exists
    expect(fs.existsSync(path.join(cacheRemotePath, cacheFiles[0]))).toBe(true);
  });

  it('Skip cleanup old files by size', async () => {
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'key';
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 5;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    for (const cache of cacheFiles) {
      expect(fs.existsSync(path.join(cacheRemotePath, cache))).toBe(true);
    }
  });

  it('Skip cleanup old files by atime', async () => {
    for (const cache of cacheFiles) {
      const cachePath = path.join(cacheRemotePath, cache);
      fs.utimesSync(
        cachePath,
        new Date(Date.now() - minAccessTime / 2),
        new Date()
      );
    }

    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'key';
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 1;
        case 'max-cache-size':
          return 5;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    for (const cache of cacheFiles) {
      expect(fs.existsSync(path.join(cacheRemotePath, cache))).toBe(true);
    }
  });

  it('Cleanup recently used files by max cache limit', async () => {
    for (const cache of cacheFiles) {
      const cachePath = path.join(cacheRemotePath, cache);
      fs.utimesSync(
        cachePath,
        new Date(Date.now() - minAccessTime / 2),
        new Date()
      );
    }

    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'key';
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 1;
        case 'max-cache-size':
          return 2;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    // cache2 and cache3 should be removed
    for (const cache of cacheFiles.slice(1, 2)) {
      expect(fs.existsSync(path.join(cacheRemotePath, cache))).toBe(false);
    }
    // check that file1 exists
    expect(fs.existsSync(path.join(cacheRemotePath, cacheFiles[0]))).toBe(true);
  });

  it('Test unexpected behaviour', async () => {
    // Set folder permissions to read-only
    fs.chmodSync(cacheRemotePath, '555');
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'key':
          return 'cache';
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 2;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    expect(setFailedMock).not.toHaveBeenCalled();

    fs.chmodSync(cacheRemotePath, '777');
  });

  it('Cleanup absent directory', async () => {
    const cachePathAbsent = path.join(tempDir, 'cache_remote_absent');
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cachePathAbsent;
        case 'key':
          return 'cache';
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 2;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    expect(setFailedMock).not.toHaveBeenCalled();
  });

  it('Cleanup directory with subdirectory', async () => {
    const cacheSubPath = path.join(cacheRemotePath, 'cache_subdir.cache');
    fs.mkdirSync(cacheSubPath, { recursive: true });
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheRemotePath;
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 1;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    // cache2 and cache3 should be removed
    for (const cache of cacheFiles.slice(1, 2)) {
      expect(fs.existsSync(path.join(cacheRemotePath, cache))).toBe(false);
    }
    // check that file1 exists
    expect(fs.existsSync(path.join(cacheRemotePath, cacheFiles[0]))).toBe(true);
    // check that sub directory exists
    expect(fs.existsSync(cacheSubPath)).toBe(true);
  });

  it('Cleanup directory with subdirectories and files (recursive=true)', async () => {
    const cacheSubPath = path.join(tempDir, 'subdir_1');
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheSubPath;
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 1;
        case 'recursive':
          return true;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    // cache2 and cache3 should be removed
    for (const cache of cacheFiles.slice(1, 2)) {
      expect(fs.existsSync(path.join(cacheRemotePath, cache))).toBe(false);
    }
    // check that file1 exists
    expect(fs.existsSync(path.join(cacheRemotePath, cacheFiles[0]))).toBe(true);
  });

  it('Cleanup directory with subdirectories and files (recursive=false)', async () => {
    const cacheSubPath = path.join(tempDir, 'subdir_1');
    // Set the action's inputs as return values from core.getInput()
    getInputMock.mockImplementation(name => {
      switch (name) {
        case 'cache-path':
          return cacheSubPath;
        case 'restore-keys':
          return 'cache';
        case 'cache-size':
          return 1;
        case 'recursive':
          return false;
        default:
          return '';
      }
    });

    await cleanupImpl.cleanUp();

    // cache2 and cache3 should be removed
    for (const cache of cacheFiles) {
      expect(fs.existsSync(path.join(cacheRemotePath, cache))).toBe(true);
    }
  });
});
