import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import {
  mkdtempSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
  rmSync
} from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';

const action = fileURLToPath(new URL('../dist/index.js', import.meta.url));

for (const fail of [false, true]) {
  test(`streams large pip output and ${fail ? 'reports failures' : 'preserves arguments'}`, () => {
    const directory = mkdtempSync(path.join(tmpdir(), 'install-wheel-'));
    try {
      const wheels = path.join(directory, 'local wheels');
      mkdirSync(wheels);
      writeFileSync(path.join(wheels, 'whowhatbench-1.0-py3-none-any.whl'), '');
      const argsFile = path.join(directory, 'arguments.json');
      writeFileSync(
        path.join(directory, 'python'),
        `#!/usr/bin/env node
const fs = require('node:fs');
const args = process.argv.slice(2);
if (args[0] === '--version') {
  console.log('Python 3.12.3');
} else {
  fs.writeFileSync(process.env.ARGS_FILE, JSON.stringify(args));
  process.stdout.write('x'.repeat(2 * 1024 * 1024));
  process.stderr.write('pip diagnostic\\n');
  process.exitCode = ${fail ? 7 : 0};
}
`,
        { mode: 0o755 }
      );
      const result = spawnSync(process.execPath, [action], {
        env: {
          ...process.env,
          PATH: `${directory}${path.delimiter}${process.env.PATH}`,
          INPUT_PACKAGES: 'whowhatbench[llm-test-openvino]',
          INPUT_LOCAL_WHEEL_DIR: wheels,
          INPUT_REQUIREMENTS_FILES: path.join(
            directory,
            'test requirements.txt'
          ),
          ARGS_FILE: argsFile
        },
        encoding: 'utf8',
        maxBuffer: 16 * 1024 * 1024
      });
      assert.equal(result.error, undefined);
      assert.equal(result.status, fail ? 1 : 0);
      assert.ok(result.stdout.includes('x'.repeat(2 * 1024 * 1024)));
      assert.ok(result.stderr.includes('pip diagnostic'));
      assert.deepEqual(JSON.parse(readFileSync(argsFile, 'utf8')), [
        '-m',
        'pip',
        'install',
        `${path.join(wheels, 'whowhatbench-1.0-py3-none-any.whl')}[llm-test-openvino]`,
        '-r',
        path.join(directory, 'test requirements.txt')
      ]);
      if (fail) {
        assert.ok(
          result.stdout.includes(
            'Attempt 3 failed: pip install exited with code 7'
          )
        );
      }
    } finally {
      rmSync(directory, { recursive: true, force: true });
    }
  });
}
