// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {execFileSync} = require('node:child_process');
const {test, after} = require('node:test');
const {Aggregator, engine, latestJobs} = require('./aggregate.js');
const {PLATFORMS} = require('./comment.js');

const root = path.resolve(__dirname, '..', '..', '..');
const workspace = path.join(root, `.bom-aggregate-test-${process.pid}`);
fs.mkdirSync(workspace);
process.chdir(workspace);
after(() => {
    process.chdir(root);
    fs.rmSync(workspace, {recursive: true, force: true});
});

function fixture() {
    const state = {calls: [], comments: [], outputs: {}, runs: new Map(), jobs: new Map(),
        artifacts: new Map(), downloads: 0, patchCalls: [], patch: 'validated-patch\n'};
    state.pr = {number: 7, state: 'open', changed_files: 1, created_at: '2025-12-31T23:00:00Z',
        head: {sha: 'a'.repeat(40), ref: 'feature', repo: {id: 2, owner: {login: 'fork'}, name: 'repo'}},
        base: {sha: 'b'.repeat(40), ref: 'master', repo: {id: 1}}};
    state.prs = [state.pr];
    state.files = [{filename: 'src/core/file.cpp'}];
    const workflows = new Map(Object.entries(PLATFORMS).map(([file, [name]], index) =>
        [file, {id: index + 10, path: `.github/workflows/${file}`, name}]));
    let id = 100;
    for (const [file, workflow] of workflows) {
        const run = {id: id++, workflow_id: workflow.id, path: workflow.path, name: workflow.name,
            repository: {id: 1}, head_repository: {id: 2, owner: {login: 'fork'}},
            head_branch: 'feature', head_sha: state.pr.head.sha, event: 'pull_request',
            status: 'completed', conclusion: 'failure', run_number: 1, run_attempt: 1,
            created_at: '2026-01-01T00:00:00Z',
            pull_requests: []}; // Real fork payloads may have no PR association.
        state.runs.set(file, [run]);
        state.jobs.set(run.id, [
            {id: run.id * 10, name: file === 'manylinux_2_28.yml' ? 'Build' : 'Build / Build',
                run_id: run.id, run_attempt: 1, status: 'completed', conclusion: 'success',
                started_at: '2026-01-01T00:00:00Z', completed_at: '2026-01-01T01:00:00Z',
                steps: [{name: 'Upload CPack component BOM', conclusion: 'success'}]},
            {id: run.id * 10 + 1, name: 'Smart_CI', run_id: run.id, run_attempt: 1,
                status: 'completed', conclusion: 'success',
                steps: [{name: 'Get affected components', conclusion: 'success'}]},
            {id: run.id * 10 + 2, name: 'Docker', run_id: run.id, run_attempt: 1,
                status: 'completed', conclusion: 'success'},
            {id: run.id * 10 + 3, name: 'BOM tests / BOM tests', run_id: run.id, run_attempt: 1,
                status: 'completed', conclusion: 'failure'},
        ]);
        state.artifacts.set(run.id, [{id: run.id + 1000, name: `bom_${PLATFORMS[file][1]}`,
            expired: false, size_in_bytes: 10, created_at: '2026-01-01T00:30:00Z',
            workflow_run: {id: run.id, head_sha: state.pr.head.sha, head_repository_id: 2}}]);
    }
    state.context = {repo: {owner: 'owner', repo: 'repo'}, runId: 900,
        payload: {action: 'completed', repository: {id: 1, default_branch: 'master'},
            workflow_run: structuredClone(state.runs.get('ubuntu_22.yml')[0])}};
    const endpoint = (key, fn) => async args => {
        state.calls.push({key, args});
        if (state.fail === key) throw new Error('API unavailable');
        return {data: await fn(args)};
    };
    const github = {rest: {
        pulls: {
            get: endpoint('pr', () => state.pr),
            list: endpoint('prs', () => state.prs),
            listFiles: endpoint('files', () => state.files),
        },
        actions: {
            getWorkflow: endpoint('workflow', args => workflows.get(args.workflow_id)),
            listWorkflowRuns: endpoint('runs', args => {
                const file = [...workflows].find(([, value]) => value.id === args.workflow_id)[0];
                return state.runs.get(file);
            }),
            getWorkflowRun: endpoint('run', args => [...state.runs.values()].flat()
                .find(run => run.id === args.run_id)),
            listJobsForWorkflowRun: endpoint('jobs', args => state.jobs.get(args.run_id)),
            listWorkflowRunArtifacts: endpoint('artifacts', args => state.artifacts.get(args.run_id)),
            downloadArtifact: endpoint('download', args => {
                state.downloads++;
                if (state.onDownload) state.onDownload();
                return state.archive ? state.archive(args.artifact_id) : Buffer.from('mock-zip');
            }),
            getArtifact: endpoint('uploaded', () => state.uploaded || {
                id: 901, name: 'gold-bom-patch-900-1', expired: false, workflow_run: {id: 900},
            }),
        },
        issues: {
            listComments: endpoint('comments', () => state.comments),
            createComment: endpoint('create', args => {
                state.comments.push({id: 500 + state.comments.length, body: args.body,
                    user: {type: 'Bot', login: 'github-actions[bot]'}});
            }),
            updateComment: endpoint('update', args => {
                state.comments.find(comment => comment.id === args.comment_id).body = args.body;
            }),
            deleteComment: endpoint('delete', args => {
                state.comments = state.comments.filter(comment => comment.id !== args.comment_id);
            }),
        },
    }};
    github.paginate = async (method, args) => {
        assert.equal(args.per_page, 100);
        return (await method(args)).data;
    };
    github.graphql = async () => {
        if (state.graphqlError) throw new Error('GraphQL API unavailable');
        return {repository: {databaseId: 1, pullRequest: {
            number: state.pr.number, state: 'OPEN', headRefOid: state.pr.head.sha,
            baseRefName: state.pr.base.ref, createdAt: state.pr.created_at,
            timelineItems: {nodes: state.retargetedAt ? [{createdAt: state.retargetedAt}] : []},
        }}};
    };
    state.github = github;
    state.aggregator = new Aggregator({github, context: state.context,
        core: {setOutput: (key, value) => { state.outputs[key] = value; }},
        runEngine: input => {
            if (input.operation === 'policy') return engine(input);
            state.patchCalls.push(input);
            if (state.invalidData) throw new Error('Invalid manifest');
            return state.patch;
        }});
    const source = {
        directory: async parts => {
            if (parts[0] === '.github') return [...workflows.keys()].map(file => ({path: file}));
            if (state.goldMissing) return null;
            return [{path: 'core.json', content: '{"component":"core","files":["old"]}'}];
        },
        blob: async entry => entry.content ? Buffer.from(entry.content) :
            Buffer.from(state.workflowOverride?.[entry.path] ??
                fs.readFileSync(path.join(root, '.github', 'workflows', entry.path), 'utf8')),
    };
    state.realSource = state.aggregator.source.bind(state.aggregator);
    state.aggregator.source = async () => source;
    state.prepare = () => state.aggregator.prepare();
    state.publish = id => state.aggregator.publish(id);
    state.body = () => state.comments.find(comment => comment.body.startsWith('<!-- openvino-bom:unified-gold -->'))?.body;
    state.run = file => state.runs.get(file || 'ubuntu_22.yml')[0];
    state.build = file => state.jobs.get(state.run(file).id)[0];
    return state;
}

test('all seven expected Builds, not overall/BOM status, gate one uploaded proposal for a fork', async () => {
    const f = fixture();
    await f.prepare();
    assert.equal(f.outputs.upload, 'true');
    assert.equal(f.downloads, 7);
    assert.equal(f.patchCalls[0].platforms.length, 7);
    assert.equal(f.comments.length, 1);
    assert.doesNotMatch(f.body(), /Download patch artifact/);
    assert.ok(fs.existsSync('bom-gold-output/pr-7-gold.patch'));
    await f.publish('901');
    assert.equal(f.comments.length, 1);
    assert.match(f.body(), /actions\/runs\/900\/artifacts\/901/);
    assert.match(f.body(), /engineering approval of the public package layout/);
    assert.match(f.body(), /git apply --check pr-7-gold.patch/);
    assert.ok(f.calls.filter(call => call.key === 'jobs').every(call => call.args.filter === 'all'));
});

for (const state of ['missing', 'queued', 'in_progress', 'failure', 'cancelled', 'skipped']) {
    test(`never publishes a partial proposal when one Build is ${state}`, async () => {
        const f = fixture();
        if (state === 'missing') f.runs.set('windows_vs2022_debug.yml', []);
        else if (['queued', 'in_progress'].includes(state)) f.run('windows_vs2022_debug.yml').status = state;
        else f.build('windows_vs2022_debug.yml').conclusion = state;
        await f.prepare();
        assert.equal(f.outputs.upload, 'false');
        assert.equal(f.downloads, 0);
        assert.match(f.body(), /Waiting \/ blocked/);
        assert.doesNotMatch(f.body(), /Download patch artifact/);
    });
}

test('skipped platforms need successful Smart CI, prerequisites and a skipped BOM', async () => {
    const f = fixture();
    const jobs = f.jobs.get(f.run('linux_arm64.yml').id);
    jobs[0].conclusion = jobs[3].conclusion = 'skipped';
    await f.prepare();
    assert.equal(f.downloads, 6);
    assert.match(f.body(), /skipped by Smart CI/);
    assert.ok(!f.patchCalls[0].platforms.some(platform => platform.configuration === 'ubuntu_22_04_arm64'));
    jobs[2].conclusion = 'failure';
    f.downloads = 0;
    await f.prepare();
    assert.equal(f.outputs.upload, 'false');
    assert.equal(f.downloads, 0);
});

test('Windows 2026 is expected only for its trigger path; missing runs still block it', async () => {
    const f = fixture();
    f.files = [{filename: '.github/workflows/windows_vs2026_release.yml'}];
    f.runs.set('windows_vs2026_release.yml', []);
    await f.prepare();
    assert.equal(f.outputs.upload, 'false');
    assert.match(f.body(), /windows_vs2026_Release.*missing/);
});

for (const [label, change] of [
    ['missing artifact', f => { f.artifacts.set(f.run().id, []); }],
    ['expired artifact', f => { f.artifacts.get(f.run().id)[0].expired = true; }],
    ['wrong artifact SHA', f => { f.artifacts.get(f.run().id)[0].workflow_run.head_sha = 'c'.repeat(40); }],
    ['wrong artifact repository', f => { f.artifacts.get(f.run().id)[0].workflow_run.head_repository_id = 99; }],
    ['artifact from previous Build', f => { f.artifacts.get(f.run().id)[0].created_at = '2025-01-01T00:00:00Z'; }],
    ['wrong run repository', f => { f.run().repository.id = 99; }],
    ['wrong workflow ID', f => { f.run().workflow_id = 99; }],
    ['wrong base association', f => { f.run().pull_requests = [{
        number: 7, base: {ref: 'releases/old', repo: {id: 1}}, head: {sha: f.pr.head.sha, repo: {id: 2}},
    }]; }],
    ['wrong job identity', f => { f.build().run_id = 99; }],
    ['ambiguous Build', f => { f.jobs.get(f.run().id).push({...f.build(), name: 'Build'}); }],
    ['no verified upload', f => { f.build().steps = []; }],
    ['incomplete PR files', f => { f.pr.changed_files = 100; }],
    ['API error', f => { f.fail = 'artifacts'; }],
    ['malformed BOM data', f => { f.invalidData = true; }],
    ['archive digest mismatch', f => { f.artifacts.get(f.run().id)[0].digest = 'sha256:bad'; }],
]) {
    test(`fails closed and removes stale download links: ${label}`, async () => {
        const f = fixture();
        await f.prepare();
        await f.publish('901');
        assert.match(f.body(), /Download patch artifact/);
        change(f);
        await assert.rejects(f.prepare());
        assert.match(f.body(), /Blocked/);
        assert.doesNotMatch(f.body(), /Download patch artifact/);
        assert.ok(!fs.existsSync('bom-gold-output'));
    });
}

test('latest run, not the last completed run, controls pending and failed states', async () => {
    const f = fixture();
    f.runs.get('ubuntu_24.yml').push({...f.run('ubuntu_24.yml'), id: 999, run_number: 2, status: 'queued'});
    await f.prepare();
    assert.equal(f.outputs.upload, 'false');
    assert.match(f.body(), /pending/);
});

test('rerun failed jobs can reuse the earlier successful Build and its original artifact', async () => {
    const f = fixture();
    f.run().run_attempt = 2;
    await f.prepare();
    assert.equal(f.outputs.upload, 'true');
    f.jobs.get(f.run().id).push({...f.build(), id: 9999, run_attempt: 2, conclusion: 'failure'});
    await f.prepare();
    assert.equal(f.outputs.upload, 'false');
});

test('PR pushes while downloading prevent upload; old-SHA events retire old patches', async () => {
    const f = fixture();
    f.onDownload = () => {
        f.pr = {...f.pr, head: {...f.pr.head, sha: 'c'.repeat(40)}};
    };
    await f.prepare();
    assert.equal(f.outputs.upload, 'false');
    assert.match(f.body(), /superseded/);
    assert.ok(!fs.existsSync('bom-gold-output'));
    f.onDownload = null;
    await f.prepare();
    assert.equal(f.outputs.upload, 'false');
    assert.doesNotMatch(f.body(), /Download patch artifact/);
});

for (const change of ['head', 'base', 'rerun', 'upload', 'identity', 'expired']) {
    test(`final publication revalidates ${change} after upload`, async () => {
        const f = fixture();
        await f.prepare();
        if (change === 'head') f.pr.head.sha = 'c'.repeat(40);
        if (change === 'base') f.pr.base.ref = 'releases/2026';
        if (change === 'rerun') { f.run().run_attempt = 2; f.run().status = 'in_progress'; }
        if (change === 'identity') f.uploaded = {id: 901, workflow_run: {id: 1}};
        if (change === 'expired') f.uploaded = {id: 901, expired: true};
        const publish = () => f.publish(change === 'upload' ? '' : '901');
        if (['upload', 'identity', 'expired'].includes(change)) await assert.rejects(publish());
        else await publish();
        assert.doesNotMatch(f.body(), /Download patch artifact/);
    });
}

test('all-clear and all-Smart-CI-skipped retire a previous patch without uploading', async () => {
    const f = fixture();
    await f.prepare();
    await f.publish('901');
    f.patch = '';
    await f.prepare();
    assert.equal(f.outputs.upload, 'false');
    assert.match(f.body(), /All clear/);
    assert.doesNotMatch(f.body(), /Download patch artifact/);
    for (const jobs of f.jobs.values()) jobs[0].conclusion = jobs[3].conclusion = 'skipped';
    f.downloads = 0;
    await f.prepare();
    assert.equal(f.downloads, 0);
    assert.match(f.body(), /All expected platforms were skipped/);
});

test('unified comment repairs bot duplicates but never edits user markers', async () => {
    const f = fixture();
    const marker = '<!-- openvino-bom:unified-gold -->\n';
    f.comments = [
        {id: 1, body: marker, user: {type: 'User', login: 'github-actions[bot]'}},
        ...[2, 3].map(id => ({id, body: marker, user: {type: 'Bot', login: 'github-actions[bot]'}})),
    ];
    await f.prepare();
    assert.equal(f.comments.length, 2);
    assert.equal(f.comments[0].body, marker);
    assert.equal(f.comments[1].id, 2);
});

test('ambiguous fork target bases are blocked explicitly, never merged', async () => {
    const f = fixture();
    f.prs.push({...f.pr, number: 8});
    // The source PR fetcher in this fixture returns the same live PR twice.
    await assert.rejects(f.prepare(), /Multiple open PR bases/);
    assert.match(f.body(), /multiple open PR bases/);
});

test('trigger edits do not silently downgrade expected platforms', async () => {
    const f = fixture();
    const text = fs.readFileSync(path.join(root, '.github', 'workflows', 'ubuntu_22.yml'), 'utf8');
    f.workflowOverride = {'ubuntu_22.yml': text.replace('  pull_request:', '  disabled_pull_request:')};
    await assert.rejects(f.prepare());
    assert.match(f.body(), /Blocked/);
});

test('missing baseline configurations are passed to strict patch generation as additions', async () => {
    const f = fixture();
    f.goldMissing = true;
    await f.prepare();
    assert.equal(f.outputs.upload, 'true');
    assert.ok(f.patchCalls[0].platforms.every(platform => platform.gold.length === 0));
});

test('source tree/blob validation rejects symlinks, truncation and hash tampering', async () => {
    const f = fixture();
    const raw = Buffer.from('trusted blob');
    const sha = crypto.createHash('sha1').update(`blob ${raw.length}\0`).update(raw).digest('hex');
    const entry = {path: 'boms', type: 'tree', mode: '040000', sha: 'tree'};
    f.github.rest.repos = {get: async () => ({data: {id: 2}})};
    f.github.rest.git = {
        getCommit: async () => ({data: {sha: f.pr.head.sha, tree: {sha: 'root'}}}),
        getTree: async args => ({data: {sha: args.tree_sha, truncated: false, tree: [entry]}}),
        getBlob: async () => ({data: {sha, size: raw.length, encoding: 'base64', content: raw.toString('base64')}}),
    };
    let source = await f.realSource(f.pr);
    entry.mode = '120000';
    await assert.rejects(source.directory(['boms']), /Non-directory/);
    f.github.rest.git.getTree = async () => ({data: {sha: 'root', truncated: true, tree: []}});
    source = await f.realSource(f.pr);
    await assert.rejects(source.directory(['boms']), /Incomplete/);
    const blob = {type: 'blob', mode: '100644', size: raw.length, sha};
    assert.deepEqual(await source.blob(blob), raw);
    await assert.rejects(source.blob({...blob, sha: 'bad'}), /identity mismatch/);
    await assert.rejects(source.blob({...blob, mode: '120000'}), /Invalid source blob/);
});

test('latest job attempts take precedence over job IDs and prior failures', () => {
    const jobs = latestJobs([
        {name: 'Build', id: 100, run_attempt: 1, conclusion: 'success'},
        {name: 'Build', id: 10, run_attempt: 2, conclusion: 'failure'},
    ]);
    assert.equal(jobs[0].conclusion, 'failure');
});

function storedZip(filename, contents) {
    const data = Buffer.from(contents);
    const name = Buffer.from(filename);
    let crc = 0xffffffff;
    for (const byte of data) {
        crc ^= byte;
        for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
    }
    crc = (crc ^ 0xffffffff) >>> 0;
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50);
    local.writeUInt16LE(20, 4);
    local.writeUInt32LE(crc, 14);
    local.writeUInt32LE(data.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(name.length, 26);
    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50);
    central.writeUInt16LE(20, 4);
    central.writeUInt16LE(20, 6);
    central.writeUInt32LE(crc, 16);
    central.writeUInt32LE(data.length, 20);
    central.writeUInt32LE(data.length, 24);
    central.writeUInt16LE(name.length, 28);
    const end = Buffer.alloc(22);
    end.writeUInt32LE(0x06054b50);
    end.writeUInt16LE(1, 8);
    end.writeUInt16LE(1, 10);
    end.writeUInt32LE(central.length + name.length, 12);
    end.writeUInt32LE(local.length + name.length + data.length, 16);
    return Buffer.concat([local, name, data, central, name, end]);
}

test('real orchestration plus strict ZIP engine produces a combined patch applicable to exact PR GOLD', async () => {
    const f = fixture();
    f.aggregator.runEngine = engine;
    f.archive = artifactId => {
        const runId = artifactId - 1000;
        const [file] = [...f.runs].find(([, runs]) => runs[0].id === runId);
        return storedZip(`${file.startsWith('windows_') ? 'windows' : 'linux'}/core.json`,
            '{"component":"core","files":["new"]}\n');
    };
    await f.prepare();
    assert.equal(f.outputs.upload, 'true');
    const filename = path.resolve('bom-gold-output/pr-7-gold.patch');
    const patch = fs.readFileSync(filename, 'utf8');
    const paths = [...patch.matchAll(/^--- a\/(boms\/[A-Za-z0-9_]+\/core.json)$/gm)].map(match => match[1]);
    assert.equal(paths.length, 7);
    const repo = path.resolve('apply-fixture');
    fs.mkdirSync(repo);
    try {
        execFileSync('git', ['init', '--quiet'], {cwd: repo});
        execFileSync('git', ['config', 'core.autocrlf', 'false'], {cwd: repo});
        for (const filename of paths) {
            const destination = path.join(repo, filename);
            fs.mkdirSync(path.dirname(destination), {recursive: true});
            fs.writeFileSync(destination, '{"component":"core","files":["old"]}');
        }
        execFileSync('git', ['apply', '--check', filename], {cwd: repo});
        execFileSync('git', ['apply', filename], {cwd: repo});
        for (const filename of paths) {
            assert.deepEqual(JSON.parse(fs.readFileSync(path.join(repo, filename), 'utf8')).files, ['new']);
        }
        await f.publish('901');
        assert.match(f.body(), /Download patch artifact/);
    } finally {
        fs.rmSync(repo, {recursive: true, force: true});
    }
});

test('newly ambiguous fork PR bases also block final publication', async () => {
    const f = fixture();
    await f.prepare();
    f.prs.push({...f.pr, number: 8});
    await assert.rejects(f.publish('901'), /association changed or became ambiguous/);
    assert.doesNotMatch(f.body(), /Download patch artifact/);
});

test('fork runs without PR associations cannot be reused after a PR base retarget', async () => {
    const f = fixture();
    f.retargetedAt = '2026-01-02T00:00:00Z';
    await f.prepare();
    assert.equal(f.outputs.upload, 'false');
    assert.equal(f.downloads, 0);
    assert.match(f.body(), /predates this PR\/base selection/);
    // Rerunning the old run keeps its original base event context.
    for (const runs of f.runs.values()) {
        runs[0].run_attempt = 2;
        runs[0].run_started_at = '2026-01-03T00:00:00Z';
    }
    await f.prepare();
    assert.equal(f.outputs.upload, 'false');
});

test('runs from before the PR existed and unavailable base history cannot supply a patch', async () => {
    const f = fixture();
    f.pr.created_at = '2026-01-02T00:00:00Z';
    await f.prepare();
    assert.equal(f.outputs.upload, 'false');
    f.graphqlError = true;
    await assert.rejects(f.prepare(), /GraphQL/);
    assert.match(f.body(), /Blocked/);
});
