const utils = require('./utils');

class OpenPoseDecoder {
  constructor(
    numJoints = 18,
    skeleton = [
      [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
      [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17],
      [2, 16], [5, 17],
    ],
    pafIndices = [12, 20, 14, 16, 22, 24, 0, 2, 4, 6, 8, 10, 28, 30, 34, 32, 36,
      18, 26],
    maxPoints = 100,
    scoreThreshold = 0.1,
    minPafAlignmentScore = 0.05,
    delta = 0.5,
  ) {
    this.numJoints = numJoints;
    this.skeleton = skeleton;
    this.pafIndices = pafIndices;
    this.maxPoints = maxPoints;
    this.scoreThreshold = scoreThreshold;
    this.minPafAlignmentScore = minPafAlignmentScore;
    this.delta = delta;

    this.pointsPerLimb = 10;
    this.grid = utils.range(this.pointsPerLimb, 0, 1, 'float32')
      .reshape([1, -1, 1]);
  }

  async call(heatmaps, nmsHeatmaps, pafs) {
    const [batchSize, , h, w] = heatmaps.shape;
    if (batchSize !== 1) {
      throw new Error('Batch size of 1 only supported');
    }

    const keypoints = await this.extractPoints(heatmaps, nmsHeatmaps);
    pafs = pafs.transpose([0, 2, 3, 1]);

    if (this.delta > 0) {
      keypoints.forEach(kpts => {
        kpts.forEach(kpt => {
          kpt[0] = Math.min(Math.max(kpt[0] + this.delta, 0), w - 1);
          kpt[1] = Math.min(Math.max(kpt[1] + this.delta, 0), h - 1);
        });
      });
    }

    const [poseEntries, keypointsArray] =
      await this.groupKeypoints(keypoints, pafs, this.numJoints + 2);
    let [poses, scores] = this.convertToCocoFormat(poseEntries, keypointsArray);

    if (poses.length > 0) {
      poses = utils.tensor(poses, [poses.length, -1, 3], 'float32');
    } else {
      poses = utils.tensor([], [0, 17, 3], 'float32');
      scores = utils.tensor([], [0], 'float32');
    }

    return [poses, scores];
  }

  async extractPoints(heatmaps, nmsHeatmaps) {
    const [batchSize, channelsNum, h, w] = heatmaps.shape;
    if (batchSize !== 1 || channelsNum < this.numJoints) {
      throw new Error('Invalid input dimensions');
    }

    const [xs, ys, scores] = await this.topK(nmsHeatmaps);
    const masks = scores.greater(this.scoreThreshold);
    let allKeypoints = [];
    let keypointId = 0;

    for (let k = 0; k < this.numJoints; k++) {
      const mask = masks.slice([0, k], [1, 1]).reshape([-1]);
      const x = utils.booleanMaskAsync(xs.slice([0, k], [1, -1])
        .reshape([-1]), mask);
      const y = utils.booleanMaskAsync(ys.slice([0, k], [1, -1])
        .reshape([-1]), mask);
      const score = utils.booleanMaskAsync(scores.slice([0, k], [1, -1])
        .reshape([-1]), mask);
      const n = (await x.array()).length;

      if (n === 0) {
        allKeypoints.push(utils.tensor([], [0, 4], 'float32'));
        continue;
      }

      const [refinedX, refinedY] = await this.refine(heatmaps.slice([0, k],
        [1, -1, -1]).reshape([h, w]), x, y);
      const clippedX = refinedX.clipByValue(0, w - 1);
      const clippedY = refinedY.clipByValue(0, h - 1);

      const keypoints = utils.stack([clippedX, clippedY, score, utils.arange(
        keypointId, keypointId + n, 1)], 1);
      keypointId += n;
      allKeypoints.push(keypoints);
    }

    return allKeypoints;
  }

  async topK(heatmaps) {
    const [N, K, _, W] = heatmaps.shape;
    const reshaped = heatmaps.reshape([N, K, -1]);
    const [scores, ind] = utils.topk(reshaped, this.maxPoints);

    const subind = utils.argsort(scores.neg(), 2);
    const sortedInd = utils.gather(ind, subind, 2);
    const sortedScores = utils.gather(scores, subind, 2);

    const [y, x] = utils.split(utils.div(sortedInd, W).reshape([-1]), 2, 1);

    return [
      x.reshape([N, K, this.maxPoints]), y.reshape([N, K, this.maxPoints]),
      sortedScores,
    ];
  }

  static async refine(heatmap, x, y) {
    const [h, w] = heatmap.shape;
    const valid = x.greater(0).logicalAnd(x.less(w - 1))
      .logicalAnd(y.greater(0)).logicalAnd(y.less(h - 1));

    const validX = x.booleanMaskAsync(valid);
    const validY = y.booleanMaskAsync(valid);
    const dx = (heatmap.gather([validY, validX.add(1)])
      .sub(heatmap.gather([validY, validX.sub(1)]))).sign().mul(0.25);
    const dy = (heatmap.gather([validY.add(1), validX])
      .sub(heatmap.gather([validY.sub(1), validX]))).sign().mul(0.25);

    return [x.add(dx), y.add(dy)];
  }

  static isDisjoint(poseA, poseB) {
    poseA = poseA.slice(0, -2);
    poseB = poseB.slice(0, -2);

    return poseA.equal(poseB).logicalOr(poseA.less(0))
      .logicalOr(poseB.less(0)).all();
  }

  async updatePoses(kptAId, kptBId, allKeypoints, connections, poseEntries,
    poseEntrySize) {
    for (const connection of connections) {
      let poseAIdx = -1;
      let poseBIdx = -1;
      poseEntries.forEach((pose, j) => {
        if (pose[kptAId] === connection[0]) poseAIdx = j;
        if (pose[kptBId] === connection[1]) poseBIdx = j;
      });

      if (poseAIdx < 0 && poseBIdx < 0) {
        const poseEntry = utils.fill([poseEntrySize], -1, 'float32');
        poseEntry[kptAId] = connection[0];
        poseEntry[kptBId] = connection[1];
        poseEntry[poseEntrySize - 1] = 2;
        poseEntry[poseEntrySize - 2] = utils.sum(
          utils.gather(
            allKeypoints,
            [connection[0], connection[1]],
            0,
          ).slice([0, 2], [-1, 1]),
        ).add(connection[2]);
        poseEntries.push(poseEntry);
      } else if (poseAIdx >= 0 && poseBIdx >= 0 && poseAIdx !== poseBIdx) {
        const poseA = poseEntries[poseAIdx];
        const poseB = poseEntries[poseBIdx];
        if (await OpenPoseDecoder.isDisjoint(poseA, poseB)) {
          poseA.add(poseB);
          poseA.slice(0, -2).add(1);
          poseA[poseEntrySize - 2] =
            poseA[poseEntrySize - 2].add(connection[2]);
          poseEntries.splice(poseBIdx, 1);
        }
      } else if (poseAIdx >= 0) {
        const pose = poseEntries[poseAIdx];

        if (pose[kptBId] < 0)
          pose[poseEntrySize - 2] =
            pose[poseEntrySize - 2].add(allKeypoints[connection[1], 2]);

        pose[kptBId] = connection[1];
        pose[poseEntrySize - 2] = pose[poseEntrySize - 2].add(connection[2]);
        pose[poseEntrySize - 1] = pose[poseEntrySize - 1].add(1);
      } else if (poseBIdx >= 0) {
        const pose = poseEntries[poseBIdx];

        if (pose[kptAId] < 0)
          pose[poseEntrySize - 2] =
            pose[poseEntrySize - 2].add(allKeypoints[connection[0], 2]);

        pose[kptAId] = connection[0];
        pose[poseEntrySize - 2] = pose[poseEntrySize - 2].add(connection[2]);
        pose[poseEntrySize - 1] = pose[poseEntrySize - 1].add(1);
      }
    }

    return poseEntries;
  }

  async groupKeypoints(allKeypointsByType, pafs, poseEntrySize = 20) {
    const allKeypoints = utils.concat(allKeypointsByType, 0);
    let poseEntries = [];

    for (const [partId, pafChannel] of this.pafIndices.entries()) {
      const [kptAId, kptBId] = this.skeleton[partId];
      const kptsA = allKeypointsByType[kptAId];
      const kptsB = allKeypointsByType[kptBId];
      const n = kptsA.shape[0];
      const m = kptsB.shape[0];

      if (n === 0 || m === 0) continue;

      const a =
        utils.tile(kptsA.slice([0, 0], [n, 2]).reshape([1, n, 2]), [m, 1, 1]);
      const b = kptsB.slice([0, 0], [m, 2]);
      const vecRaw = b.reshape([m, 1, 2]).sub(a).reshape([-1, 1, 2]);

      const steps = vecRaw.div(this.pointsPerLimb - 1);
      const points =
        steps.mul(this.grid).add(a.reshape([-1, 1, 2])).round().cast('int32');
      const x = points.slice([0, 0, 0], [-1, -1, 1]).reshape([-1]);
      const y = points.slice([0, 0, 1], [-1, -1, 1]).reshape([-1]);

      const partPafs =
        pafs.slice([0, 0, 0, pafChannel], [1, -1, -1, 2]).reshape([h, w, 2]);
      const field =
        utils.gather(partPafs, y, 0).reshape([-1, this.pointsPerLimb, 2]);
      const vecNorm = vecRaw.norm('euclidean', -1).reshape([-1, 1, 1]);
      const vec = vecRaw.div(vecNorm.add(1e-6));
      const affinityScores1 =
        utils.sum(field.mul(vec), -1).reshape([-1, this.pointsPerLimb]);
      const validAffinityScores =
        affinityScores1.greater(this.minPafAlignmentScore);
      const validNum = utils.sum(validAffinityScores, 1);
      const affinityScoresSum =
        utils.sum(affinityScores1.mul(validAffinityScores), 1);
      const affinityScores2 = affinityScoresSum.div(validNum.add(1e-6));
      const successRatio = validNum.div(this.pointsPerLimb);

      const validLimbs = affinityScores2.greater(0)
        .logicalAnd(successRatio.greater(0.8)).toBool().where().flatten();

      if (validLimbs.shape[0] === 0) continue;

      const validIdx = utils.div(validLimbs, n).reshape([-1]);
      const aIdx = validLimbs.mod(n).reshape([-1]);
      const affinityScores3 = utils.gather(affinityScores2, validLimbs);

      const connections =
        await this.connectionsNms(aIdx, validIdx, affinityScores3);

      if (connections.length === 0) continue;

      poseEntries = await this.updatePoses(kptAId, kptBId, allKeypoints,
        connections, poseEntries, poseEntrySize);
    }

    poseEntries =
      utils.tensor(poseEntries, [poseEntries.length, poseEntrySize], 'float32');
    poseEntries = poseEntries.where(
      poseEntries.slice([0, poseEntrySize - 1], [-1, 1]).greaterEqual(3));

    return [poseEntries, allKeypoints];
  }

  static async connectionsNms(aIdx, bIdx, affinityScores) {
    const order = affinityScores.argsort().reverse();
    const sortedAffinityScores = utils.gather(affinityScores, order);
    const sortedAIdx = utils.gather(aIdx, order);
    const sortedBIdx = utils.gather(bIdx, order);

    const idx = [];
    const hasKptA = new Set();
    const hasKptB = new Set();

    for (let t = 0; t < sortedAIdx.shape[0]; t++) {
      const i = sortedAIdx.arraySync()[t];
      const j = sortedBIdx.arraySync()[t];

      if (!hasKptA.has(i) && !hasKptB.has(j)) {
        idx.push(t);
        hasKptA.add(i);
        hasKptB.add(j);
      }
    }

    const finalIdx = utils.tensor1d(idx, 'int32');

    return [
      utils.gather(sortedAIdx, finalIdx),
      utils.gather(sortedBIdx, finalIdx),
      utils.gather(sortedAffinityScores, finalIdx),
    ];
  }

  convertToCocoFormat(poseEntries, allKeypoints) {
    const numJoints = 17;
    const cocoKeypoints = [];
    const scores = [];
    for (const pose of poseEntries.arraySync()) {
      if (pose.length === 0) continue;

      const keypoints = Array(numJoints * 3).fill(0);
      const reorderMap =
        [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3];
      const personScore = pose[pose.length - 2];

      for (let i = 0; i < pose.length - 2; i++) {
        const targetId = reorderMap[i];

        if (targetId < 0) continue;

        let [cx, cy, score] = [0, 0, 0];
        if (pose[i] !== -1) {
          [cx, cy, score] = allKeypoints.arraySync()[pose[i]];
        }

        keypoints[targetId * 3 + 0] = cx;
        keypoints[targetId * 3 + 1] = cy;
        keypoints[targetId * 3 + 2] = score;
      }

      cocoKeypoints.push(keypoints);
      scores.push(personScore * Math.max(0, (pose[pose.length - 1] - 1)));
    }

    return [cocoKeypoints, scores];
  }
}

module.exports = OpenPoseDecoder;
