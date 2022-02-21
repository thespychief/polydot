const EventEmitter = require('events');
const _ = require('lodash');

const Functions = require('./Functions');
const Matrix = require('./Matrix');
const Segue = require('./Segue');

class Network {
  constructor({
    structure,
    segues,
    learningRate = 0.1,
    normalization = {
      method: 'None',
    },
  }) {
    this.structure = structure;
    this.segues = [];
    this.layerCount = structure.length - 1;

    this.learningRate = learningRate;
    this.normalization = normalization;

    for (let i = 0; i < this.layerCount; i++) {
      this.segues[i] = new Segue(structure[i], structure[i + 1]);
    }

    if (segues) {
      for (let i = 0; i < this.segues.length; i++) {
        this.segues[i].updateWeights(segues[i].weights);
        this.segues[i].updateBias(segues[i].bias);
      }
    }

    this.eventEmitter = new EventEmitter();
  }

  normalizeInput(input) {
    let normalizedInput = input;

    if (this.normalization.method === 'normalizeByMax') {
      normalizedInput = Functions.normalizeByMax(input);
    }

    if (this.normalization.method === 'normalizeByConstant') {
      normalizedInput = Functions.normalizeByConstant(
        input, this.normalization.constant,
      );
    }

    return normalizedInput;
  }

  normalizeDataset(data) {
    const normalizedData = data;

    if (this.normalization.method === 'normalizeByMax') {
      for (let i = 0; i < data.length; i++) {
        normalizedData[i].input = Functions.normalizeByMax(
          data[i].input,
        );
      }
    }

    if (this.normalization.method === 'normalizeByConstant') {
      for (let i = 0; i < data.length; i++) {
        normalizedData[i].input = Functions.normalizeByConstant(
          data[i].input, this.normalization.constant,
        );
      }
    }

    return normalizedData;
  }

  train(trainingData) {
    const normalizedData = this.normalizeDataset(trainingData);

    for (let i = 0; i < normalizedData.length; i++) {
      if (i % 1000 === 0) {
        this.eventEmitter.emit('event', {
          iteration: i,
          total: normalizedData.length,
        });
      }

      this.backprop(normalizedData[i].input, normalizedData[i].output);
    }

    this.eventEmitter.emit('event', {
      iteration: normalizedData.length,
      total: normalizedData.length,
    });
  }

  backprop(inputArray, targetArray) {
    const input = _.chunk(inputArray);
    const layerResult = [];
    layerResult[0] = input;
    for (let i = 0; i < this.layerCount; i++) {
      layerResult[i + 1] = Matrix.product(
        this.segues[i].getWeights(),
        layerResult[i],
      );
      layerResult[i + 1] = Matrix.add(
        layerResult[i + 1],
        this.segues[i].getBias(),
      );
      layerResult[i + 1] = Matrix.map(
        layerResult[i + 1],
        Functions.sigmoid,
      );
    }

    const targets = _.chunk(targetArray);
    const layerErrors = [];
    layerErrors[this.layerCount] = Matrix.subtract(
      targets, layerResult[this.layerCount],
    );

    const gradients = [];
    for (let i = this.layerCount; i > 0; i--) {
      gradients[i] = Matrix.map(layerResult[i], Functions.derivative);
      gradients[i] = Matrix.hadamardProduct(gradients[i], layerErrors[i]);
      gradients[i] = Matrix.map(gradients[i], (x) => x * this.learningRate);

      const hiddenTranspose = Matrix.transpose(layerResult[i - 1]);
      const weightDeltas = Matrix.product(gradients[i], hiddenTranspose);

      this.segues[i - 1].addToWeights(weightDeltas);
      this.segues[i - 1].addToBias(gradients[i]);

      layerErrors[i - 1] = Matrix.product(
        Matrix.transpose(this.segues[i - 1].getWeights()), layerErrors[i],
      );
    }
  }

  predict(input) {
    const normalizedInput = this.normalizeInput(input);

    let layerResult = _.chunk(normalizedInput);
    for (let i = 0; i < this.layerCount; i++) {
      layerResult = Matrix.product(this.segues[i].getWeights(), layerResult);
      layerResult = Matrix.add(this.segues[i].getBias(), layerResult);
      layerResult = Matrix.map(layerResult, Functions.sigmoid);
    }

    return _.flatten(layerResult);
  }

  evaluate(testData) {
    const normalizedData = this.normalizeDataset(testData);

    const results = [];
    for (let i = 0; i < normalizedData.length; i++) {
      const point = normalizedData[i];
      const prediction = this.predict(point[0]);
      results.push([
        _.indexOf(prediction, _.max(prediction)),
        point[1],
      ]);
    }

    const matchCount = results.filter((arr) => arr[0] === arr[1]).length;
    return {
      matches: matchCount,
      accuracy: matchCount / results.length,
    };
  }

  save() {
    return {
      ...this,
    };
  }
}

module.exports.Network = Network;
