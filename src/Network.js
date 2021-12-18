/* eslint-disable no-console */
const cliProgress = require('cli-progress');
const _ = require('lodash');

const Functions = require('./Functions');
const Matrix = require('./Matrix');
const Layer = require('./Layer');

class Network {
  constructor({
    structure,
    learningRate = 0.1,
    layers,
  }) {
    this.structure = structure;
    this.learningRate = learningRate;
    this.layers = [];
    this.layerCount = structure.length - 1;

    for (let i = 0; i < this.layerCount; i++) {
      this.layers[i] = new Layer(structure[i], structure[i + 1]);
    }

    if (layers) {
      for (let i = 0; i < this.layers.length; i++) {
        this.layers[i].updateWeights(layers[i].weights);
        this.layers[i].updateBias(layers[i].bias);
      }
    }
  }

  train(trainingData) {
    console.time('Training Time');
    const trainingProgressBar = new cliProgress.SingleBar(
      {}, cliProgress.Presets.shades_classic,
    );
    trainingProgressBar.start(trainingData.length, 0);

    for (let i = 0; i < trainingData.length; i++) {
      if (i % 1000 === 0) trainingProgressBar.update(i);
      this.backprop(trainingData[i].input, trainingData[i].output);
    }

    trainingProgressBar.update(trainingData.length);
    trainingProgressBar.stop();
    console.timeEnd('Training Time');
  }

  backprop(inputArray, targetArray) {
    const input = _.chunk(inputArray);
    const layerResult = [];
    layerResult[0] = input;
    for (let i = 0; i < this.layerCount; i++) {
      layerResult[i + 1] = Matrix.product(
        this.layers[i].getWeights(),
        layerResult[i],
      );
      layerResult[i + 1] = Matrix.add(
        layerResult[i + 1],
        this.layers[i].getBias(),
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

      this.layers[i - 1].addToWeights(weightDeltas);
      this.layers[i - 1].addToBias(gradients[i]);

      layerErrors[i - 1] = Matrix.product(
        Matrix.transpose(this.layers[i - 1].getWeights()), layerErrors[i],
      );
    }
  }

  predict(input) {
    let layerResult = _.chunk(input);
    for (let i = 0; i < this.layerCount; i++) {
      layerResult = Matrix.product(this.layers[i].getWeights(), layerResult);
      layerResult = Matrix.add(this.layers[i].getBias(), layerResult);
      layerResult = Matrix.map(layerResult, Functions.sigmoid);
    }
    return _.flatten(layerResult);
  }

  evaluate(testData) {
    const results = [];
    for (let i = 0; i < testData.length; i++) {
      const point = testData[i];
      const prediction = this.predict(point[0]);
      results.push([
        _.indexOf(prediction, _.max(prediction)),
        point[1],
      ]);
    }

    const matchCount = results.filter((arr) => arr[0] === arr[1]).length;
    console.log(`Accuracy: ${matchCount} / ${results.length}`);
  }

  save() {
    return {
      ...this,
    };
  }
}

module.exports.Network = Network;
