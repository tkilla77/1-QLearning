"""
Usage:

Training: train a new network from scratch

$ python3 01_toy_problem_feedforward.py --dim 4 --dim 3 --dim 2 \
    --savefile toy_network.nn.npz \
    --datafile data_toy_problem/data_dark_bright_training_20000.csv \
    --train --learningrate 0.01 --maxruns 20000

Eval:
$ python3 01_toy_problem_feedforward.py \
    --loadfile toy_network.nn.npz \
    --datafile data_toy_problem/data_dark_bright_test_4000.csv \
    --maxruns 4000

"""
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def sigmoid(input):
    """The sigmoid function."""
    return 1 / (1 + np.exp(-input))

class Layer:
    """One layer in a neural network.
    It carries its neurons' input state and its input weights.
    """
    def __init__(self, weights, state, activation=sigmoid):
        self.weights = weights
        self.state = state
        self.activation = activation
        assert weights.shape[1] == state.shape[0], "Expected compatible size: %s/%s" % (weights.shape, state.shape)
    
    def __str__(self) -> str:
        return "FC: %s" % str(self.weights.shape[::-1])

    def Evaluate(self, input):
        logger.debug("Input state: %s", input)
        self.state = input
        logger.debug("Weights: %s", self.weights)
        state = np.dot(self.weights, input)
        logger.debug("Output state after weights: %s", state)
        state = self.activation(state)
        logger.debug("Output state after activation: %s", state)
        return state

class NN:
    """A neural network."""
    def __init__(self, layers):
        self.activation = sigmoid
        self.layers = layers
        logger.info("New NN with shape: %s", [str(l) for l in self.layers])

    @staticmethod
    def WithRandomWeights(dimensions):
        lastDim = dimensions[0]
        layers = []
        for dimension in dimensions[1:]:
            layers.append(Layer(np.random.rand(dimension, lastDim) / 2, np.zeros(lastDim)))
            lastDim = dimension
        return NN(layers)

    @staticmethod
    def WithGivenWeights(weights):
        layers = []
        for newweights in weights:
            layers.append(Layer(newweights, np.zeros(newweights.shape[1])))
        return NN(layers)

    @staticmethod
    def LoadFromFile(file):
        """Loads a NN from file."""
        npzfile = np.load(file)
        layers = []
        for weights in sorted(npzfile.files):
            logger.debug("loading layer %s, %s", weights, npzfile[weights])
            layers.append(npzfile[weights])
        return NN.WithGivenWeights(layers)

    def Store(self, file):
        """Stores a NN to file."""
        arrays = [layer.weights for layer in self.layers]
        np.savez(file, *arrays)        


class ForwardEvaluator:
    def __init__(self, optimizer = None):
        self.optimizer = optimizer

    """Forward-evaluates a neural network."""
    def EvalLoop(self, nn, maxruns, input, reportingBatchSize = 100):
        count = 0
        correct = 0
        batchCorrect = 0
        batchError = 0.0

        for line in input:
            if count == maxruns:
                break

            count += 1
            target = line['target']
            input = line['input']
            output = self.Evaluate(nn, input)

            # Target value for classification problem is a one hot vector.
            targetVector = np.zeros(output.size)
            targetVector[target] = 1

            # Choose an arbitrary index with the highest activation as the output.
            outputScalar = np.where(output == max(output))[0][0]

            if target == outputScalar:
                correct += 1
            if self.optimizer:
                batchError += self.optimizer.Optimize(nn, output, targetVector)
            
            # Report a few quality stats each batch.
            if count % reportingBatchSize == 0:
                batchSuccess = correct - batchCorrect
                batchRate = batchSuccess * 100.0 / reportingBatchSize
                overallRate = correct * 100.0 / count
                avgError = batchError / reportingBatchSize
                logger.info("Batch (%d): Avg error / Success rate / Overall: %.3f / %.1f%% / %.1f%%", count / reportingBatchSize, avgError, batchRate, overallRate)
                batchError = 0.0
                batchCorrect = correct

        logger.info("Success rate: %d of %d (%f%%)", correct, count, correct * 100.0 / count)

    def Evaluate(self, nn, input):
        state = input
        for layer in nn.layers:
            state = layer.Evaluate(state)
        return state

class GradientDescentOptimizer:
    """Implements gradient descent and changes weights in the NN."""
    def __init__(self, learning_rate = 0.1):
        self.learning_rate = learning_rate

    def Cost(self, output, target):
        return target - output

    def Optimize(self, nn, output, target):
        error = self.Cost(output, target)
        output_error = np.linalg.norm(error)
        
        for layer in reversed(nn.layers):
            logger.debug("Error is: %s", error)
            # Store the error we will forward to the next layer.
            next_error = np.dot(layer.weights.T, error)
            # The inner term of the gradient.
            term = np.atleast_2d(error * output * (1-output)).T
            logger.debug("Gradient term shape: %s", term.shape)
            state_T = np.atleast_2d(layer.state)
            logger.debug("State shape is: %s", state_T.shape)
            gradient = -1 * np.dot(term, state_T)
            logger.debug("Gradient is: %s", gradient.shape)
            logger.debug("Increment is: %s", self.learning_rate * gradient)
            layer.weights = layer.weights - self.learning_rate * gradient
            output = layer.state
            error = next_error

        return output_error

def readCsvLines255(filename):
    """
    Reads CSV lines and divides input values by 255.
    Returns a dictionary with entries 'target' and 'input'.
    """
    for row in open(filename, "r"):
        split = row.split(",")
        target = int(split[0])
        input =  np.asfarray(split[1:]) / 255
        yield {'target': target, 'input': input}
