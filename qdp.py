import numpy as np
import nn
import logging
from absl import app
from absl import flags
from rlboard import *

FLAGS = flags.FLAGS

flags.DEFINE_string('loadfile', '', 'The file to read weights from. If not given, use random weights')
flags.DEFINE_string('savefile', '', 'The file to store the network weights. If not given, nothing is saved')
flags.DEFINE_float('learningrate', 0.01, 'The learning rate')
flags.DEFINE_boolean('train', False, 'Whether to train the model or only evaluate')
flags.DEFINE_integer('maxruns', 1000, 'The number of runs to execute.')
flags.DEFINE_integer('reportingBatchSize', 1000, 'The number of evaluations after which to report.')
flags.DEFINE_string('loglevel', 'INFO', 'logging level')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Network size
field_size = 8
D = field_size * field_size
H = 200  # hidden neurons
A = 4 # 4 possible actions


# Connect to game and return observation
class Game:
    def __init__(self):
        self.m = Board(8,8)
        self.m.randomize()
        self.actions = {'L':(-1,0), 'R':(1,0), 'D':(0,-1), 'U':(0,1)}

    def observe(self):
        return self.m.matrix.flatten()
    
    def act(self, action):
        self.m.move(action[1], check_correctness=False)

    def reward(self):
        if not self.m.is_valid(self.m.human):
            return -10
        if self.m.at() in [self.m.Cell.water]:
            return -10
        if self.m.at() in [self.m.Cell.wolf]:
            # TODO reintroduce energy & fatigue to observation and reward
            # if self.m.energy > self.m.fatigue:
            #     return 10
            return -10
        return -0.1

class DqlEvaluator:
    def __init__(self, optimizer = None):
        self.optimizer = optimizer

    """Forward-evaluates a neural network."""
    def EvalLoop(self, nn, maxruns, game, reportingBatchSize = 100):
        total_reward = 0
        path_length = 0
        gamma = 0.9

        for count in range(maxruns):
            state = game.observe()
            output = self.Evaluate(nn, state)
            action = probstrategy(game, output)
            game.act(action)
            state_after = game.observe()

            r = game.reward()

            total_reward += r
            path_length += 1

            # Compute the temporal difference update
            if r < - 5:
                # terminal state
                update = r
            else:
                # non-terminal state
                update = r + gamma * self.Evaluate(nn, state_after).max()
            loss = output * r
            # Fake a 'target': in supervised learning, we'd have a target
            # here, we fake a target taking the sampled action / output and
            # multiplying it by the advantage given by reward (should it be by total reward?)
            target = loss + output

            if self.optimizer:
                self.optimizer.Optimize(nn, output, target)
            
            if r < -5 or total_reward < -100 or r > 5:
                break

        if total_reward > 0:
            logger.info(f"Eval loop: path_length: {path_length}, total_reward: {total_reward}")
        else:
            logger.debug(f"Eval loop: path_length: {path_length}, total_reward: {total_reward}")
    
    def Evaluate(self, nn, input):
        state = input
        for layer in nn.layers:
            state = layer.Evaluate(state)
        return state

def normalize(v,eps=1e-4):
    w = v.copy()
    v = v-v.min()+eps
    v = v/v.sum()
    if not np.isfinite(v).any():
        print(w)
        print(v)
    return v

def probstrategy(game, weights):
    weights = normalize(weights)
    action = random.choices(list(game.actions.items()), weights=weights)
    return action[0]


def main(argv):
    logger.setLevel(FLAGS.loglevel)
    # Initialize NN Model serving as Q estimator
    if FLAGS.loadfile:
        Q = nn.NN.LoadFromFile(FLAGS.loadfile)
    else:
        Q = nn.NN.WithRandomWeights([D, H, A])

    for episode in range(1000):
        game = Game()
        
        if FLAGS.train:
            evaluator = DqlEvaluator(nn.GradientDescentOptimizer(FLAGS.learningrate))
        else:
            evaluator = DqlEvaluator()
    
        evaluator.EvalLoop(Q, FLAGS.maxruns, game, FLAGS.reportingBatchSize)

    if FLAGS.savefile:
        Q.Store(FLAGS.savefile)

if __name__ == '__main__':
  app.run(main)

