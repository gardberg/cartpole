import gym
import random
import numpy as np
import tensorflow as tf
import keras
from collections import deque
from keras.optimizers import Adam
import matplotlib.pyplot as plt

# This file consists of an implementation as well means for training
# a deep-Q network agent that interacts with the Cartpole-v1 environment.
# For details: https://gym.openai.com/envs/CartPole-v1/

# Inspiration and help has been obtained from https://github.com/rlcode/
# as well as the DeepMind paper https://arxiv.org/pdf/1312.5602.pdf


# Agent class for interacting with the environment
class QAgent:
    def __init__(self, action_size, state_size):
        self.action_size = action_size  # How many actions we have, 2
        self.state_size = state_size  # State dimensions, 4
        self.epsilon = 1.0  # Random action rate
        self.learning_rate = 0.001

        self.model = self.__init_model()  # Q value func based on neural network
        self.target_model = self.__init_model()  # Additional target model for more stable learning
        self.update_target_model()  # Copy weights of main model onto target

        self.min_epsilon = 0.01  # Lower bound of exploration
        self.epsilon_decay = 0.999
        self.gamma = 0.95  # Reward discount rate

        # For storing past experiences (samples)
        # Oldest sample is discarded when length exceeds 'maxlen'
        self.memory = deque(maxlen=2048)
        self.training_threshold = 1024  # Sample threshold for starting to use memory
        self.batch_size = 32  # How many samples to train on at a time


    def __init_model(self):
        # Neural network for approximating Q(s,a)
        # Input: array of state values
        # Output: estimated values for all actions (in this case two)
        init = tf.keras.initializers.RandomUniform(-1e-3, 1e-3)  # Define initialization of weights

        model = keras.models.Sequential()
        model.add(keras.Input(shape=(self.state_size,)))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear', kernel_initializer=init))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def get_action(self, state):
        # Get epsilon greedy action
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state)[0])


    def memorize(self, sample):
        self.memory.append(sample)


    def _split_batch(self, training_batch):
        # Separates a training batch into arrays of states, actions, etc.

        states = np.array([sample[0][0] for sample in training_batch])
        actions = np.array([sample[1] for sample in training_batch])
        rewards = np.array([sample[2] for sample in training_batch])
        next_states = np.array([sample[3][0] for sample in training_batch])
        dones = np.array([sample[4] for sample in training_batch])

        return states, actions, rewards, next_states, dones


    def training_step(self):
        # Performs one training step of our model

        training_batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = self._split_batch(training_batch)

        # Array of predicted q values for the next state
        target_qs = self.target_model.predict(next_states)

        # Our temporal difference targets
        targets = rewards + self.gamma * np.amax(target_qs, axis=-1) * (1 - dones)

        # Array of predicted q values for current state
        current_qs = self.model.predict(states)

        # Update the value for the action we have just performed

        # TF Tensors do not support item assignment, workaround
        # Performs the operation: target_output[0][action] = target, for each action
        mask = np.array(np.stack((actions, 1 - actions))).T
        mask = tf.constant(mask, dtype=float)
        current_qs = current_qs * mask + np.stack((targets, targets)).T * (1 - mask)

        self.model.train_on_batch(x=states, y=current_qs.numpy())

        # Decaying exploration
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay


# Main method
# Runs the training of the model
if __name__ == "__main__":
    RENDER = False
    env = gym.make("CartPole-v1")

    # 0: Left, 1: Right
    action_size = env.action_space.n

    # 0: Cart position, 1: Cart Velocity, 2: Pole Angle, 3: Pole Angular Velocity
    state_size = env.observation_space.shape[0]
    agent = QAgent(action_size, state_size)
    n_episodes = 1000
    average_scores = []
    avg_score = 0

    for e in range(n_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])  # Reshape into Keras format
        done = False
        time_step = 0
        score = 0  # Score for evaluation of agent

        while not done and time_step < 500:
            if RENDER:
                env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)  # Environment interaction

            score += reward
            reward = 0.1 if not done else -1  # Negative reward if the game has ended

            next_state = np.reshape(next_state, [1, state_size])  # Convert to "tensor form"
            sample = (state, action, reward, next_state, done)

            agent.memorize(sample)  # Store the sample in memory

            # In order to avoid the problem with correlating samples,
            # we don't train until we have enough samples in our memory to randomize from
            if len(agent.memory) >= agent.training_threshold:
                agent.training_step()

            state = next_state
            time_step += 1

        agent.update_target_model()
        avg_score = 0.9 * avg_score + 0.1 * score if avg_score != 0 else score
        average_scores.append(avg_score)

        print('ep {}/{}, avg score: {:3.2f}, epsilon: {:.2}, steps done: {}, mem length: {}'
              .format(e, n_episodes, avg_score, agent.epsilon, time_step, len(agent.memory)))

    agent.model.save_weights("./saved_models/model", save_format="tf")

    # Plot of average scores
    plt.plot(average_scores)
    plt.xlabel('episode')
    plt.ylabel('average score')
    plt.show()
