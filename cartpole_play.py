import gym
from gym.utils import play
import numpy as np
import sys
import keras

# Allows for loading and running a pre-trained model,
# or for the user to interact with the environment.
# See 'cartpole_train' for more details.

USER_PLAY = False
MODEL_DIR = "saved_models/trained_model/model"

class QAgent:
    def __init__(self, action_size, state_size):
        self.action_size = action_size
        self.state_size = state_size
        self.value_function = self.__init_model()
        self.value_function.load_weights(MODEL_DIR)

    def __init_model(self):
        inputs = keras.Input(shape=(self.state_size,))
        x = keras.layers.Dense(24, activation='relu')(inputs)
        x = keras.layers.Dense(24, activation='relu')(x)
        outputs = keras.layers.Dense(self.action_size, activation='linear')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mse', optimizer='Adam')

        return model

    def get_action(self, state):
        # Get action from already trained network
        return np.argmax(self.value_function(state)[0])

if __name__ == "__main__":

    env = gym.make("CartPole-v1")

    if USER_PLAY:
        env.reset()
        key_action_map = {(ord('a'),): 0, (ord('d'),): 1}  # Map 'a' and 'd' to going left and right
        play.play(env, keys_to_action=key_action_map, fps=15)  # Lower fps cause its too hard otherwise
        sys.exit()

    # 0: Left, 1: Right
    action_size = env.action_space.n
    # 0: Cart position, 1: Cart Velocity, 2: Pole Angle, 3: Pole Angular Velocity
    state_size = env.observation_space.shape[0]

    agent = QAgent(action_size, state_size)
    n_episodes = 40

    for e in range(n_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        done = False
        time_step = 0

        score = 0
        while not done and time_step < 500:
            env.render()

            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)

            score += reward
            reward = 0.1 if not done else -1
            next_state = np.reshape(next_state, [1, state_size])

            state = next_state
            time_step += 1

        print("episode: {}, time steps: {}, score: {}".format(e, time_step, score))
