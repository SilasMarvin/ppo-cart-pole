import tensorflow as tf
import numpy as np
import random
import gym
import sys


EPISODES = 100000
GAMMA = 0.99
CLIP_PARAM = 0.2
LMDA = 0.95


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.shared1 = tf.keras.layers.Dense(128, input_shape=(None,4), activation='relu')
        self.shared2 = tf.keras.layers.Dense(64, activation='relu')
        self.critic1 = tf.keras.layers.Dense(32, activation='relu')
        self.critic2 = tf.keras.layers.Dense(1, activation=None)
        self.actor1 = tf.keras.layers.Dense(32, activation='relu')
        self.actor2 = tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=7e-3, epsilon=1e-5)

    def call(self, inputs):
        shared = self.shared2(self.shared1(inputs))
        return (self.actor2(self.actor1(shared)), self.critic2(self.critic1(shared)))

    def actor_loss_func(self, probs, actions, advs, old_probs):
        actions = [[i, actions[i]] for i in range(len(actions))]
        probs = tf.gather_nd(probs, actions)
        old_probs = tf.gather_nd(old_probs, actions)
        ratio = tf.math.exp(tf.negative(tf.math.log(old_probs + 1e-10)) - tf.math.negative(tf.math.log(probs + 1e-10)))
        pg_losses = -advs * ratio
        pg_losses2 = -advs * tf.clip_by_value(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) 
        return tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))

    def critic_loss_func(self, preds, old_preds, returns):
        preds_clipped = old_preds + tf.clip_by_value(preds - old_preds, -CLIP_PARAM, CLIP_PARAM)
        vf_losses = tf.square(preds - returns)
        vf_losses2 = tf.square(preds_clipped - returns)
        vf_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_losses, vf_losses2))
        return vf_loss

    def entropy_func(self, probs):
        return -tf.reduce_sum(probs * tf.math.log(probs + 1e-10))

    def train(self, states, actions, old_preds, advs, old_probs, returns):
        states = tf.reshape(states, (len(states),4))
        discounted_rewards = tf.reshape(returns, (len(returns),1))
        old_probs = tf.reshape(old_probs, (len(probs),2))
        old_preds = tf.reshape(old_preds, (len(old_preds),))
        advs = tf.reshape(advs, (len(advs),))

        with tf.GradientTape() as tape:
            p, v = self(states)
            critic_loss = self.critic_loss_func(v, old_preds, returns)
            entropy = self.entropy_func(p)
            actor_loss = self.actor_loss_func(p, actions, advs, old_probs)
            loss_value = actor_loss + (0.5 * critic_loss) + (0.01 * entropy)
            grads = tape.gradient(loss_value, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss_value


def get_returns_advantages(rewards, dones, values):
    g = 0
    returns = []
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + GAMMA * values[i+1] * (1- dones[i]) - values[i]
        g = delta + GAMMA * LMDA * (1- dones[i]) * g
        returns.append(g + values[i])
    returns.reverse()

    advs = np.array(returns, dtype=np.float32) - values[:-1]
    advs = (advs - np.mean(advs)) / (np.std(advs) + 1e-10) #Normalize advantages
    returns = np.array(returns, dtype=np.float32)
    return returns, advs


if __name__ == "__main__":
    agent = Model()
    env = gym.make("CartPole-v0")
    state = env.reset()
    for episode in range(EPISODES):
        done = False
        total_episode_reward = 0

        rewards = []
        states = []
        actions = []
        probs = []
        dones = []
        values = []

        training = True
        if episode % 50 == 0:
            training = False
        while not done:
            state = tf.constant([state])
            i_probs, value = agent(state) 
            if training:
                if random.random() < 0.25:
                    try:
                        action = np.random.choice([0,1], p=i_probs.numpy()[0])
                    except:
                        action = np.argmax(i_probs.numpy()[0])
                else:
                    action = np.random.choice([0,1])
            else:
                action = np.argmax(i_probs.numpy()[0])
            next_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            states.append(state.numpy())
            actions.append(action)
            probs.append(i_probs)
            dones.append(done)
            values.append(value.numpy()[0][0])

            state = next_state
            total_episode_reward += reward

        state = env.reset()
        i_probs, value = agent(tf.constant([state]))
        values.append(value.numpy()[0][0])

        if training:
            returns, advs = get_returns_advantages(rewards, dones, values)
            values = values[:-1]
            for _ in range(10):
                loss = agent.train(states=states, actions=actions, old_preds=values, advs=advs, old_probs=probs, returns=returns)
            print("EPISODE: {}    REWARD: {}    LOSS: {}    ".format(episode, total_episode_reward, loss))
        else:
            print("TESTED: EPISODE: {}    REWARD: {}".format(episode, total_episode_reward))
