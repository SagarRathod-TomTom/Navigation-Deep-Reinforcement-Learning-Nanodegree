from collections import deque
import numpy as np
import torch


def train_agent(agent, env, brain_name, episodes=2000, max_t=1000):
    scores = []
    scores_window = deque(maxlen=100)  # last 100 scores

    for episode in range(episodes):

        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]

        # get the current state
        state = env_info.vector_observations[0]

        # initialize the score
        score = 0
        for t in range(max_t):
            # Select agent action
            action = agent.act(state)

            # send the action to the environment
            env_info = env.step(action)[brain_name]
            # get the next state
            next_state = env_info.vector_observations[0]
            # get the reward
            reward = env_info.rewards[0]
            # see if episode has finished
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)

            # update the score
            score = score + reward
            # roll over the state to next time step
            state = next_state

            # exit loop if episode finished
            if done:
                break

        scores_window.append(score)
        scores.append(score)

        print(f"\rEpisode {episode + 1}/{episodes} \tScore: {np.mean(scores_window)}", end="")

        if (episode + 1) % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode + 1, np.mean(scores_window)))

        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break

    return scores
