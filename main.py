from unityagents import UnityEnvironment
from dqn_agent import Agent, Config
from trainer import train_agent
from pydantic_cli import run_and_exit


def init_unity_environment(path_to_unity_env, show_graphics=True):
    env = UnityEnvironment(file_name=path_to_unity_env, no_graphics=not show_graphics)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    print(env.brain_names)
    print(env.brains)

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    return env, brain_name, state_size, action_size


def run(config: Config) -> int:

    env, brain_name, state_size, action_size = init_unity_environment(path_to_unity_env=config.unity_env_path,
                                                                      show_graphics=config.show_graphics)
    agent = Agent(state_size=state_size, action_size=action_size, config=config)
    train_agent(agent, env, brain_name)

    # Cleanup environment
    env.close()

    return 0


if __name__ == '__main__':
    run_and_exit(Config, run, description="Train DQN Agent to solve Banana Unity Environment.", version="0.0.1")
