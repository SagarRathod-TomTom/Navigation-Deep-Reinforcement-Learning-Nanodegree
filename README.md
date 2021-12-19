# Navigation

## Project Goal

The goal is to train an agent to navigate a virtual world and collect as many yellow banana as possible while avoiding 
blue bananas.

![In Project 1, train an agent to navigate a large world.](resources/images/banana.gif)

## Environment Details

The environment is based on [Unity ML-agents](https://github.com/Unity-Technologies/ml-agents).

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


## Note

The project environment is similar to, but not identical to the Banana Collector environment on the [Unity ML-Agents GitHub page](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector).
For this project, you will not need to install Unity. You can download the environment from one of the links below. 
You need only select the environment that matches your operating system:

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Once you download the zip file, unzip or decompress it.

## Quick Start

### 1. Install:

#### Prerequisite:
Python 3.6+
    
1. Clone the repository.
```shell
git clone https://github.com/SagarRathod-TomTom/Navigation-Deep-Reinforcement-Learning-Nanodegree.git
```
2. Install dependencies.
```shell
cd Navigation-Deep-Reinforcement-Learning-Nanodegree
pip install -r requirements.txt
```

### 2. Usage:
```shell
python main.py --help
```
##### Parameters:
```shell
usage: main.py --unity_env_path UNITY_ENV_PATH [--batch_size BATCH_SIZE]
               [--gamma GAMMA] [--eps EPS] [--eps_end EPS_END]
               [--eps_decay EPS_DECAY] [--tau TAU] [--lr LR]
               [--update_every UPDATE_EVERY] [--disable-graphics]
               [--seed SEED] [--help] [--version]

Train DQN Agent to solve Banana Unity Environment.

optional arguments:
  --unity_env_path UNITY_ENV_PATH
                        (type:str default:None)
  --batch_size BATCH_SIZE
                        (type:int default:64)
  --gamma GAMMA         (type:float default:0.9)
  --eps EPS             (type:float default:1.0)
  --eps_end EPS_END     (type:float default:0.01)
  --eps_decay EPS_DECAY
                        (type:float default:0.995)
  --tau TAU             (type:float default:0.001)
  --lr LR               (type:float default:0.001)
  --update_every UPDATE_EVERY
                        (type:int default:4)
  --disable-graphics    Set graphics to False
  --seed SEED           (type:int default:42)
  --help                Print Help and Exit
  --version             show program's version number and exit
  
```

### 3. Train agent with default parameters.
```shell
python main.py --unity_env_path path/to/BananaEnv
```
The process terminates when the agent achieves the average score of 13 or more.


## Implementation Details:
Checkout [Report.md](Report.md) for in-depth implementation details.
