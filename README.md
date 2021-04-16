[image1]: assets/examples.png "image1"
[image2]: assets/drl_concept.png "image2"
[image3]: assets/episodic_conti.png "image3"
[image4]: assets/reward_hypo.png "image4"
[image5]: assets/reward_strategy.png "image5"



# Deep Reinforcement Learning Theory - Part 1 

## Content
- [Introduction](#intro)
- [Overview](#overview)
- [Install Open AI Gym](#install_open_ai_gym)
- [Cheatsheet and TextBook](#cheat_text)
- [Udacity DRL Github Repository](#uda_git_repo)
- [Deep Reinforcement Learning Nanodegree Links](#uda_nano_drl_links)
- [Elements of Reinforcement Learning](#rl_elements)
- [The Setting](#setting)
- [Episodic vs. Continuing Tasks](#episodic_continuous)
- [The Reward Hypothesis](#reward_hypo)
- [Goals and Rewards](#goals_rewards)
- [Cumulative Reward](#cum_reward)
- 
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)


## Introduction <a name="what_is_reinforcement"></a>
- Reinforcement learning is **learning** what to do — **how to map situations to actions** — so as **to maximize a numerical reward** signal. The learner is not told which actions to take, but instead must discover which actions yield the most reward by trying them. (Sutton and Barto, [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html))
- Deep reinforcement learning refers to approaches where the knowledge is represented with a deep neural network

Examples:
- [TD-Gammon](https://en.wikipedia.org/wiki/TD-Gammon), one of the first successful applications of neural networks to reinforcement learning.
- [AlphaGo Zero](https://deepmind.com/blog/article/alphago-zero-starting-scratch), the computer program that's able to beat world champions at the ancient Chinese game of Go. Reinforcement learning helps **AlphaGo Zero** to learn from this **gameplay experience**
- [Atari Games](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning)
- [Open AI's bot](https://openai.com/blog/dota-2/) that beat the world’s top players of [Dota 2](https://www.dota2.com/home)
- [Humanoid bodies learn to walk](https://deepmind.com/blog/article/producing-flexible-behaviours-simulated-environments)
- Companies such as **Uber** and **Google** are actively testing reinforcement learning algorithms and their **self-driving cars**
- **Amazon** currently uses reinforcement learning to make their **warehouses more efficient**
- **Video Games** (like Atari Breakout) are excellent candidates for Reinforcemnt Learning algorithms
- **Google Deep Mind** a piece of code to show superhuman performance at many Atari games,
and the same algorithms that we used to play games can be adapted for **robotics**
    ![image1]

## Overview <a name="overview"></a>
- Foundations of Reinforcement Learning: 
    - How to define real-world problems as **Markov Decision Processes (MDPs)**
    - Classical methods such as **SARSA** and **Q-learning** to solve several environments in **OpenAI Gym**
    - **Tile coding** and **coarse coding** to expand the size of the problems
- Value-Based Methods

    how to leverage neural networks when solving complex problems using
    - **Deep Q-Networks (DQN)**
    - **Double Q-learning**
    - **prioritized experience replay**
    - **dueling networks**

- Policy-Based Methods and actor-critic methods
    - **Proximal Policy Optimization (PPO)**
    - **Advantage Actor-Critic (A2C)**
    - **Deep Deterministic Policy Gradients (DDPG)**
    - **evolution strategies** 
    - **hill climbing**
- Multi-Agent Reinforcement Learning
    - Agents to become truly intelligent, they must be able to communicate with and learn from other agents
    - **Monte Carlo Tree Search (MCTS)**

## Install Open AI Gym <a name="install_open_ai_gym"></a>
- [Installation Guide](https://github.com/openai/gym#installation)
- It includes dozens of different environments for testing reinforcement learning agents
- It was designed as a toolkit for developing and comparing reiforcement learning algorithms
- It was developed to adddress the lack of benchmarks and standardization in RL research

## Cheatsheet and TextBook <a name="cheat_text"></a> 
- [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
- [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
- [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
    
## Udacity DRL Github Repository <a name="uda_git_repo"></a>  
- Switch to [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
- To clone the repo use
    ```
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    ```
## Deep Reinforcement Learning Nanodegree Links <a name="uda_nano_drl_links"></a>
- Student-curated list of resources for [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)

## Elements of Reinforcement Learning <a name="rl_elements"></a>  
- ***Interaction*** between an active ***decision-making agent*** and its ***environment***, within which the agent seeks to achieve a ***goal*** despite uncertainty about its environment
- Four main subelements of a reinforcement learning system: 
    - a ***policy*** - propability of taking an action when in a certain state at a given time, or mapping from perceived states of the environment to actions to be taken when in those states, a set stimulus–response rules.
    - a ***reward*** signal - the agent’s sole objective is to maximize the total reward it receives over the long run. If an action selected by the policy is followed by low reward, then the policy may be changed to select some other action in that situation in the future.
    - a ***value function*** - the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state. Rewards are in a sense primary, whereas values, as predictions of rewards, are secondary. Without rewards there could be no values, and the only purpose of estimating values is to achieve more reward. Nevertheless, it is values with which we are most concerned when making and evaluating decisions. Action choices are made based on value judgments. We seek actions that bring about states of highest value, not highest reward, because these actions obtain the greatest amount of reward for us over the long run.Unfortunately, it is much harder to determine values than it is to determine rewards. Rewards are basically given directly by the environment, but values must be estimated and re-estimated from the sequences of observations an agent makes over its entire lifetime
    - a ***model*** of the environment - a model allows inferences to be made about how the environment will behave. For example, given a state and action, the model might predict the resultant next state and next reward. Models are used for planning. Methods for solving reinforcement learning problems that use models and planning are called model-based methods. Simpler are model-free methods that are explicitly trial-and-error learners (almost the opposite of planning).

## The Setting  <a name="setting"></a>
- At the initial timestep, the agent observes the environment.
- Then, it must select an appropriate action in response.
- Then at the next timestep in response to the agents action, the environment presents a new situation to the agent.
- At the same time the environment gives the agent a reward which provides
some indication of whether the agent has responded appropriately to the environment.
- Then the process continues where at each timestep
the environment sends the agent an observation and reward.
    ![image2]

## Episodic vs. Continuing Tasks <a name="episodic_continuous"></a>
A task is an instance of the reinforcement learning (RL) problem.

### Episodic Tasks
- Reinforcemnt Learning Tasks with ***a well-defined ending point*** are called ***episodic tasks***
- In this case, we refer to a complete sequence of interaction, from start to finish, as an episode.
- Episodic tasks come to an end whenever the agent reaches a ***terminal state***.
- When the episode ends, the agent looks at the total amount of ***reward*** it received to ***figure out how well it did***. 
- Example: Playing chess
- It's then able to start from scratch as if it has been completely reborn into
the same environment but now with the ***added knowledge*** of what happened in its past life.
- In this way, as time passes over its many lives, the agent makes better and better decisions.
- Problem: Feedback is only delivered at the very end of the game. 
- ***Sparse rewards***

### Continuing Tasks
- Tasks that go on forever, ***without end*** are called ***continuing tasks***
- For instance, an algorithm that buys and sells stocks in response to
the financial market would be best modeled as an agent in the continuing tasks.
- In this case, the agent lives forever.
- So it has to learn the best way to choose actions
while simultaneously interacting with the environment.
    ![image3]

## The Reward Hypothesis <a name="reward_hypo"></a>
- Very different goals can be addressed with the same theoretical framework.
- All agents formulate their goals in terms of maximizing expected cumulative reward.
- It's important to note that the word "Reinforcement" and
"Reinforcement Learning" is a term originally from **behavioral science**. It refers to a **stimulus** that's delivered immediately after behavior to make the behavior more likely to occur in the future. In fact, it's an important to defining hypothesis.
- And we call this hypothesis, the "Reward Hypothesis". If this still seems weird or uncomfortable to you, you are not alone.

    ![image4]

## Goals and Rewards <a name="goals_rewards"></a>
Goals

- Google DeepMind addressed the problem of teaching a robot to walk.
- They worked with a physical simulation of a humanoid robot and they
managed to apply some nice reinforcement learning to get great results.
- In order to frame this as a reinforcement learning problem,
we'll have to specify the state's actions and rewards.
- In case of humanoid robot
    - ***actions*** are just the ***forces that the robot applies to its joints*** in order to move.
    - ***states*** contain the ***current positions and velocities*** of all of the joints, along with some measurements about the surface that the robot was standing on.

Rewards
- Google DeepMind developed a a reward strategy
- Each term communicates to the agent some part of what we'd like it to accomplish.
    - ***Velocity***: if robot moves faster, it gets more reward, but up to a limit (Vmax)
    - ***Force to joints***: robot is penalized by
an amount proportional to the force applied to each joint.
    - ***Moving forward***: robot should move forward,
the agent is also penalized for moving left, right, or vertically.
    - ***Center movement***: robot is penalized, if it is not close to the center.
    - ***Constant reward - Not fallen***: At every time step,
the agent also receives some positive reward if the humanoid has not yet fallen.
- Episodic task: episode is terminated when robot falls

Reward Strategy:

- Of course, the robot can't focus just on 
    - walking fast,
    - or just on moving forward,
    - or only on walking smoothly,
    - or just on walking for as long as possible.

- There are competing requirements that the agent has to balance for
all time steps towards its goal of maximizing expected cumulative reward.
- Google DeepMind demonstrated that from this very simple reward function, the agent is able to learn how to walk in a very human like fashion.

    ![image5]

## Cumulative Reward <a name="cum_reward"></a>
- The Overall goal of the walking robot: 
    - to stay walking forward for as long as possible 
    - as quickly as possible 
    - while also exerting minimal effort.
- Could the agent just maximize the reward and each time step? NO
- The agent cannot focus on individual time steps, but it needs to keep all time steps in mind.
- Actions have short and long term consequences and the agent needs
to gain some understanding of the complex effects its actions have on the environment. The robot needs to understand **long term stability**
- So in this way, the robot **moves a bit slowly to sacrifice a little bit of reward** but it will payoff because it will avoid falling for longer and collect higher cumulative reward.



## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Sparkify-Project.git
```

- Change Directory
```
$ cd Sparkify-Project
```

- Create a new Python environment, e.g. spark_env. Inside Git Bash (Terminal) write:
```
$ conda create --name spark_env
```

- Activate the installed environment via
```
$ conda activate spark_env
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
pyspark = 2.4.3
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Further Deep Reinforcement Learning References
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)