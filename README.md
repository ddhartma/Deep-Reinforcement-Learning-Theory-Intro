[image1]: assets/examples.png "image1"
[image2]: assets/drl_concept.png "image2"
[image3]: assets/episodic_conti.png "image3"
[image4]: assets/reward_hypo.png "image4"
[image5]: assets/reward_strategy.png "image5"
[image6]: assets/reward_return.png "image6"
[image7]: assets/discounted_return.png "image7"
[image8]: assets/cart_pole.png "image8"
[image9]: assets/recycle_robot_1.png "image9"
[image10]: assets/recycle_robot_2.png "image10"
[image11]: assets/one_step_dyn.png "image11"
[image12]: assets/mdp.png "image12"
[image13]: assets/obs_space.png "image13"
[image14]: assets/action_space.png "image14"
[image15]: assets/policies.png "image15"
[image16]: assets/grid_world.png "image16"
[image17]: assets/state_value_func_1.png "image17"
[image18]: assets/state_value_func_2.png "image18"
[image19]: assets/bellman_equ.png "image19"
[image20]: assets/optimality.png "image20"
[image21]: assets/action_value_func.png "image21"
[image22]: assets/v_pi_q_pi.png "image22"
[image23]: assets/optimal_policy_from_q.png "image23"
[image24]: assets/q_as_table.png "image24"

# Deep Reinforcement Learning Theory - Intro

## Content
- [Introduction](#intro)
- [Overview](#overview)
- [Install Open AI Gym](#install_open_ai_gym)
- [Cheatsheet and TextBook](#cheat_text)
- [Udacity DRL Github Repository](#uda_git_repo)
- [Deep Reinforcement Learning Nanodegree Links](#uda_nano_drl_links)
- [The RL Framework: The Problem](#rl_frame_prob)
    - [Elements of Reinforcement Learning](#rl_elements)
    - [The Setting](#setting)
    - [Episodic vs. Continuing Tasks](#episodic_continuous)
    - [The Reward Hypothesis](#reward_hypo)
    - [Goals and Rewards](#goals_rewards)
    - [Cumulative Reward](#cum_reward)
    - [Discounted Return](#disc_return)
    - [Markov Decision Process (MDP)](#mdp)
    - [Finite MDPs](#finite_mdps)
- [The RL Framework: The Solution](#rl_frame_sol)
    - [Policies](#Policies)
    - [State-Value Functions](#state_val_func)
    - [The Gridworld Example](#grid_world_example)
    - [Bellman Equations](#Bellman_Equations)
    - [Calculating Expactations](#calc_expectations)
    - [Optimality](#Optimality)
    - [Action-Value Functions](#Action_Value_Functions)
    - [Optimal Policies](#Optimal_Policies)
    
- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)


## Introduction <a name="intro"></a>
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
    - **Prioritized experience replay**
    - **Dueling networks**

- Policy-Based Methods and actor-critic methods
    - **Proximal Policy Optimization (PPO)**
    - **Advantage Actor-Critic (A2C)**
    - **Deep Deterministic Policy Gradients (DDPG)**
    - **Evolution strategies** 
    - **Hill climbing**
- Multi-Agent Reinforcement Learning
    - Agents to become truly intelligent, they must be able to communicate with and learn from other agents
    - **Monte Carlo Tree Search (MCTS)**

## Install Open AI Gym <a name="install_open_ai_gym"></a>
- [Installation Guide](https://github.com/openai/gym#installation)
- It includes dozens of different environments for testing reinforcement learning agents
- It was designed as a toolkit for developing and comparing reiforcement learning algorithms
- It was developed to address the lack of benchmarks and standardization in RL research

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

## The RL Framework: The Problem <a name="rl_frame_prob"></a> 

## Elements of Reinforcement Learning <a name="rl_elements"></a>  
- ***Interaction*** between an active ***decision-making agent*** and its ***environment***, within which the agent seeks to achieve a ***goal*** despite uncertainty about its environment
- Four main subelements of a reinforcement learning system: 
    - a ***policy*** - propability of taking an action when in a certain state at a given time, or mapping from perceived states of the environment to actions to be taken when in those states, a set stimulus–response rules.
    - a ***reward*** signal - the agent’s sole objective is to maximize the total reward it receives over the long run. If an action selected by the policy is followed by low reward, then the policy may be changed to select some other action in that situation in the future.
    - a ***value function*** - the value of a state is the total amount of reward an agent can expect to accumulate over the future, starting from that state. Rewards are in a sense primary, whereas values, as predictions of rewards, are secondary. Without rewards there could be no values, and the only purpose of estimating values is to achieve more reward. Nevertheless, it is values with which we are most concerned when making and evaluating decisions. Action choices are made based on value judgments. We seek actions that bring about states of highest value, not highest reward, because these actions obtain the greatest amount of reward for us over the long run. Unfortunately, it is much harder to determine values than it is to determine rewards. Rewards are basically given directly by the environment, but values must be estimated and re-estimated from the sequences of observations an agent makes over its entire lifetime
    - a ***model*** of the environment - a model allows inferences to be made about how the environment will behave. For example, given a state and action, the model might predict the resultant next state and next reward. Models are used for planning. Methods for solving reinforcement learning problems that use models and planning are called model-based methods. Simpler are model-free methods that are explicitly trial-and-error learners (almost the opposite of planning).

## The Setting  <a name="setting"></a>
- The reinforcement learning (RL) framework is characterized by an **agent** learning to **interact** with its **environment**.
- At each time step, the agent receives the environment's **state** (the environment presents a situation to the agent), and the agent must choose an appropriate **action** in response. One time step later, the agent receives a **reward** (the environment indicates whether the agent has responded appropriately to the state) and a new **state**.
- All agents have the goal to maximize **expected cumulative reward**, or the expected sum of rewards attained over all time steps.

    ![image2]

## Episodic vs. Continuing Tasks <a name="episodic_continuous"></a>
A task is an instance of the reinforcement learning (RL) problem.

### Episodic Tasks
- Reinforcement Learning Tasks with ***a well-defined ending point*** are called ***episodic tasks***
- In this case, we refer to a complete sequence of interaction, from start to finish, as an ***episode***.
- Episodic tasks come to an end whenever the agent reaches a ***terminal state***.
- When the episode ends, the agent looks at the total amount of ***reward*** it received to ***figure out how well it did***. 
- Example: Playing chess
- It's then able to start from scratch as if it has been completely reborn into the same environment but now with the ***added knowledge*** of what happened in its past life.
- In this way, as time passes over its many lives, the agent makes better and better decisions.
- Problem: Feedback is only delivered at the very end of the game. 
- ***Sparse rewards***

### Continuing Tasks
- Tasks that go on forever, ***without end*** are called ***continuing tasks***
- For instance, an algorithm that buys and sells stocks in response to the financial market would be best modeled as an agent in the continuing tasks.
- In this case, the agent lives forever.
- So it has to learn the best way to choose actions
while simultaneously interacting with the environment.
    ![image3]

## The Reward Hypothesis <a name="reward_hypo"></a>
- All goals can be framed as the maximization of (expected) cumulative reward.

    ![image4]

## Goals and Rewards <a name="goals_rewards"></a>
### Goals
- Google DeepMind addressed the problem of teaching a robot to walk.
- They worked with a physical simulation of a humanoid robot and they
managed to apply some nice reinforcement learning to get great results.
- In order to frame this as a reinforcement learning problem,
we'll have to specify the state's actions and rewards.
- In case of humanoid robot
    - ***actions*** are just the ***forces that the robot applies to its joints*** in order to move.
    - ***states*** contain the ***current positions and velocities*** of all of the joints, along with some measurements about the surface that the robot was standing on.

### Rewards
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

### Reward Strategy:
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
- Could the agent just **maximize the reward in each time step**? NO
- The agent **cannot focus on individual time steps**, but it needs to **keep all time steps in mind**.
- Actions have short and long term consequences and the agent needs
to gain some understanding of the complex effects its actions have on the environment. The robot needs to understand **long term stability**
- So in this way, the robot **moves a bit slowly to sacrifice a little bit of reward** but it will payoff because it will avoid falling for longer and **collect higher cumulative reward**.

- How exactly does the robot keep all time steps in mind?
    - Looking at some time step, t, it's important to note that the rewards for all previous time steps have already been decided as they're in the past.
    - Only future rewards are inside the agent's control.
    - The **sum of rewards** from the next time step onward is the **return G**, and at an arbitrary time step, the agent will always **choose an action with goal of maximizing the expected return**.
    - **Expected return** instead of return: The agent normally can't predict with complete certainty what the future reward is likely to be. So it has to rely on a prediction or an estimate.

    ![image6]

## Discounted Return <a name="disc_return"></a> 
- **Rewards that come sooner should be valued more** highly, since those rewards are **more predictable**.
- Use a **discount rate** for the expected return **to care about future time steps**.
- The **larger the discaount rate** is, the more the agent cares about the **distant future**.
- The **smaller the discaount rate** is, the more the agent cares about the **most immediate reward**.
- It's important to note that discounting is particularly relevant to **continuing tasks** (interaction goes on without end). A discount rate helps to avoid to look too far into the limitless future.
- But it's important to note that with or without discounting,
the goal is always the same. It's always to maximize cumulative reward.

    ![image7]

- Here "Return" and "discounted return" is used interchangably. For an arbitrary time step t, both refer to
    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle G_{t}=\sum _{k=0}^{\infty }\gamma ^{k}R_{t %2B k %2B 1}" width="180px"> 

    and

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle \gamma \in [0,1)" width="100px">

- A good choice for discount rate in many problems is **0.9**.


- Example: [Cart-pole-balancing in OpenAI Gym](https://gym.openai.com/envs/CartPole-v0/)
    - [Medium-Source](https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947)
    - Recall that the agent receives a reward of +1 for every time step, including the final step of the episode. Which discount rates (1, 0.9, 0.5) would encourage the agent to keep the pole balanced for as long as possible? Answer: For all three discount rates
    - Say that the reward signal is amended to only give reward to the agent at the end of an episode. So, the reward is 0 for every time step, with the exception of the final time step. When the episode terminates, the agent receives a reward of -1. Which discount rates (1, 0.9, 0.5, none of these) would encourage the agent to keep the pole balanced for as long as possible? Answer: 0.9 and 0.5 (With discounting, the agent will try to keep the pole balanced for as long as possible, as this will result in a return that is relatively less negative.)
    - Say that the reward signal is amended to only give reward to the agent at the end of an episode. So, the reward is 0 for every time step, with the exception of the final time step. When the episode terminates, the agent receives a reward of +1. Which discount rates (1, 0.9, 0.5, none of these) would encourage the agent to keep the pole balanced for as long as possible? Answer: None of these (If the discount rate is 1, the agent will always receive a reward of +1, If the discount rate is 0.5 or 0.9, the agent will try to terminate the episode as soon as possible)

    ![image8]

## Markov Decision Process (MDP) <a name="mdp"></a>

### The problem statement:
- So consider a robot that's designed for picking up empty soda cans.
- The robot is equipped with arms to grab the cans and runs on a rechargeable battery.
- There's a docking station set up in one corner of the room and
the robot has to sit at the station if it needs to recharge its battery.
- The robot is be able to decide for itself when it needs to recharge its battery.

### Frame this as a reinforcement learning problem:
- **Actions A**:
    - Three potential actions.
    - It can search the room for cans,
    - It can head to the docking station to recharge its battery,
    - It can stay put in the hopes that someone brings it a can.
    - The action space **A** is the set of possible actions available to the agent.
    -  Use **A(s)** if only a small set of actions is available in state **S**.

- **States S**:
    - States are just the context provided to the agent for making intelligent actions.
    - States could be the charge left on the robot's battery.
    - For simplicity, let's assume that the battery has one of two states (high amount of charge left --> higher chance that robot will search for cans, low amount of charge --> lower chance that robot will search for cans)
    - In general, the state space **S** is the set of all nonterminal states.
    - In continuing tasks (like here), this is equivalent to the set of all states.
    - In episodic tasks, we use **S<sup>+</sup>** to refer to the set of all states, including terminal states.

    ![image9]

    ![image10]

### One-Step Dynamics
- At an arbitrary time step **t**, the agent-environment interaction has evolved as a sequence of states, actions, and rewards

    **(S<sub>0</sub>, A<sub>0</sub>, R<sub>1</sub>, S<sub>1</sub>, A<sub>1</sub>, …, R<sub>t−1</sub>, S<sub>t−1</sub>, A<sub>t−1</sub>, R<sub>t</sub>, S<sub>t</sub>, A<sub>t</sub>)**
- When the environment responds to the agent at time step **t+1**, it considers **only** the state and action at the previous time step **(S<sub>t</sub>, A<sub>t</sub>)**. Prior states are not regarded by the environment.
- How much reward the agent is collecting, has no effect on how the environment chooses to respond to the agent. Hence, the environment does not consider any of **{R<sub>0</sub>, …, R<sub>t</sub>}**.

- Because of this, we can completely define how the environment decides the state and reward by specifying 
    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle p(s^{'},r \mid s,a) =\P(S_{t %2B 1}, R_{t %2B 1}= r \mid S_{t} = s,A_{t}=a)" width="500px">

    for each possible **s′**, **r**, **s** and **a**. These conditional probabilities are said to specify the **one-step dynamics** of the environment.

    ![image11]

### The Markow Decision Process (MDP) definition:
 
  - ![image12]

## Finite MDPs <a name="finite_mdps"></a> 
 - Use this [link](https://github.com/openai/gym/wiki/Table-of-environments) to use the available environments in OpenAI Gym
 - The environments are indexed by Environment Id, and each environment has corresponding **Observation Space**, **Action Space**, **Reward Range**, **tStepL**, **Trials**, and **rThresh**.
 - Every environment comes with first-class Space objects that describe the valid actions and observations.
    - The **Discrete space** allows a fixed range of non-negative numbers.
    - The **Box space** represents an n-dimensional box, so valid actions or observations will be an array of n numbers.
- **Observation Space**: The observation space for the CartPole-v0 environment has type **Box(4,)**. Thus, the state at each time point is an array of 4 numbers. Check this [document](https://github.com/openai/gym/wiki/CartPole-v0). 

    ![image13]

    Since the entry in the array corresponding to each of these indices can be any real number, the state space **S<sup>+</sup>** is infinite.

    
- **Action Space**: The action space for the CartPole-v0 environment has type **Discrete(2)**. Thus, at any time point, there are only two actions available to the agent. You can look up what each of these numbers represents in this document (note that it is the same document you used to look up the observation space!). After opening the page, scroll down to the description of the action space.

    ![image14]

    In this case, the action space **A** is a finite set containing only two elements.


### Finite MDP definition:
- In a finite MDP, the state space **S** (or **S<sup>+</sup>**, in the case of an episodic task) and action space **A** must both be finite.
- Thus for Cart-Pole-v0 there are infinite states, it is not a finite MDP.


## The RL Framework: The Solution  <a name="rl_frame_sol"></a> 
- This lesson covers material in Chapter 3 (especially 3.5-3.6) of the [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf).
- Before we defined the problem via MDP
- Now let's find a solution for this problem statement
- Reward is always decided in the context of the state that it
was decided in along with the state that follows.



## Policies <a name="Policies"></a>
- As long as the agent learns an appropriate action response to any environment state that it can observe, we have a solution to our problem.
- This motivates the idea of a policy.
- Deterministic policy: The simplest kind of policy is a mapping from the set of environment states to the set of possible actions. The deterministic policy would specify something
like whenever the battery is low, recharge it. And whenever the battery has a high amount of charge, search for cans.
- Stochastic policy: will allow the agent to choose actions randomly.
  It is a mapping of a state S and action A and returns the probability that the agent takes action a while in state s. The stochastic policy does something more like whenever the battery is low, recharge it with 50 percent probability, wait where you are with 40 percent probability. And otherwise, search for cans. Whenever the battery is high, earch for cans with 90 percent probability. And otherwise, wait for a can.

### A deterministic policy 
- is a mapping 

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle \pi : S  \rightarrow A" width="130px">

- For each state 
    
    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle s \in S " width="70px">

    it yields the action 

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle a \in A " width="70px">

    that the agent will choose while in state **s**.

### A stochastic policy 
- is a mapping 

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle \pi : S  \rightarrow [0, 1]" width="160px">

- For each state 

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle s \in S " width="70px">

    and action

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle a \in A " width="70px">

    it yields the probability 

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle \pi(a \mid s) " width="70px">

    that the agent will choose action **a** while in state **s**.

    ![image15]



## The Gridworld Example <a name="grid_world_example"></a>
- In order to understand how to find the best policy consider this very small Gridworld Example and an agent who lives in it.
- **Environment**: The world is primarily composed of nice patches of grass,
but two out of the nine locations in the world have large mountains.
- **States**: 9 locations are states.
- **Actions**: move up, down, left or right, and can only take actions that lead it to not fall off the grid.
- **Episodic task** where an episode finishes when the agent reaches the goal. So we won't have to worry about transitions away from this goal state.
- The **reward signal** punishes the agent for every timestep that it spends away from the goal (-1 and -3 in case of mountain).
- The reward structure encourages the agent to get to the goal as quickly as possible.
- When it reaches the goal, it gets a reward of 5, and the episode ends.

    ![image16]

- We're working with this grid world example and looking for
the best policy that leads us to a goal state as quickly as possible.
- So, let's start with a very, very bad policy, e.g. where the agent
visits every state in this very roundabout manner,
and we can ignore the transition that the agent will never take under the policy.
- If the agent starts in the top left corner of
the world and follows this policy to get to the goal state,
it just collects all of the reward along the way.

    ![image17]

## State-Value Functions <a name="state_val_func"></a> 

- The state-value function for a policy **π**  is denoted **v<sub>π</sub>**. 
- For each state 
    
    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle s \in S " width="70px">

    it yields the expected return if the agent starts in state **s** and then uses the policy to choose its actions for all time steps.

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle v_{\pi}(s) \doteq \E_{\pi}[G_t \mid S_t = s]" width="300px">

- **v<sub>π</sub>(s)** is the value of state **s** under policy **π**.

    ![image18]

## Bellman Equations <a name="Bellman_Equations"></a>
- For value calculation you don't need to start your calculations from scratch every time. It turns out to be redundant effort.
- The value function has a recursive property.
- The Bellman expectation equation for **v<sub>π</sub>** is: 

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle v_{\pi}(s) \doteq \E_{\pi}[R_{t %2B 1} %2B \gamma v_{\pi}(S_{t %2B 1}) \mid S_t = s]" width="500px">

    ![image19]

- In total there are 4 Bellman Equations. See sections 3.5 and 3.6 of the [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf). The Bellman equations are incredibly useful to the theory of MDPs.

## Calculating Expectations <a name="calc_expectations"></a>
- **Deterministic  policy π**: the agent in state **s** selects action **π(s)**, and the Bellman Expectation Equation can be rewritten as the sum over two variables (**s′** and **r**):

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle v_{\pi}(s) = \sum _{s^{'} \in S^{%2B}, r \in R}^{} p(s^{'}, r \mid s, \pi(s))(r %2B \gamma v_{\pi}(s^{'}))" width="600px">

    In this case, we multiply the sum of the reward and discounted value of the next state **(r + γv<sub>π</sub>(s′))** by its corresponding probability **p(s′,r ∣ s,π(s))** and sum over all possibilities to yield the expected value.

- **Stochastic policy π**: the agent in state **s** selects action **a** with probability **π(a ∣ s)**, and the Bellman Expectation Equation can be rewritten as the sum over three variables (**s′**, **r**, and **a**):

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle v_{\pi}(s) = \sum _{s^{'} \in S^{%2B}, r \in R, a \in A}^{} \pi(a \mid s) p(s^{'},r \mid s,a)(r %2B \gamma v_{\pi}(s^{'}))" width="700px">

    In this case, we multiply the sum of the reward and discounted value of the next state **(r + γvπ(s′))** by its corresponding probability **π(a ∣ s)p(s',r | s,a)** and sum over all possibilities to yield the expected value.

## Optimality <a name="Optimality"></a>
-  Look at any state in particular and compare the two value functions, the value function for a **π'** is always bigger
than or equal to the value function for policy **π**.
- So this says, for any state in the environment,
it's better to follow policy **π'**, because no matter where the agent starts in the grid world, the expected discounted return is larger, and remember that the goal of the agent is to maximize return.
- So, ***a greater expected return makes for a better policy***.
- Definition: A policy **π'** is better than or equal to
a policy **π** if it's state-value function is greater
than or equal to that of policy **π** for all states.

- A policy **π′** is defined to be **better than or equal to** a policy **π**, i.e.

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle \pi^' \geq \pi" width="90px">

    if and only if 

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle v_{\pi^'}(s) \geq v_{\pi}(s)" width="200px">

    for all 

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle s \in S " width="70px">

- An optimal policy **π<sub>∗</sub>** satisfies 

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle \pi_{*} \geq \pi" width="100px">

    for all policies **π**. 

- Some important notes: 
    1. It is possible to take two policies, that are equally good, i.e. it's possible that they can't be compared.
    2. At least there must be one policy that's better than or equal to all other policies. We call this policy an **optimal policy** **π<sub>*</sub>**,
    3. An **optimal policy** **π<sub>*</sub>** is guaranteed to exist but it may not be unique.
    4. The agent is searching for an optimal policy. It's the solution to the MDP and the best strategy to accomplish it's goal.
    5. All **optimal policies** **π<sub>*</sub>** have the same value function **v<sub>*</sub>**.

    ![image20]

    
## Action-Value Functions <a name="Action_Value_Functions"></a>
- The action-value function for a policy **π** is denoted **q<sub>π</sub>**. 
- For each state 

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle s \in S " width="70px">
    
    and action 
     
    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle \pi(a \mid s) " width="70px">
    
    it yields the expected return if the agent starts in state **s**,takes action **a**, and then follows the policy for all future time steps. That is, 
    
    
    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle q_{\pi}(s,a) \doteq \E_{\pi}[G_{t} \mid S_t = s, A_t = a]" width="500px">
    
- We refer to **q<sub>π</sub>(s,a)** as the **value of taking action a in state s under a policy π** (or alternatively as the **value of the state-action pair s,a**).
- All optimal policies have the same action-value function **q<sub>∗</sub>**, called the **optimal action-value function**.

   ![image21]

- Important note for a **deterministic policy π**:

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle v_{\pi}(s) = q_{\pi}(s, \pi(s))" width="300px">

    holds for all 

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle s \in S" width="70px">
    
    ![image22]



## Optimal Policies <a name="Optimal_Policies"></a>
- Once the agent determines the optimal action-value function **q<sub>∗</sub>**, it can quickly obtain an optimal policy π<sub>∗</sub> by setting 

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle \pi_{*}(s) = argmax_{a \in A(s)} q_{*}(s,a)" width="500px">

    for all 

    <img src="https://render.githubusercontent.com/render/math?math=\displaystyle s \in S" width="70px">

- The main idea is this:
    - The agent interacts with the environment.
    - From that interaction, it estimates the optimal action value function.
    - Then, the agent uses that value function to get the optimal policy.
    - Let's assume it already knows the optimal action-value function, but it doesn't know the corresponding optimal policy.
    - For each state, we just need to **pick the action that yields the highest expected return**.
    - Finding the optimal policy is the solution to the MDP and the best strategy to accomplish the agent's goal.

    ![image23]

- Let's populate some values for a hypothetical Markov decision process (MDP), where **S = {s1, s2, s3}** and **A = {a1, a2, a3}**.
- The optimal action-value function **q<sub>*</sub>** can be represented in a table
- Select the entries that maximize the action-value function, for each row (or state).

    ![image24]


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
* [Very good summary of DQN](https://medium.com/@nisheed/udacity-deep-reinforcement-learning-project-1-navigation-d16b43793af5)
* [Cheatsheet](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/cheatsheet/cheatsheet.pdf)
* [Reinforcement Learning Textbook](https://s3-us-west-1.amazonaws.com/udacity-drlnd/bookdraft2018.pdf)
* [Reinforcement Learning Textbook - GitHub Repo to Python Examples](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction)
* [Udacity DRL Github Repository](https://github.com/udacity/deep-reinforcement-learning)
* [Open AI Gym - Installation Guide](https://github.com/openai/gym#installation)
* [Deep Reinforcement Learning Nanodegree Links](https://docs.google.com/spreadsheets/d/19jUvEO82qt3itGP3mXRmaoMbVOyE6bLOp5_QwqITzaM/edit#gid=0)