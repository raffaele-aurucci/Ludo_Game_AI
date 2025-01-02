# Ludo Game AI

Project of course **Artificial Intelligence** - University of Salerno.

## Contributors
[@raffaele-aurucci](https://github.com/raffaele-aurucci), [@AngeloPalmieri](https://github.com/AngeloPalmieri), [@CSSabino](https://github.com/CSSabino).

## Table of Contents

1. [Introduction](#introduction)
2. [Methodology](#methodology)
   - [Simplified Ludo](#simplified-ludo)
   - [Observation Space and Action Space](#observation-space-and-action-space)
   - [Gameplay Strategy](#gameplay-strategy)
   - [Training models](#training-models)
3. [Experimental Results](#experimental-results)
4. [Installation Guide](#installation-guide)
   - [Installing Python](#installing-python)
   - [Cloning the Repository](#cloning-the-repository)
   - [Creating the Virtual Environment](#creating-the-virtual-environment)
   - [Installing Requirements](#installing-requirements)
5. [References](#references)

## Introduction
In this project, we explore the application of **Reinforcement Learning (RL)** to the classic board game **Ludo**, 
an environment that combines elements of strategy, randomness, and luck. More information on the game available at this [link](https://en.wikipedia.org/wiki/Ludo).  
Ludo, with its inherent complexity, was simplified to represent various game scenarios in reduced manner. 
This simplification facilitates the implementation of **Q-table-based** techniques for training intelligent agents. 
Starting with the definition of the state space, we developed a gameplay strategy and trained three types of agents: 
one based on **Q-learning**, one on **SARSA**, and one on **DQ-learning**. The goal was to evaluate which approach proved most 
effective in the context of Ludo. Results show that the DQ-learning model achieves the best performance, 
with win rates exceeding 70%.

## Methodology
In this work we have trained three types of agents, following the below methodology.

### Simplified Ludo
The Ludo game was simplified to include only 2 players, each with 2 tokens. 
In this version of the game, rolling a six on the dice is not required to move a token out of the base, and a single 
token can capture two enemy tokens on the same tile. The game ends when one player has two tokens in GOAL. 
This simplification makes it easier to apply Q-table-based techniques for training intelligent agents.

The image below shows a Ludo board with 2 players, each controlling 2 tokens. The agent is represented by the green tokens, while the enemy is represented by the blue tokens.

<img width=500 src="https://github.com/user-attachments/assets/9a9137b4-416c-4e91-8612-12a6409d1d8e">

### Observation Space and Action Space
We have defined the **observation space** and **action space** of the agent using gymnasium environment. The observation space includes:

| **State** | **Values**    | **Description**                                                             |
|-----------|---------------|-----------------------------------------------------------------------------|
| HOME      | {0, 1, 2}     | Specifies the number of tokens in the base.                                 |
| PATH      | {0, 1, 2}     | Specifies the number of tokens on the path.                                 |
| SAFE      | {0, 1, 2}     | Specifies the number of tokens in the safe zone.                            |
| GOAL      | {0, 1, 2}     | Specifies the number of tokens that have reached the goal.                  |
| EV1       | {0, 1}        | Boolean value indicating if the enemy is vulnerable to the agent's token 1. |
| EV2       | {0, 1}        | Boolean value indicating if the enemy is vulnerable to the agent's token 2. |
| TV1       | {0, 1}        | Boolean value indicating if the agent's token 1 is under attack by enemy.   |
| TV2       | {0, 1}        | Boolean value indicating if the agent's token 2 is under attack by enemy.   |


The action space includes: 

| **Action**   | **Values** | **Description**                                   |
|--------------|------------|---------------------------------------------------|
| MOVE TOKEN 1 | {0, 1}     | Boolean value indicating whether to move token 1  |
| MOVE TOKEN 2 | {0, 1}     | Boolean value indicating whether to move token 2. |

### Gameplay Strategy
When designing a gameplay strategy, the primary goal is to win. To achieve this, 
a strategy was developed that alternates between **defense** and **offense**: the agent defends its tokens when they are under 
attack and attempts offensive moves when one of its tokens can attack the enemy.  
The strategy is supported by a **reward system** based on the agent's state and actions. For example, if TOKEN 1 is under attack and the agent successfully defends it, 
a positive reward (**+5**) is granted, alternatively a negative reward (**-5**) is obtained.  
Other rewards include moving the token 
that is furthest ahead in situations where there is indecision on which token to defend or attack (**+30** for a favorable move, 
**-30** for an unfavorable one), reaching the SAFE zone (**+5**), reaching the GOAL zone (**+20**), exiting from HOME (**+18**), 
capturing the enemy's token (**+7**), and being captured by the enemy (**-7**). Finally, winning the game rewards the agent with (**+50**).  

In this scenario, the agent's two tokens are positioned in PATH, and TOKEN 1 is in a position to attack the enemy. By moving TOKEN 1, the agent can successfully capture the enemy, earning a total reward of **+42**. This reward is composed of **+5** for the offensive action, **+30** for a strategically favorable move, and **+7** for capturing the enemy."

<img width=500 src="https://github.com/user-attachments/assets/0a3a635c-6267-482a-9aaf-5a5002cf8c9f"> <br>

In this other scenario, the agent's two tokens are positioned in PATH, and TOKEN 1 is under attack by the enemy. By moving TOKEN 1, the agent can successfully defend itself, earning a total reward of **+35**. This reward includes **+5** for the defensive action and **+30** for making a strategically favorable move.

<img width=500 src="https://github.com/user-attachments/assets/548569a6-2fab-477c-aec7-7cd3da9fd0b5">



### Training models
The agent implementation involved the development of three algorithms: **Q-learning**, **SARSA** and **DQ-learning**. 
The simplification of the game environment made the application of a Q-table-based strategy particularly interesting, 
which is why the first two algorithms were chosen. DQ-learning, which approximates the Q-table using a neural network, 
was included to explore the differences compared to the other two methods. The training was carried out in several phases:

1. Training against a random enemy (identified as **Random**) while trying different parameter configurations, typical of a grid search approach.
2. Training against itself, an enemy that follows the same policy as the best agent obtained in the previous step. 
The Q-table of both the agent and the enemy is initialized with previously saved values. In DQ-learning, the neural network is initialized with previously saved weights. It is important to note that only the agent is updated during this process, not the opponent. The newly trained model is identified as the **Self-Agent**.
3. Retraining the **Self-Agent** against itself to assess if iterative improvements are made.
4. Testing the models obtained in the previous three steps against random enemy and enemy trained with previously learned policies.

## Experimental Results
The table below shows the performance of the agents after training with the best hyperparameters:

| Agent            | Enemy       | % Win | ε   | α   | γ   |
|------------------|-------------|-------|-----|-----|-----|
| Q-Agent          | Random      | 64.58 | 0.1 | 0.3 | 0.5 |
| SARSA-Agent      | Random      | 61.25 | 0.2 | 0.4 | 0.3 |
| DQL-Agent        | Random      | 70.96 | 1   | 0.3 | 0.3 |
|                  |             |       |     |     |     |
| Q-Self-Agent     | Random      | 58.8  | 0.1 | 0.3 | 0.5 |
| SARSA-Self-Agent | Random      | 62.48 | 0.2 | 0.4 | 0.3 |
| DQL-Self-Agent   | Random      | 70.96 | 1   | 0.3 | 0.3 |
|                  |             |       |     |     |     |
| Q-Self-Agent     | Q-Agent     | 45.6  | 0.1 | 0.3 | 0.5 |
| SARSA-Self-Agent | SARSA-Agent | 49.46 | 0.2 | 0.4 | 0.3 |
| DQL-Self-Agent   | DQL-Agent   | 56.71 | 1   | 0.3 | 0.3 |

## Installation Guide
To install the necessary requirements for the project, please follow the steps below.

### Installing Python
Verify you have Python installed on your machine. The project is compatible with Python `3.12`.

If you do not have Python installed, please refer to the official [Python Guide](https://www.python.org/downloads/).

### Cloning the Repository 
To clone this repository, download and extract the `.zip` project files using the `<Code>` button on the top-right or run the following command in your terminal:
```shell 
git clone https://github.com/raffaele-aurucci/Ludo_Game_AI.git
```

### Creating the Virtual Environment 
It's strongly recommended to create a virtual environment for the project and activate it before proceeding. 
Feel free to use any Python package manager to create the virtual environment. However, for a smooth installation of the requirements we recommend you use `pip`. Please refer to [Creating a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).

You may skip this step, but please keep in mind that doing so could potentially lead to conflicts if you have other projects on your machine. 
### Installing Requirements
To install the requirements, please: 
1. Make sure you have **activated the virtual environment where you installed the project's requirements**. If activated, your terminal, assuming you are using **bash**, should look like the following: ``(name-of-your-virtual-environment) user@user path``

2. Install the project requirements using `pip`:
```shell 
pip install -r requirements.txt
```


## References
This project borrows partially from [LudoPy](https://github.com/SimonLBSoerensen/LUDOpy.git) repository.
