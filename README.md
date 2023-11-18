# Are Language Agents powerful enough for classic control problems in RL?

Large Language Models (LLM) exhibit emergent abilities in tasks like planning and decision-making akin to those of human beings, which has inspired the creation of agents that use LLM as their main controller. While many of the augmentations being added to these models today (tool usage, memory, prompting, etc.) improve the performance of language agents in various tasks, it is still unclear how well they perform in classic control environments.

This project aims to test whether a statistically significant difference exists between the performance of LLM agents and some of the oldest RL models used to solve classic control problems. This is with the objective of knowing if there is any merit at all in the LLM Agent methodology in low dimensionality environments where the reward is not sparse and simple solutions have achieved great results.

Up to this point in time, we have implemented three different agents to learn a control policy for the [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) classic control problem.

## Agents

### Boxes

The first agent is called [Boxes](https://citeseerx.ist.psu.edu/document?doi=2f027193fb703d0af58ec382bd1438daff9417d7) which main idea is to divide the environment into mutually independent sub-games that can be mapped from the environment states and learn a control policy for each of the sub-games. The reward achieved (episode duration) for by this agent:

+ Mean: 176.68.
+ Standard deviation: 111.04709631503202.
+ Max: 500.0.
+ Min: 11.0.
+ Total episodes with a reward gte 500: 1.
+ Sample size (total episodes): 100.

log transformation of the reward:

+ Mean: 4.866795190242914.
+ Standard deviation: 0.8987026953290442.

### Random

This is the control group for the experiment, specifically, Random is an agent whose policy of control is distributed as a Bernoulli(p=0.5). If the random experiment is a success, then the cart gets pushed to the right, otherwise the cart is pushed to the left. The reward achieved (episode duration) for by this agent:

+ Mean: 20.29.
+ Standard deviation: 11.661299241508212.
+ Max: 77.0.
+ Min: 9.0.
+ Total episodes with a duration gte 500 is 0.
+ Sample size (total episodes): 100.

log transformation of the reward:

+ Mean: 2.8906728158585313.
+ Standard deviation: 0.4592719550488882.

### LLM

Our final agent uses an LLM as its controller. The main idea here is, we ask the LLM to make the decision of whether to push the cart to the right or the left. This follows a zero shot learning strategy where the only prior information given to the agent is a brief context of the problem and the current state representation. We also added to the context the previous 15 (state, decision) tuples with the idea of giving the agent a time reference.

We use two prompting strategies, 1. the agent is asked to only respond with the decision, and 2. the agent to respond to the user with its thought process and the decision following the idea of [CoT](https://arxiv.org/abs/2201.11903).

The reward achieved by the agent when prompted to respond only with the decision:

+ Mean 22.54.
+ Standard deviation: 16.768076812801162.
+ Max: 109.0.
+ Min: 8.0.
+ Total episodes with a duration gte 500 is 0.
+ Sample size (total episodes): 100.

log transformation of the reward:

+ Mean: 2.895682578907467.
+ Standard deviation: 0.636933386411483.

The reward achieved by the agent when prompted to respond with its thought process and decision (using CoT strategy):

+ Mean: 20.85.
+ Standard deviation: 13.270550101634825.
+ Max: 81.0.
+ Min: 8.0.
+ Total episodes with a duration gte 500 is 0.
+ Sample size (total episodes): 100.

log transformation of the reward:

+ Mean: 2.8660452369702067.
+ Standard deviation: 0.5690782998841872.

## Experiment results

H0: mean_agent_one == mean_agent_two
Ha: mean_agent_one != mean_agent_two

### LLM vs. Random

P value and statistical significance:

The two-tailed P value equals 0.9492

Confidence interval:
The mean of Group One minus Group Two equals -0.005009763048935856
95% confidence interval of this difference: From -0.159862166088429650 to 0.149842639990557940

Intermediate values used in calculations:
t = 0.0638
df = 198
standard error of difference = 0.079

### Boxes vs. Random

P value and statistical significance:

The two-tailed P value is less than 0.0001

Confidence interval:
The mean of Group One minus Group Two equals -1.976122374384382300
95% confidence interval of this difference: From -2.175149470153931600 to -1.777095278614833000

Intermediate values used in calculations:
t = 19.5800
df = 198
standard error of difference = 0.101

### Conclusion

With a significance level of 5% we can conclude that LLMs are not much better than random policies when it comes to balance a pole in a moving cart. Also, we can conclude that the Boxes agent is significantly different from a random policy.

## Project Setup

Create a virtual environment using virtualenv (make sure to use a python version >= 3.10) and run the following commands after activating your virtual env.

```sh
pip install -r requirements.txt
pip install -e .
```

### If using vscode and your editor shows is not able to find the project dependencies

Follow the instructions [here](https://code.visualstudio.com/docs/python/environments#_working-with-python-interpreters) to change the python interpreter used by the editor. Make sure you select the interpreter of the virtual environment you created when installing the project.

## Run

### To execute the Boxes Model
```sh
python run.py --train=True --render_mode="human"
```

### To execute Language Agent
Make sure to add OpenAI's API key to the .env file.
```sh
echo "OPEN_AI_API_KEY=${OPEN_AI_API_KEY}" > .env
```

Run python project selecting the model backend you want to use ("gpt-4" or "gpt-3.5-turbo").

```sh
python run.py --backend="gpt-3.5-turbo" --render_mode="human" --save_chat=True --chat_file=messages.txt --prompt_file=data/prompt-3.txt
```
