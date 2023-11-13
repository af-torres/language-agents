# Are Language Agents powerful enough?

Large Language Models (LLM) exhibit emergent abilities in tasks like planing and decision-making akin those of human beings, which has inspired the creation of agents that use LLM as their main controller. While many of the augmentations being added to this models today (tool usage, memory, prompting, etc.) improve the performance of language agents in a variety of tasks, it is still unclear how well they perform in classic control environments.

This project aims to test weather there is a statistically significant difference between the performance of LLM agents and some of the oldest RL models used to solve classic control problems. This with the objective of knowing if there is any merit at all in the LLM Agent methodology to RL. In my opinion (yes it is a highly opinionated project), it wouldn't be fair to compare LLM agents against models that leverage Deep Neural Networks to learn a control policy (in low dimensionality environments). 

Up to this point in time, we have developed a base agent called [Boxes](https://citeseerx.ist.psu.edu/document?doi=2f027193fb703d0af58ec382bd1438daff9417d7) to learn a control policy for the [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) environment. The results of this agent are:

+ Mean episode duration of 163.77.
+ Standard deviation for episode duration of 105.66814609900187.
+ Longest episode with a duration of 437.0.
+ Shortest episode with a duration of 11.0.
+ Total episodes with a duration gte 500 is 0.

The next step (coming soon) is to use an LLM as the controller in a zero shot learning context and perform a two sample t-test to compare if the LLM agent has a better performance over the Boxes agent. 

## Setup

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
python run.py --backend="gpt-4" --render_mode="human"
```
