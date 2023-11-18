import argparse
import logging
import os
import numpy as np

from agents.agent import CartPoleAgent, Policy
from agents.environment import CartPole, Reward
from agents.policies.boxes import Boxes
from agents.policies.language import LLM, Chat, ChatDumpToFile, RateLimit
from dotenv import dotenv_values
from typing import List
from openai import OpenAI


config = dotenv_values(".env")


log = logging.getLogger(__name__)
logging.basicConfig(level=int(config.get("LOGGING_LEVEL", logging.INFO)))


f = None # IO to dump data to. Used to save LLM messages.

def run(args) -> None:
    try:
        policy = loadPolicy(args)

        log.info("Starting policy evaluation...")
        results = play(args.eval_episodes, policy, renderMode=args.render_mode)

        eval(results)
        eval(np.log(results)) # hack to perform better t-tests
    except Exception as e:
        log.error(e, exc_info=True)
    finally:
        # clean up
        global f
        if not f is None:
            f.close()


def eval(results: List[Reward]) -> None:
    r = np.array(results)
    mean = np.mean(r)
    sd = np.std(r)
    max = np.max(r)
    min = np.min(r)
    up500 = np.sum(r >= 500)

    log.info(f"A total of {len(r)} episodes were played during evaluation.")
    log.info(f"""The policy under evaluation has:
+ A mean episode duration of {mean} with a standard deviation of {sd}.
+ The longest episode was {max}.
+ The shortest was {min}.
+ Total episodes with a duration gte 500 is {up500}.""")


def play(episodes: int, policy: Policy, renderMode: str | None = None) -> List[Reward]:
    agent = CartPoleAgent(policy)
    env = CartPole(renderMode)

    episodesReward = []
    for e in range(episodes):
        total = 0 
        while not env.terminated():
            env, reward = agent.action(env)
            total += reward

            if total % 100 == 0:
                log.info(f"total reward for current episode has reached {total}")

        log.info(f"episode {e} terminated with a total reward of {total}")

        env.reset()
        agent.terminate()
        episodesReward.append(total)

    return episodesReward  


def loadPolicy(args) -> Policy:
    if args.backend == "boxes":
        policy = Boxes()
        if args.train:
            log.info("Training demons to make decisions...")
            play(args.train_episodes, policy)
        
        return policy
    else:
        apiKey = config.get("OPEN_AI_API_KEY", None)
        if apiKey == None:
            raise Exception("OPEN_AI_API_KEY missing in .env file")
        
        client = OpenAI(api_key=apiKey)
        
        chat = None
        if args.save_chat:
            global f
            f = open(args.chat_file, "w")
            chat = ChatDumpToFile(client, args.backend, f)
        else:
            chat = Chat()

        promptFile = open(args.prompt_file, "r")
        prompt = promptFile.read()
        promptFile.close()

        policy = LLM(chat, prompt)
        policy = RateLimit(policy, args.rqm)
        
        return policy
    

def parseArgs() -> argparse.Namespace:
    args = argparse.ArgumentParser()

    args.add_argument(
        "--render_mode",
        choices=[None, "human"],
        default=None
    )
    args.add_argument(
        "--backend",
        type=str,
        choices=["gpt-4", "gpt-3.5-turbo", "boxes"],
        default="boxes"
    )
    args.add_argument(
        "--rqm",
        type=int,
        default=500,
        help="limit to the requests per minute to open ai api"
    )
    args.add_argument(
        "--save_chat",
        type=bool,
        default=False
    )
    args.add_argument(
        "--chat_file",
        type=str,
        default=os.path.join("data", "chat_history.txt")
    )
    args.add_argument(
        "--prompt_file",
        type=str,
        default=os.path.join("data", "prompt-1.txt")
    )
    args.add_argument(
        "--train",
        type=bool,
        default=False
    )
    args.add_argument(
        "--train_episodes",
        type=int,
        default=20000
    )
    args.add_argument(
        "--eval_episodes",
        type=int,
        default=100
    )

    return args.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    run(args)
