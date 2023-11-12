import argparse
import logging
import numpy as np

from agents.agent import CartPoleAgent, Policy
from agents.environment import CartPole
from agents.policies.boxes import Boxes
from typing import List


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run(args):
    policy = None

    if args.train and args.backend == "boxes":
        log.info("Training demons to make decisions...")
        policy = Boxes()
        play(args.train_episodes, policy)
    
    log.info("Starting policy evaluation...")
    results = play(args.eval_episodes, policy, renderMode=args.render_mode)
    eval(results)


def eval(results: List[int]) -> None:
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


def play(episodes: int, policy: Policy, renderMode: str | None = None) -> List[int]:
    agent = CartPoleAgent(policy)
    env = CartPole(renderMode)

    episodesReward = []
    for _ in range(episodes):
        total = 0 
        while not env.terminated():
            env, reward = agent.action(env)
            total += reward

        env.reset()
        agent.terminate()
        episodesReward.append(total)

    return episodesReward  


def parse_args():
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
        "--temperature",
        type=float,
        default=0.7
    )
    args.add_argument(
        "--train",
        type=bool,
        default=False
    )
    args.add_argument(
        "--train_episodes",
        type=int,
        default=250000
    )
    args.add_argument(
        "--eval_episodes",
        type=int,
        default=100
    )

    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
