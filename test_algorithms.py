import numpy as np

from tqdm import tqdm
from envs.environment import Environment


def main():
    num_nodes = 4  # 4, 8, 12, 16, 32
    
    reward_function = 'multi'
    alg = 'dqn_deepsets'
    strategy = "cost/"
    path = "data/train/v2-nov-dec/nodes/"
    
    cost_weight = 1.0
    bandwidth_weight = 0.0
    latency_weight = 0.0
    gini_weight = 0.0

    episodes = 100
    episode_length = 100
    call_duration_r = 1


    env = Environment(num_nodes=num_nodes, reward_function=reward_function, strategy=strategy, path=path,
                        cost_weight=cost_weight, bandwidth_weight=bandwidth_weight, latency_weight=latency_weight, gini_weight=gini_weight, episodes=episodes,
                        episode_length=episode_length, call_duration_r=call_duration_r)

    for i in tqdm(range(episodes)):
        print(i)

        obs = env.reset()
        action_mask = env.action_masks()
        done = False

        while not done:
            action = np.random.choice(np.arange(num_nodes), p=action_mask)
            obs, reward, done, info = env.step(action)
            action_mask = env.action_masks()

            print(f"Action: {action} | Reward: {reward} | Done: {done}")
            return_ += reward
    env.close()


if __name__ == "__main__":
    main()
