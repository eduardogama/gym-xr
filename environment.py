import gym
import os

import numpy as np
import pandas as pd


# Defaults
# Adjusted based on min/max values of dataset
MIN_RTT = 0.0  # corresponds to 0.0 ms v1: 0 - 30/v2: 0 - 40
MAX_RTT = 40.0  # corresponds to 30.0 ms

MIN_LATENCY = 1.0  # corresponds to min access latency of node - 1.0
MAX_LATENCY = 10.0  # corresponds to max access latency of node - 10.0

MIN_JITTER = 0.0  # v1: 0 - 513/v2: 0 - 229
MAX_JITTER = 229.0

MIN_UL = 0.0  # v1: 0 - 90/v2: 0 - 93
MAX_UL = 93.0

MIN_DL = 1.0  # v1: 1 - 620/v2: 1 - 516
MAX_DL = 516.0

MIN_PKT_LOSS_RATE = 0.0
MAX_PKT_LOSS_RATE = 1.0

MIN_OBS = 0.0
MAX_OBS = 1000.0

PROCESSING_DELAY = 2.0  # 2.0 ms
MIN_PROC = 0.0
MAX_PROC = 200.0  # 2.0 * 100 steps = 200.0 ms

# Dataframe column names
DF_COLUMN_PKT_LOSS_RATE = "pkt_loss_rate"
DF_COLUMN_RTT_AVG = "rtt_avg"
# DF_COLUMN_RTT_MEDIAN = "rtt_median"
# DF_COLUMN_RTT_STD = "rtt_std"
DF_COLUMN_RTT_Q90 = "rtt_q90"
DF_COLUMN_RSSI = "rssi"
DF_COLUMN_RSRQ = "rsrq"
DF_COLUMN_RSRP = "rsrp"
DF_COLUMN_UL = "speedtest_ul_mbps"
DF_COLUMN_DL = "speedtest_dl_mbps"
DF_COLUMN_LATENCY = "speedtest_latency"
DF_COLUMN_JITTER = "speedtest_jitter"

# Defaults for Weights
LATENCY_WEIGHT = 1.0
GINI_WEIGHT = 0.0
COST_WEIGHT = 0.0
BANDWIDTH_WEIGHT = 0.0

FACTOR = 1.0
SEED = 42
PATH_CSV_FILES = "data/train/v1/nodes/"


class Environment(gym.Env):
    """ NNE Scheduling env in Kubernetes - an OpenAI gym environment"""
    metadata = {'render.modes': ['human', 'ansi', 'array']}

    def __init__(self, num_nodes=4, reward_function='multi', strategy="cost/", path="data/train/v1/nodes/",
                    cost_weight=1.0, bandwidth_weight=0.0, latency_weight=0.0, gini_weight=0.0, episodes=100,
                    episode_length=100, call_duration_r=1):
        super(Environment, self).__init__()

        self.frame_idx = 0
        
        self.timer_remote_tasks = pd.read_csv(
            f'/{os.getcwd()}/timer_server_gpu_4k.csv',
            sep=' '
        )

        self.timer_local_tasks = pd.read_csv(
            f'/{os.getcwd()}/timer_server_cpu_hd.csv',
            sep=' '
        )

        self.trace = np.array(
            pd.read_csv(
                f'{os.getcwd()}/trace.csv'
            ).values
        ).flatten()

        self.obs_low = np.array([0] * 11)
        self.obs_high = np.array([10e6, 100, 100, 500, 5, 10e6, 10e6, 10e6, 10e6, 10e6, 10e6])
        
        self.num_nodes = num_nodes
        self.reward_function = reward_function
        self.strategy = strategy
        self.path = path
        self.cost_weight = cost_weight
        self.bandwidth_weight = bandwidth_weight
        self.latency_weight = latency_weight
        self.gini_weight = gini_weight
        self.episodes = episodes
        self.episode_length = episode_length
        self.call_duration_r = call_duration_r

        self.action_space = gym.spaces.Discrete(num_nodes)
        self.observation_space = gym.spaces.Box(low=MIN_OBS, high=MAX_OBS, shape=(num_nodes, 1), dtype=float)

    def step(self, action):
        # Note: sizes are in bytes, times are in seconds
        frame_size = self.frame_sizes[action][self.frame_idx]
        
        # Simulate environment response
        processing_time = self.get_processing_time(frame_size, action)
        edge_node_load = self.get_edge_node_load() if action == 1 else 0

        delay = 0

        while frame_size > 1e-8:  # floating number business

            throuput = self.trace[self.curr_t_idx] / 8.0 * 1e6  # bytes/second            

            frame_time_used = min(self.frame_time_left, frame_size / throuput)

            frame_size -= throuput * frame_time_used
            self.frame_time_left -= frame_time_used
            delay += frame_time_used

            if self.frame_time_left == 0:
                self.frame_t_idx += 1
                if self.curr_t_idx == len(self.trace[1]):
                    self.curr_t_idx = 0

                self.frame_time_left = self.get_frame_time(self.trace, self.curr_t_idx)

        # store action for future bitrate change penalty
        self.past_action = action

        reward = self.bitrate_map[action] - 4.3 * delay
        done = (self.chunk_idx == self.total_num_chunks)
        info = {'bitrate': self.bitrate_map[action]}
        self.frame_idx += 1

        return self.observe(), reward, done, info
    

    def get_chunk_time(self, trace, t_idx):
        if t_idx == len(trace[0]) - 1:
            return 1  # bandwidth last for 1 second
        else:
            return trace[0][t_idx + 1] - trace[0][t_idx]

    def get_processing_time(self, task, action):
        if action == 1:
            return self.timer_local_tasks[task]
        else:
            return self.timer_remote_tasks[task]

    def get_edge_node_load(self):
        return 0
    
    def observe(self):
        if self.frame_idx < self.total_num_chunks:
            valid_frame_idx = self.frame_idx
        else:
            valid_frame_idx = 0

        if self.past_action is not None:
            valid_past_action = self.past_action
        else:
            valid_past_action = 0

    def reset(self):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def configure(self, *args, **kwargs):
        pass

    def __str__(self):
        return f"Environment(num_nodes={self.num_nodes}, reward_function={self.reward_function}, strategy={self.strategy}, path={self.path}, cost_weight={self.cost_weight}, bandwidth_weight={self.bandwidth_weight}, latency_weight={self.latency_weight}, gini_weight={self.gini_weight}, episodes={self.episodes}, episode_length={self.episode_length}, call_duration_r={self.call_duration_r})"
    
    def __repr__(self):
        return f"Environment(num_nodes={self.num_nodes}, reward_function={self.reward_function}, strategy={self.strategy}, path={self.path}, cost_weight={self.cost_weight}, bandwidth_weight={self.bandwidth_weight}, latency_weight={self.latency_weight}, gini_weight={self.gini_weight}, episodes={self.episodes}, episode_length={self.episode_length}, call_duration_r={self.call_duration_r})"
    
    def __eq__(self, other):
        return self.__dict__ == other.__dict__
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))
    
    def __len__(self):
        return len(self.__dict__)
    
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)
    
    def __contains__(self, item):
        return item in self.__dict__
    
    def __getattr__(self, item):
        return self.__dict__[item]
    
    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __delattr__(self, item):
        del self.__dict__[item]

    def __getstate__(self):
        return self.__dict__