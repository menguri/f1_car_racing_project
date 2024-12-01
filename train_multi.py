import gymnasium as gym
import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms as T
import torch.nn.functional as F
from collections import namedtuple, deque
from itertools import count
import random
import math
import pickle
from tqdm import tqdm
import wandb
import yaml

# For parallel Computing Code
import os
import pickle
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def image_preprocessing(img):
    img = cv2.resize(img, dsize=(84, 84))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img

class CarEnvironment(gym.Wrapper):
    def __init__(self, env, skip_frames=2, stack_frames=4, no_operation=5, **kwargs):
        super().__init__(env, **kwargs)
        self._no_operation = no_operation
        self._skip_frames = skip_frames
        self._stack_frames = stack_frames

    def reset(self):
        observation, info = self.env.reset()

        for i in range(self._no_operation):
            observation, reward, terminated, truncated, info = self.env.step(0)

        observation = image_preprocessing(observation)
        self.stack_state = np.tile(observation, (self._stack_frames, 1, 1))
        return self.stack_state, info


    def step(self, action):
        total_reward = 0
        for i in range(self._skip_frames):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        observation = image_preprocessing(observation)
        self.stack_state = np.concatenate((self.stack_state[1:], observation[np.newaxis]), axis=0)
        return self.stack_state, total_reward, terminated, truncated, info

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_features = 32 * 9 * 9

    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=4, stride=2),
        nn.ReLU(),
    )

    self.fc = nn.Sequential(
        nn.Linear(self._n_features, 256),
        nn.ReLU(),
        nn.Linear(256, out_channels),
    )


    def forward(self, x):
        x = self.conv(x)
        x = x.view((-1, self._n_features))
        x = self.fc(x)
        return x

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self, action_space, batch_size=256, gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=1000, lr=0.001, use_vit=False):
        super(DQN, self).__init__()
        self._n_observation = 4
        self._n_actions = 5
        self._action_space = action_space
        self._batch_size = batch_size
        self._gamma = gamma
        self._eps_start = eps_start # For epsilon greedy choice
        self._eps_end = eps_end     # For epsilon greedy choice
        self._eps_decay = eps_decay # For epsilon greedy choice
        self._lr = lr
        self._total_steps = 0
        self._evaluate_loss = []
        self.use_vit = use_vit  
        
        # 네트워크 초기화
        self.network = self.build_network()
        self.target_network = self.build_network()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = optim.AdamW(self.network.parameters(), lr=self._lr, amsgrad=True)
        self._memory = ReplayMemory(10000)  
        
        # 초기 네트워크 파라미터 저장 (Shrink & Perturb용)
        self.initial_params = self.network.state_dict()  
      
    def build_network(self):
        if self.use_vit:
            # ViT 모델을 weights 매개변수를 사용하여 불러옴
            weights = ViT_B_16_Weights.DEFAULT
            model = vit_b_16(weights=weights)
            model.heads.head = nn.Linear(model.heads.head.in_features, self._n_actions)
            return model.to(device)
        else:
            return CNN(self._n_observation, self._n_actions).to(device)

    """
    This function is called during training & evaluation phase when the agent
    interact with the environment and needs to select an action.

    (1) Exploitation: This function feeds the neural network a state
    and then it selects the action with the highest Q-value.
    (2) Evaluation mode: This function feeds the neural network a state
    and then it selects the action with the highest Q'-value.
    (3) Exploration mode: It randomly selects an action through sampling

    Q -> network (policy)
    Q'-> target network (best policy)
    """
    def select_action(self, states, evaluation_phase=False):
        """
        Determine the action to take for a batch of states.
        Exploration & Exploitation.

        Parameters:
        - states: A batch of states (tensor of shape [batch_size, ...])
        - evaluation_phase: If True, perform evaluation (no exploration).

        Returns:
        - A tensor of actions (shape [batch_size, 1]).
        """
        # Calculate epsilon threshold for exploration vs exploitation
        sample = random.random()
        eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(-1. * self._total_steps / self._eps_decay)
        self._total_steps += 1

        if evaluation_phase:
            # Choose actions with the highest Q-value (evaluation phase)
            with torch.no_grad():
                return self.target_network(states).max(1).indices.view(-1, 1)  # Shape: [batch_size, 1]
        elif sample > eps_threshold:
            # Exploitation: Choose actions with the highest Q-value
            with torch.no_grad():
                return self.network(states).max(1).indices.view(-1, 1)  # Shape: [batch_size, 1]
        else:
            # Exploration: Choose random actions
            batch_size = states.size(0)
            random_actions = [self._action_space.sample() for _ in range(batch_size)]
            return torch.tensor(random_actions, device=device, dtype=torch.long).view(-1, 1)  # Shape: [batch_size, 1]
        

    def train(self, replay_ratio=1):
        """
        Training Step
        
        1.  sample memory from replay memory (batch size)
        2.  non_final_mask = select states that are non-terminal
            non_final_next_states = make tensor with non-terminal states
            state_batch, action_batch, reward_batch = concat each data into tensor
        3.  Calculate Q Value in current state using policy network.
            Extract Q Value only about selected action in current state.
        4.  Calculate Next State maximum Q value using target_network
            Update only if next state exists.
        5.  Calculate Expected Q Value = reward + gamma * max(Q')
        6.  Calculate Loss(Minimize the difference between Expected Q value and calculated Q value in current state.
        7.  Update with backpropagation.
        """
        # if length of replay memory is smaller than batch size, stop
        if len(self._memory) < self._batch_size:
            return
        
        for _ in range(replay_ratio):
            # Initializing our memory
            transitions = self._memory.sample(self._batch_size)
            # Initializing our batch
            batch = Transition(*zip(*transitions))
            
            # Saving in a new tensor all the indices of the states that are non terminal
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Feeding our Q network the batch with states and then we gather the Q values of the selected actions
            # state_action_values = self.network(state_batch).gather(1, action_batch)
            
            action_batch = action_batch.view(-1, 1)  # [batch_size] -> [batch_size, 1]
            state_action_values = self.network(state_batch).gather(1, action_batch)
            # We then, for every state in the batch that is NOT final, we pass it in the target network to get the Q'-values and choose the max one
            next_state_values = torch.zeros(self._batch_size, device=device)
            with torch.no_grad():
                next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values

            # Computing the expecting values with: reward + gamma * max(Q')
            expected_state_action_values = (next_state_values * self._gamma) + reward_batch
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
            self.optimizer.step()
            self._evaluate_loss.append(loss.item())
            
    def shrink_and_perturb(self, alpha=0.9):
        current_params = self.network.state_dict()
        perturbed_params = {k: alpha * current_params[k] + (1 - alpha) * self.initial_params[k] for k in current_params}
        self.network.load_state_dict(perturbed_params)
        
    def copy_weights(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def get_loss(self):
        return self._evaluate_loss

    def save_model(self, i):
        torch.save(self.target_network.state_dict(), f'./model/multi/model_weights_{i}.pth')

    def load_model(self, version, i):
        self.target_network.load_state_dict(torch.load(f'model/{version}/model_weights_{i}.pth', map_location=device))
    
def reset_envs_with_random_seeds(envs):
    """
    각 환경에 랜덤 시드를 설정하고 초기화.
    """
    new_seeds = [np.random.randint(1000) for _ in range(len(envs))]
    states = []
    for env, seed in zip(envs, new_seeds):
        env.np_random = np.random.default_rng(seed)  # 랜덤 시드 재설정
        state, _ = env.reset()  # 환경 초기화
        states.append(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
    return states

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 설정값 가져오기
training_params = config['training']
env_params = config['environment']

# 초기 시드 설정
np.random.seed(42)  # NumPy 랜덤 시드 고정
torch.manual_seed(42)  # PyTorch 랜덤 시드 고정

# 환경 및 파라미터 설정
env_count = env_params['env_count']
TU = training_params['TU']
TC = training_params['TC']
TR = training_params['TR']
map_reset_interval = training_params['map_reset_interval']
replay_ratio = training_params['replay_ratio']
episodes = training_params['episodes']
alpha = training_params['alpha']
save_point = training_params['save_point']


version=f"multi_{map_reset_interval}_{replay_ratio}_{alpha}_{TU}_{TR}"

def init_wandb():
    # wandb 초기화
    wandb.init(
        project="F1024 Project (Car-racing)",  # 프로젝트 이름
        name=version,          # 실험 이름
        config={               # 설정값 로깅
            "TU": TU,
            "TC": TC,
            "TR": TR,
            "map_reset_interval": map_reset_interval,
            "replay_ratio": replay_ratio,
            "episodes": episodes,
            "alpha": alpha,
            "env_count": env_count
        }
    )

def init_distributed(rank, world_size):
    """Initialize the distributed environment variables."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Master node IP
    os.environ['MASTER_PORT'] = '29500'     # Master node port
    os.environ['WORLD_SIZE'] = str(world_size)  # Total number of processes
    os.environ['RANK'] = str(rank)          # Rank of the current process
    os.environ['LOCAL_RANK'] = str(rank % torch.cuda.device_count())  # GPU ID for the process
    
    dist.init_process_group(backend='nccl', init_method='env://')  # Initialize process group

def cleanup_distributed():
    """Clean up the distributed process group."""
    dist.destroy_process_group()

def format_gpu_memory(rank):
    """Format GPU memory usage for display."""
    allocated = torch.cuda.memory_allocated(rank) / 1024**2 
    reserved = torch.cuda.memory_reserved(rank) / 1024**2 
    return f"Alloc {allocated:.1f}MB, Reserved {reserved:.1f}MB"

def distributed_training(rank, world_size):
    """Perform distributed training."""
    
    if rank == 0:
        init_wandb()
        
    init_distributed(rank, world_size)  # Initialize distributed environment
    torch.cuda.set_device(rank % torch.cuda.device_count())  # Set CUDA device for the current process

    # Set up environments and agent
    env_count_per_proc = env_count // world_size
    envs = [CarEnvironment(gym.make('CarRacing-v2', continuous=False)) for _ in range(env_count_per_proc)]
    agent = DQN(envs[0].action_space, use_vit=False).to(rank)
    agent = DDP(agent, device_ids=[rank])

    # Training statistics
    rewards_per_episode = []
    episode_duration = []
    average_episode_loss = []

    # Initialize tqdm
    progress_bar = tqdm(range(1, episodes + 1), desc=f"Training", leave=False, total = episodes)

    for episode in progress_bar:
        if episode % map_reset_interval == 1:
            states = reset_envs_with_random_seeds(envs)
        else:
            states = [env.reset()[0] for env in envs]
            states = [torch.tensor(s, dtype=torch.float32, device=rank).unsqueeze(0) for s in states]

        episode_total_rewards = [0] * len(envs)
        episode_losses = []
        episode_durations = [0] * len(envs)
        done_mask = [False] * len(envs)

        for t in range(500):
            valid_indices = [i for i in range(len(envs)) if not done_mask[i]]
            valid_states = [states[i] for i in valid_indices if states[i] is not None]

            if len(valid_states) == 0:
                break
            states_tensor = torch.cat(valid_states, dim=0)

            actions = [
                agent.module.select_action(states_tensor[idx].unsqueeze(0))
                for idx in range(len(valid_indices))
            ]

            for idx, env_idx in enumerate(valid_indices):
                next_state, reward, terminated, truncated, _ = envs[env_idx].step(actions[idx].item())
                reward = torch.tensor([reward], device=rank)
                done_flags = terminated or truncated

                if not done_flags:
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=rank).unsqueeze(0)
                    agent.module._memory.push(states[env_idx], actions[idx], next_state, reward)
                    states[env_idx] = next_state
                    episode_total_rewards[env_idx] += reward
                    episode_durations[env_idx] += 1
                else:
                    states[env_idx] = None
                    done_mask[env_idx] = True

            if t % TU == 0:
                agent.module.train(replay_ratio)
                loss_values = agent.module.get_loss()
                if loss_values:
                    episode_losses.append(loss_values)

            if all(done_mask):
                avg_reward = sum(episode_total_rewards) / len(envs)
                avg_loss = sum([sum(losses) / len(losses) for losses in episode_losses if losses]) / len(envs)
                avg_duration = sum(episode_durations) / len(envs)

                # Distributed averaging
                avg_reward_tensor = torch.tensor(avg_reward, device=rank)
                avg_loss_tensor = torch.tensor(avg_loss, device=rank)
                avg_duration_tensor = torch.tensor(avg_duration, device=rank)
                
                # Reduce across all ranks
                dist.all_reduce(avg_reward_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(avg_duration_tensor, op=dist.ReduceOp.SUM)
                
                # Compute average by dividing by world size
                avg_reward_tensor /= world_size
                avg_loss_tensor /= world_size
                avg_duration_tensor /= world_size
                
                # Convert tensors to Python scalars
                avg_reward = avg_reward_tensor.item()
                avg_loss = avg_loss_tensor.item()
                avg_duration = avg_duration_tensor.item()
                
                rewards_per_episode.append(avg_reward)
                average_episode_loss.append(avg_loss)
                episode_duration.append(avg_duration)
                
                # Log to WandB only on Rank 0
                if rank == 0:
                    wandb.log({
                        "Episode": episode,
                        "Average Reward": avg_reward,
                        "Average Loss": avg_loss,
                        "Average Duration": avg_duration
                    })
                
                gpu_mem_info = format_gpu_memory(rank)
                progress_bar.set_postfix({
                    "Avg Reward": f"{avg_reward:.2f}",
                    "Avg Loss": f"{avg_loss:.2f}",
                    "Avg Duration": f"{avg_duration:.2f}",
                    "GPU": gpu_mem_info
                })
                
                break

        if episode % TC == 0:
            dist.barrier()  # Synchronize before copying weights
            agent.module.copy_weights()

        if episode % TR == 0:
            agent.module.shrink_and_perturb(alpha)

        if episode % save_point == 0 and rank == 0:
            print(f"{episode} episodes done")
            agent.module.save_model(episode)
            with open('./multi_statistics.pkl', 'wb') as f:
                pickle.dump((episode_duration, rewards_per_episode, average_episode_loss), f)

    if rank == 0:
        agent.module.save_model(episode)
        with open('./multi_statistics.pkl', 'wb') as f:
            pickle.dump((episode_duration, rewards_per_episode, average_episode_loss), f)

    cleanup_distributed()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # Total number of GPUs available
    print(f"Total GPU Count: {world_size}")
    torch.multiprocessing.spawn(distributed_training, args=(world_size,), nprocs=world_size, join=True)