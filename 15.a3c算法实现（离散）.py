import torch
import gym
import numpy as np
import os

import multiprocessing as mp
from multiprocessing import Manager
from multiprocessing import Pool

torch.set_default_dtype(torch.float64)

class Stack_queue:
    def __init__(self):
        self.list = []
    
    def get(self):
        o = self.list[-1]
        del self.list[-1]
        return o
    
    def put(self, o):
        self.list.append(o)
    
    def qsize(self):
        return len(self.list)

class memory:
    def __init__(self, env, length=2000, gamma=0.9):
        self.env = env
        self.length = length
        self.gamma = 0.9

        self.state_dimension = self.env.observation_space.shape[0]
        try:self.action_dimension = self.env.action_space.shape[0] 
        except:self.action_dimension = 1
        self.data = Stack_queue()
        
    def put(self, state, action, reward, next_state):
        self.data.put(np.hstack((state, action, reward, next_state)))
        
    def get(self):
        return self.data.get()

    def preprocess(self):
        length_real = self.data.qsize()
        self.state = np.zeros((length_real, self.state_dimension))
        self.action = np.zeros((length_real, self.action_dimension), dtype=np.int64)
        self.reward = np.zeros((length_real, 1))
        self.next_state = np.zeros((length_real, self.state_dimension))
        value_last = 0
        for i in range(length_real)[::-1]:
            transaction = self.data.get()
            state = transaction[:self.state_dimension]
            action = transaction[self.state_dimension:
                                 self.state_dimension+self.action_dimension]
            value = transaction[self.state_dimension+self.action_dimension:
                                                     self.state_dimension+self.action_dimension+1]
            next_state = transaction[-self.state_dimension:]

            value_last = value + self.gamma * value_last
            self.state[i] = state
            self.reward[i] = value_last
            self.action[i] = action
            self.next_state[i] = next_state

class actor:
    def __init__(self, env, hidden_dimension=30, learning_rate=1e-2, delta=-0.2):
        self.env = env
        self.hidden_dimension = hidden_dimension
        self.learning_rate = learning_rate
        self.delta = delta

        self.state_dimension = self.env.observation_space.shape[0]
        try:self.action_dimension = self.env.action_space.shape[0]
        except:self.action_dimension = self.env.action_space.n

        self.model = self.__create_network()

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.entropy = torch.nn.CrossEntropyLoss(reduction='none')
    
    def learn(self, state, action, advantage):
        state = torch.from_numpy(state)
        action = torch.tensor(action).squeeze()
        advantage = torch.tensor(advantage)
        action_probablity_value = self.model(state)
    
        loss = self.entropy(action_probablity_value, action) * advantage.squeeze()
        action_prob = torch.nn.functional.softmax(action_probablity_value, dim=-1)
        entropy = torch.sum(-torch.log(action_prob) * action_prob, axis=1)
        
        #purpose : entropy increase
        loss += self.delta * (-entropy)
        
        loss = torch.mean(loss)
        self.optimizer.zero_grad()
        loss.backward()
        # self.optimizer.step()
        return loss.item(), self.model.parameters()

    def output_action(self, state):
        state = torch.from_numpy(state)
        action_probablity_value = self.model(state)
        action_probablity_value = torch.nn.functional.softmax(action_probablity_value, dim=-1)
        return np.random.choice(np.arange(self.action_dimension),
                                p=action_probablity_value.tolist())
        
    def __create_network(self):
        return torch.nn.Sequential(torch.nn.Linear(self.state_dimension, self.hidden_dimension),
                                                             torch.nn.ReLU(),
                                                             torch.nn.Linear(self.hidden_dimension, self.action_dimension)
                                                             )

class critic:
    def __init__(self, env, hidden_dimension=20, learning_rate=1e-2, gamma=0.9):
        self.env = env
        self.hidden_dimension = hidden_dimension
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.state_dimension = self.env.observation_space.shape[0]
        try:self.action_dimension = self.env.action_space.shape[0]
        except:self.action_dimension = 1

        self.model = self.__create_network()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

    def learn(self, state, reward, next_state):
        state = torch.from_numpy(state)
        reward = torch.tensor(reward)
        next_state = torch.from_numpy(next_state)

        # value_next = self.model(next_state).detach()
        value = self.model(state)
        advantage = reward - value

        loss = torch.square(advantage)
        loss = torch.mean(loss)

        self.optimizer.zero_grad()
        loss.backward()
        # self.optimizer.step()
        return advantage.tolist(), self.model.parameters()
        
    def __create_network(self):
        return torch.nn.Sequential(torch.nn.Linear(self.state_dimension, self.hidden_dimension),
                                                 torch.nn.ReLU(),
                                                 torch.nn.Linear(self.hidden_dimension,
                                                                  self.action_dimension)
                                                 )

class agent_actor_critic(mp.Process):
    def __init__(self, env_name, epoch, q_in, q_out):
        mp.Process.__init__(self)
        self.env_name = env_name
        self.epoch = epoch
        self.q_in = q_in
        self.q_out = q_out
        
        self.env = gym.make(self.env_name)
        self.critic_network = critic(self.env)
        self.actor_network = actor(self.env)
        self.memory = memory(self.env)

    def run(self):
        self.learn()

    def learn(self):
        print('task start, pid: %d'%os.getpid())
        for i in range(self.epoch):
            self.reset_param()
            state = self.env.reset()
            reward_total = 0
            while True:
                action = self.actor_network.output_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.memory.put(state, action ,reward, next_state)
                state = next_state
                reward_total += reward
                if done:
                    # print(reward_total)
                    break
            self.memory.preprocess()
            #self.memory.next_state is NOT used here
            advantage, critic_param = self.critic_network.learn(self.memory.state,
                                                                self.memory.reward,
                                                                self.memory.next_state)
            
            loss, actor_param = self.actor_network.learn(self.memory.state,
                                                         self.memory.action, advantage)
    
            self.update_param()
            #print('reward_total',reward_total)
    
    def reset_param(self):
        critic_net, actor_net = self.q_in.get()
        self.critic_network.model.load_state_dict(
                critic_net.state_dict())
        self.actor_network.model.load_state_dict(
                actor_net.state_dict())

    def update_param(self):
        critic_grad = []
        actor_grad = []
        for each in self.critic_network.model.parameters():
            critic_grad.append(each.grad)
        for each in self.actor_network.model.parameters():
            actor_grad.append(each.grad)
        self.q_out.put((critic_grad,
                        actor_grad))

class agent_master(mp.Process):
    def __init__(self, env_name, epoch, work_cnt, q_in, q_out):
        mp.Process.__init__(self)
        self.env = gym.make(env_name)
        self.epoch = epoch
        self.work_cnt = work_cnt
        self.q_in = q_in
        self.q_out = q_out
        
        self.critic_network = critic(self.env)
        self.actor_network = actor(self.env)
    
    def run(self):
        print('master start, pid: %d'%os.getpid())

        for i in range(self.epoch):
            for _ in range(self.work_cnt):
                self.put()
            for _ in range(self.work_cnt):
                self.get()
            
            self.critic_network.optimizer.step()
            self.actor_network.optimizer.step()

            if (i+1) % 10 == 0:
                self.perform_test(i)
  
    def perform_test(self, i):
        reward_total = 0
        state = self.env.reset()
        while True:
            action = self.actor_network.output_action(state)
            next_state, reward, done, info = self.env.step(action)
            state = next_state
            reward_total += reward
            if done :
                break
        print('%d, performance : reward is %d'%(i, reward_total))

    def put(self):
        self.q_in.put((self.critic_network.model,
                       self.actor_network.model))

    def get(self):
        critic_grad, actor_grad = self.q_out.get()
        
        self.critic_network.optimizer.zero_grad()
        self.actor_network.optimizer.zero_grad()

        for lo_grad, glo_model in zip(critic_grad, 
                                      self.critic_network.model.parameters()):
            if glo_model._grad is None:
                glo_model._grad = lo_grad
            else:
                glo_model._grad += lo_grad
        for lo_grad, glo_model in zip(actor_grad, 
                                      self.actor_network.model.parameters()):
            if glo_model._grad is None:
                glo_model._grad = lo_grad
            else:
                glo_model._grad += lo_grad
            


class algorithm_a3c:
    def __init__(self, env_name='CartPole-v0', worker_cnt=4, epoch=500):
        self.env_name = env_name
        self.worker_cnt = worker_cnt
        self.epoch = epoch
        
    def start_task(self):
        slaver = agent_actor_critic(self.env_name, self.epoch,
                                    self.q_in, self.q_out)
        slaver.run()
    
    def start_master(self):
        master = agent_master(self.env_name, self.epoch,
                              self.worker_cnt,
                              self.q_in, self.q_out)
        master.run()

    def start_execute(self):
        self.q_in = Manager().Queue()
        self.q_out = Manager().Queue()
        pool = Pool(self.worker_cnt+1)
        for i in range(self.worker_cnt):
            pool.apply_async(self.start_task,
                             args=( ),
                             )
        pool.apply_async(self.start_master,
                         args=( ),
                         )
        pool.close()
        pool.join()
        print('this is the end of the program.')

if __name__ == '__main__':
        torch.set_default_dtype(torch.float64)
        
        algorithm_test = algorithm_a3c()
        algorithm_test.start_execute()


