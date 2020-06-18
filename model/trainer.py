import torch
import torch.nn as nn
import numpy as np
from utils import collect_episodes

class ObserverTrainer:

    def __init__(self,
                 env,
                 observer,
                 writer=None):
        self.env = env
        self.observer = observer
        self.writer = writer
    
    def train(self, n_epochs, batch_size, verbose=False):
        
        for i in range(n_epochs):
            data = collect_episodes(N=batch_size, env=self.env, max_steps=1)
            states = torch.stack(tuple([x['states'][0] for x in data])) # first step across batch
            actions = torch.stack(tuple([torch.tensor(x['actions'][0]) for x in data]))
            next_states = torch.stack(tuple([x['next_states'][0] for x in data]))

            X = torch.cat((states, next_states), dim=1)
            loss = self.observer.train(X, actions, loss_fn=nn.CrossEntropyLoss())
            if self.writer is not None:
                self.writer.add_scalar('observer_train_loss', loss, i)
            
            if (verbose) and (i % (n_epochs/10.0)==0):
                print('epoch: %d  |  loss: %f' % (i, loss))
        
class AgentTrainer:

    def __init__(self,
                 env,
                 agent,
                 writer=None):

        self.env = env
        self.agent = agent
        self.writer = writer
        self.current_epoch = None

    def train(self, n_epochs, log_interval, n_test_epochs, max_ep_steps, test=True, verbose=False):
    
        performance = {'episode':[], 'avg_step':[], 'avg_step_diff':[], 'avg_action_optimal':[]}

        for i_ep in range(n_epochs):
            self.current_epoch = i_ep
            
            # single episode rollout
            data = collect_episodes(N=1, env=self.env, agent=self.agent, max_steps=max_ep_steps)
            
            for r in data[0]['rewards']:
                self.agent.rewards.append(r)

            # perform backprop
            loss = self.agent.train()
            if self.writer is not None:
                self.writer.add_scalar('agent_train_loss', loss, i_ep)
            
            ep_reward = np.sum(data[0]['rewards'])

            # log results
            if (test) and (i_ep % log_interval == 0):
                
                p = self.test(n_test_epochs, max_ep_steps)
                performance['episode'].append(i_ep)
                performance['avg_step'].append(np.mean(p['agent_step']))
                performance['avg_step_diff'].append(np.mean(p['step_diff']))
                performance['avg_action_optimal'].append(np.mean(p['action_optimal'], 0))

                if verbose:
                    print('Last episode {}\t|\t Average step: {:.2f} \t|\tAverage (step-optim): {:.2f} \t|\t % (step-optim)<=5: {:.2f} \t|\t % (step-optim)<=3: {:.2f}'.format(
                          i_episode, 
                          np.mean(p['agent_step']),
                          np.mean(p['step_diff']),
                          np.mean(np.array(p['step_diff'])<=5),
                          np.mean(np.array(p['step_diff'])<=3)))

        return self.agent, performance

    def test(self, n_test_epochs, max_ep_steps):
        self.agent.test_mode = True
        
        performance = {'test_ep':np.zeros(n_test_epochs), 
                       'agent_step':np.zeros(n_test_epochs), 
                       'step_diff':np.zeros(n_test_epochs), 
                       'action_optimal':np.zeros((n_test_epochs, 2))}
        
        data = collect_episodes(N=n_test_epochs, env=self.env, agent=self.agent, max_steps=max_ep_steps)

        for i in range(n_test_epochs):
            
            ep = data[i]
            t = len(ep['dones'])
            actions = ep['actions']
            shortest_path = self.env.world.shortest_path(ep['env_start'], ep['env_goal'])
            n = shortest_path['n_step']
            optim_actions = shortest_path['actions']
            
            performance['test_ep'][i] = i
            performance['agent_step'][i] = t
            performance['step_diff'][i] = t-n
            # how are the first two actions corresponding to the optimal actions (in cases steps==2)
            performance['action_optimal'][i] = np.array([actions[0]==optim_actions[0], actions[1]==optim_actions[1]])
        
        self.agent.test_mode = False
       
        if self.writer is not None:
            self.writer.add_scalar('agent_test_avg_step', 
                                   np.mean(performance['agent_step']), self.current_epoch)
            self.writer.add_scalar('agent_test_avg_step_diff', 
                                   np.mean(performance['step_diff']), self.current_epoch)
            self.writer.add_scalar('agent_test_avg_first_action_optimal', 
                                   np.mean(performance['action_optimal'], 0)[0], self.current_epoch)
            self.writer.add_scalar('agent_test_avg_second_action_optimal', 
                                   np.mean(performance['action_optimal'], 0)[1], self.current_epoch)
    
        return performance
