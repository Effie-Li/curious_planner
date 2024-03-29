import torch
import torch.nn as nn
import numpy as np
from utils import collect_episodes

class ObserverTrainer:

    def __init__(self,
                 env,
                 observer,
                 mode='ssa',
                 writer=None):
        
        # TODO: enable trainer constructing the observer from env state dim and action dim
        
        self.env = env
        self.observer = observer
        self.mode = mode
        self.writer = writer
    
    def train(self, n_epochs, log_interval, batch_size, checkpoint_interval=None, verbose=False):
        
        for i in range(n_epochs):
            data = collect_episodes(N=batch_size, env=self.env, max_steps=1)
            states = torch.stack(tuple([x['states'][0] for x in data])) # first step across batch
            actions = torch.stack(tuple([torch.tensor(x['actions'][0]) for x in data]))
            next_states = torch.stack(tuple([x['next_states'][0] for x in data]))
            
            if self.mode=='ss':
                # make y (next states) index label again
                next_states = torch.argmax(next_states, dim=-1)
                loss = self.observer.train(states, next_states, loss_fn=nn.CrossEntropyLoss())
            if self.mode=='ssa':
                X = torch.cat((states, next_states), dim=1)
                loss = self.observer.train(X, actions, loss_fn=nn.CrossEntropyLoss())
            elif self.mode=='sas':
                # make actions one-hot then concat
                actions = nn.functional.one_hot(torch.tensor(actions), self.env.action_dim).type('torch.FloatTensor')
                X = torch.cat((states, actions), dim=1)
                # make y (next states) index label again
                next_states = torch.argmax(next_states, dim=-1)
                loss = self.observer.train(X, next_states, loss_fn=nn.CrossEntropyLoss())
            
            if (self.writer is not None) and (i % log_interval == 0):
                self.writer.add_scalar('observer_train_loss', loss, i)
                if (checkpoint_interval is not None) and (i % checkpoint_interval == 0):
                    torch.save(self.observer.state_dict(), self.writer.get_logdir()+'/model_checkpoint_%d'%i)
            
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

    def train(self, n_epochs, log_interval, n_test_epochs, max_ep_steps, test=True, checkpoint_interval=None, verbose=False):
    
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
                if (checkpoint_interval is not None) and (i_ep % checkpoint_interval == 0):
                    torch.save(self.agent.state_dict(), self.writer.get_logdir()+'/model_checkpoint_%d'%i_ep)
            
            ep_reward = np.sum(data[0]['rewards'])

            # log results
            if (test) and (i_ep % log_interval == 0):

                # additional breakup testing
                min_dist = self.env.min_dist
                max_dist= self.env.max_dist
                for d in range(1, max_dist+1):
                    if d > 4:
                        break
                    else:
                        # test on d-step problems
                        self.env.min_dist=d
                        self.env.max_dist=d
                        p = self.test(self.env, n_test_epochs, max_ep_steps, writer_prefix=str(d)+'-step_')
                        self.writer.flush()

                # test on original env
                self.env.min_dist = min_dist
                self.env.max_dist = max_dist
                p = self.test(self.env, n_test_epochs, max_ep_steps, writer_prefix='')
                self.writer.flush()
                performance['episode'].append(i_ep)
                performance['avg_step'].append(np.mean(p['agent_step']))
                performance['avg_step_diff'].append(np.mean(p['step_diff']))
                performance['avg_action_optimal'].append(np.mean(p['first_action_optimal']))

                if verbose:
                    print('Last episode {}\t|\t Average step: {:.2f} \t|\tAverage (step-optim): {:.2f} \t|\t % (step-optim)<=5: {:.2f} \t|\t % (step-optim)<=3: {:.2f}'.format(
                          i_episode, 
                          np.mean(p['agent_step']),
                          np.mean(p['step_diff']),
                          np.mean(np.array(p['step_diff'])<=5),
                          np.mean(np.array(p['step_diff'])<=3)))

        if self.writer is not None:
            self.writer.close()
        return self.agent, performance

    def test(self, env, n_test_epochs, max_ep_steps, writer_prefix=''):
        self.agent.test_mode = True
        
        performance = {'test_ep':np.zeros(n_test_epochs), 
                       'agent_step':np.zeros(n_test_epochs), 
                       'step_diff':np.zeros(n_test_epochs), 
                       'first_action_optimal':np.zeros(n_test_epochs)}
        
        data = collect_episodes(N=n_test_epochs, env=env, agent=self.agent, max_steps=max_ep_steps)

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
            performance['first_action_optimal'][i] = actions[0]==optim_actions[0]
            if self.writer is not None:
                self.writer.add_scalar(writer_prefix+'test_epoch', i, self.current_epoch)
                self.writer.add_scalar(writer_prefix+'test_optim_a1', optim_actions[0], self.current_epoch)
                if len(optim_actions) >=2:
                    self.writer.add_scalar(writer_prefix+'test_optim_a2', optim_actions[1], self.current_epoch)
                else:
                    self.writer.add_scalar(writer_prefix+'test_optim_a2', -1, self.current_epoch)
                self.writer.add_scalar(writer_prefix+'agent_test_step', 
                                       performance['agent_step'][i], self.current_epoch)
                self.writer.add_scalar(writer_prefix+'agent_test_step_diff', 
                                       performance['step_diff'][i], self.current_epoch)
                self.writer.add_scalar(writer_prefix+'agent_test_first_action_optimal', 
                                       performance['first_action_optimal'][i], self.current_epoch)
        
        self.agent.test_mode = False
    
        return performance
