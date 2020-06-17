# TODO: write scripts replicating previous results that takes command line arguments and outputs results in /data5/liyuxuan/cupla
import torch
import numpy as np
from env.world import NetworkWorld
from env.task import TraversalTask
from model.curious_observer import CuriousObserver
from model.actor_critic import ActorCritic
from model.reinforce import Reinforce
from model.random_agent import RandomAgent
from model.trainer import ObserverTrainer, AgentTrainer

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

W1 = {}
W1['edges'] = {0:[14,1,2,3],
               1:[0,2,3,4],
               2:[1,3,4,0],
               3:[2,4,0,1],
               4:[3,5,1,2],
               5:[4,6,7,8],
               6:[5,7,8,9],
               7:[6,8,9,5],
               8:[7,9,5,6],
               9:[8,10,6,7],
               10:[9,11,12,13],
               11:[10,12,13,14],
               12:[11,13,14,10],
               13:[12,14,10,11],
               14:[13,0,11,12]}
W1['nodes'] = np.array(list(W1['edges'].keys()))

W2 = {}
W2['edges'] = {0:[14,3], 1:[0,4], 2:[1,0],
         3:[2,1], 4:[3,2], 5:[4,8],
         6:[5,9], 7:[6,5], 8:[7,6],
         9:[8,7], 10:[9,13], 11:[10,14],
         12:[11,10], 13:[12,11], 14:[13,12]}
W2['nodes'] = np.array(list(W2['edges'].keys()))

world1 = NetworkWorld(W1['nodes'], W1['edges'], action_dim=4)
world2 = NetworkWorld(W2['nodes'], W2['edges'], action_dim=2)

def run(run_ID,
        log_dir,
        cuda_idx,
        world,
        agent,
        use_observer_fc1=True,
        use_observer_fc2=True,
        n_epochs=1000,
        log_interval=10,
        n_test_epochs=50,
        max_ep_steps=20
        ):
    
    # TODO: save model checkpoints
    # TODO: add random agent support
    
    run_key = 'world%d_%s_run%d_%s_%s_%s' % (world, 
                                             agent, 
                                             run_ID,
                                             'fc1' if use_observer_fc1 else '',
                                             'fc2' if use_observer_fc2 else '',
                                             datetime.now().strftime('%y%m%d%H%M'))
    writer = SummaryWriter(log_dir+run_key)
    
    if world==1:
        world = world1
    if world==2:
        world = world2

    # train observer
    env = TraversalTask(world=world, start=0, goal=9,
                        change_start_on_reset=True, change_goal_on_reset=True,
                        goal_conditioned_obs=False, reward='sparse')
    observer = CuriousObserver(30, env.action_dim, 50)
    t = ObserverTrainer(env, observer, writer)
    t.train(1000, 10, verbose=False)
    
    fc1 = observer.fc1 if use_observer_fc1 else None
    fc2 = observer.fc2 if use_observer_fc2 else None
    
    # train agent
    if agent=='ra': # needs special trainer
        agent = RandomAgent(env.action_dim)
        t = AgentTrainer(env, agent, writer)
        for i_ep in range(int(n_epochs/log_interval)):
            t.test(n_test_epochs, max_ep_steps)
    if agent=='ac':
        agent = ActorCritic(in_size=30, hid_size=50, action_dim=2, gamma=0.9, epsilon=0.1,
                            fc1=fc1, action_head=fc2)
        t = AgentTrainer(env, agent, writer)
        t.train(n_epochs, log_interval, n_test_epochs, max_ep_steps, test=True, verbose=False)
    if agent=='ri':
        agent = Reinforce(in_size=30, hid_size=50, action_dim=2, gamma=0.9, epsilon=0.1,
                          fc1=fc1, action_head=fc2)
        t = AgentTrainer(env, agent, writer)
        t.train(n_epochs, log_interval, n_test_epochs, max_ep_steps, test=True, verbose=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier', type=int, default=0)
    parser.add_argument('--log_dir', help='logging directory', default='/data5/liyuxuan/cupla/')
    parser.add_argument('--cuda_idx', help='gpu to use', default=9)
    parser.add_argument('--world', help='world type', default=2)
    parser.add_argument('--agent', help='agent type', default='ac')
    args = parser.parse_args()
    run(run_ID=args.run_ID,
        log_dir=args.log_dir,
        cuda_idx=args.cuda_idx,
        world=args.world,
        agent=args.agent,
        )
