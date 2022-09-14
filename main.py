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
        task_min_dist,
        task_max_dist,
        agent,
        use_observer_fc1,
        use_observer_fc2,
        n_epochs=1000,
        log_interval=10,
        n_test_epochs=50,
        max_ep_steps=20,
        observer_checkpoint_interval=None,
        agent_checkpoint_interval=None,
        ):
    
    env_key = 'world{}_steps{}-{}'.format(world, task_min_dist, task_max_dist)
    agent_key = '{}{}{}'.format(agent, '_fc1' if use_observer_fc1 else '', '_fc2' if use_observer_fc2 else '')
    run_key = '{}_{}_run{}_{}'.format(env_key, agent_key, run_ID, 
                                      datetime.now().strftime('%y%m%d%H%M'))
    observer_writer = SummaryWriter(log_dir+'observer_'+run_key)
    agent_writer = SummaryWriter(log_dir+'agent_'+run_key)
    
    if world==1:
        world = world1
    if world==2:
        world = world2

    # train observer
    '''
    env = TraversalTask(world=world, start=0, goal=9,
                        change_start_on_reset=True, change_goal_on_reset=True,
                        goal_conditioned_obs=False, reward='sparse')
    observer = CuriousObserver(30, env.action_dim, 50)
    t = ObserverTrainer(env, observer, writer)
    t.train(n_epochs=1000, log_interval=1, batch_size=10, 
            checkpoint_interval=observer_checkpoint_interval, verbose=False)
    '''

    env = TraversalTask(world=world, start=0, goal=9, min_dist=1, max_dist=1,
                        change_start_on_reset=True, change_goal_on_reset=True,
                        goal_conditioned_obs=True, reward='sparse')

    if agent=='ac':
        observer = ActorCritic(in_size=30, hid_size=50, action_dim=env.action_dim, gamma=0.9, epsilon=0.1)
        t = AgentTrainer(env, observer, observer_writer)
        t.train(n_epochs, log_interval, n_test_epochs, max_ep_steps,
                checkpoint_interval=agent_checkpoint_interval, test=True, verbose=False)
    if agent=='ri':
        observer = Reinforce(in_size=30, hid_size=50, action_dim=env.action_dim, gamma=0.9, epsilon=0.1)
        t = AgentTrainer(env, observer, observer_writer)
        t.train(n_epochs, log_interval, n_test_epochs, max_ep_steps,
                checkpoint_interval=agent_checkpoint_interval, test=True, verbose=False)
    
    env = TraversalTask(world=world, start=0, goal=9, min_dist=task_min_dist, max_dist=task_max_dist,
                        change_start_on_reset=True, change_goal_on_reset=True,
                        goal_conditioned_obs=True, reward='sparse')

    if agent=='ra': # needs special trainer
        agent = RandomAgent(env.action_dim)
        t = AgentTrainer(env, agent, agent_writer)
        for i in range(n_epochs):
            if (i%log_interval==0):
                t.current_epoch = i
                min_dist = env.min_dist
                max_dist = env.max_dist
                for d in range(1, max_dist+1):
                    if d > 4:
                        break
                    else:
                        env.min_dist=d
                        env.max_dist=d
                        t.test(env, n_test_epochs, max_ep_steps, writer_prefix=str(d)+'-step_')
                env.min_dist = min_dist
                env.max_dist = max_dist
                t.test(env, n_test_epochs, max_ep_steps)
    if agent=='ac':
        fc1 = observer.fc1 if use_observer_fc1 else None
        action_head = observer.action_head if use_observer_fc2 else None
        value_head = observer.value_head if use_observer_fc2 else None
        agent = ActorCritic(in_size=30, hid_size=50, action_dim=env.action_dim, gamma=0.9, epsilon=0.1,
                            fc1=fc1, action_head=action_head, value_head=value_head)
        t = AgentTrainer(env, agent, agent_writer)
        t.train(n_epochs, log_interval, n_test_epochs, max_ep_steps, 
                checkpoint_interval=agent_checkpoint_interval, test=True, verbose=False)
    if agent=='ri':
        fc1 = observer.fc1 if use_observer_fc1 else None
        action_head = observer.action_head if use_observer_fc2 else None
        agent = Reinforce(in_size=30, hid_size=50, action_dim=env.action_dim, gamma=0.9, epsilon=0.1,
                          fc1=fc1, action_head=action_head)
        t = AgentTrainer(env, agent, agent_writer)
        t.train(n_epochs, log_interval, n_test_epochs, max_ep_steps, 
                checkpoint_interval=agent_checkpoint_interval, test=True, verbose=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--run_ID', help='run identifier', type=int, default=0)
    parser.add_argument('--log_dir', help='logging directory', default='/data5/liyuxuan/cupla/runs/')
    parser.add_argument('--cuda_idx', help='gpu to use', type=int, default=9)
    parser.add_argument('--world', help='world type', type=int, default=2)
    parser.add_argument('--task_min_dist', help='task difficulty (min dist)', type=int, default=2)
    parser.add_argument('--task_max_dist', help='task difficulty (max dist)', type=int, default=100)
    parser.add_argument('--agent', help='agent type', default='ac')
    parser.set_defaults(fc1=False, fc2=False)
    parser.add_argument('--fc1', help='use observer fc1', dest='fc1', action='store_true')
    parser.add_argument('--no-fc1', help='do not use observer fc1', dest='fc1', action='store_false')
    parser.add_argument('--fc2', help='use observer fc2', dest='fc2', action='store_true')
    parser.add_argument('--no-fc2', help='do not use observer fc2', dest='fc2', action='store_false')
    parser.add_argument('--n_epochs', help='train epochs', type=int, default=1000)
    parser.add_argument('--log_interval', help='logging interval', type=int, default=10)
    parser.add_argument('--n_test_epochs', help='test epochs per test', type=int, default=50)
    parser.add_argument('--max_ep_steps', help='max steps per episode', type=int, default=20)
    parser.add_argument('--observer_checkpoint_interval', help='frequency of observer model checkpoint', type=int, default=None)
    parser.add_argument('--agent_checkpoint_interval', help='frequency of agent model checkpoint', type=int, default=None)
    args = parser.parse_args()
    run(run_ID=args.run_ID,
        log_dir=args.log_dir,
        cuda_idx=args.cuda_idx,
        world=args.world,
        task_min_dist=args.task_min_dist,
        task_max_dist=args.task_max_dist,
        agent=args.agent,
        use_observer_fc1=args.fc1,
        use_observer_fc2=args.fc2,
        n_epochs=args.n_epochs,
        log_interval=args.log_interval,
        n_test_epochs=args.n_test_epochs,
        max_ep_steps=args.max_ep_steps,
        observer_checkpoint_interval=args.observer_checkpoint_interval,
        agent_checkpoint_interval=args.agent_checkpoint_interval,
        )
