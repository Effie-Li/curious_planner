import glob
import numpy as np
import pandas as pd
from model.random_agent import RandomAgent
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def collect_episodes(N, env, agent=None, max_steps=1):
    
    # TODO: optimize
    
    if agent is None:
        agent = RandomAgent(env.action_dim)
    
    data = []
    for i in range(N):
        ep_states = []
        ep_actions = []
        ep_next_states = []
        ep_rewards = []
        ep_done = []
        
        s = env.reset()
        
        for t in range(max_steps):
            a = agent.step(s)
            ns, r, done = env.step(a)
            
            ep_states.append(s)
            ep_actions.append(a)
            ep_next_states.append(ns)
            ep_rewards.append(r)
            ep_done.append(done)
            
            if done:
                break
            
            s = ns
            
        x = {'ep':i,
             'states':ep_states,
             'actions':ep_actions,
             'next_states':ep_next_states,
             'rewards':ep_rewards,
             'dones':ep_done,
             'env_start':env.start,
             'env_goal':env.goal
            }
        data.append(x)
    
    return data

def tfevents_to_csv(logdir, prefix=None, save_csv=False):
    
    files = glob.glob(logdir+'*') if prefix is None else glob.glob(logdir+prefix+'*')
    run_list = [f[len(logdir):] for f in files]
    
    dataframes = []
    for run in run_list:
        event = EventAccumulator(logdir+run)
        event.Reload()
        dfs = None
        for s in event.scalars.Keys():
            steps = [t.step for t in event.Scalars(s)]
            values = [t.value for t in event.Scalars(s)]
            if dfs is None:
                dfs = pd.DataFrame({'step':steps, s:values})
            else:
                df = pd.DataFrame({'step':steps, s:values})
                dfs = dfs.merge(df, on='step', how='left')
        dfs['run'] = run
        dataframes.append(dfs)
    
    data = pd.concat(dataframes)
    
    if save_csv:
        fname = logdir+prefix+'.csv'
        data.to_csv(fname)
    
    return data