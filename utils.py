from model.random_agent import RandomAgent

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