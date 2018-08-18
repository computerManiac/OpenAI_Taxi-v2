import gym
import numpy as np
import random
import pickle

env = gym.make('Taxi-v2')

action_size = env.action_space.n
state_size = env.observation_space.n
#qtable = np.zeros((state_size,action_size))
qtable = pickle.load(open('qtable.pickle','rb'))
rewards = []

hyper_params = pickle.load(open('hyper_params.pickle','rb'))
total_episodes = int(input('Enter total training Episodes: '))
lr = hyper_params['learning_rate']
#g = hyper_params['gamma']
#lr = 0.001
g = 0.655
max_steps = 99

epsilon = hyper_params['epsilon']
#epsilon = 1
min_epsilon = 0.001
max_epsilon = 1
decay = 0.0005

for episode in range(total_episodes):

	state = env.reset()
	total_rewards = 0
	steps = 0
	done = False

	for steps in range(max_steps):

		tradeoff = random.uniform(0,1)

		if tradeoff > epsilon:
			action = np.argmax(qtable[state,:])

		else:
			action = env.action_space.sample()

		new_state,reward,done,info = env.step(action)

		qtable[state,action] = qtable[state,action] + lr*(reward + g*np.max(qtable[new_state,:]) - qtable[state,action])
		total_rewards += reward
		state = new_state

		if done:
			break

	print('Done with Episode {}/{}'.format(episode+1,total_episodes))

	epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay*episode)
	rewards.append(total_rewards)

print('Training rewards: {}'.format(sum(rewards)/total_episodes))

hyper_params = {'epsilon':epsilon,'learning_rate':lr,'gamma':g}
with open('hyper_params.pickle','wb') as f:
	pickle.dump(hyper_params,f)

with open("qtable.pickle",'wb') as f:
	pickle.dump(qtable,f)

print('Beginning testing')

test_episodes = int(input('Enter test episodes: '))
test_rewards = []
test_steps = []

for episode in range(test_episodes):
	
	state = env.reset()
	steps = 0
	total_rewards = 0
	done = False

	print('Episode {}/{}'.format(episode+1,test_episodes))

	for steps in range(max_steps):

		action = np.argmax(qtable[state,:])

		new_state,reward,done,info = env.step(action)

		total_rewards += reward
		state = new_state

		if done:
			env.render()
			test_rewards.append(total_rewards)
			test_steps.append(steps)
			break


avg_rewards = sum(test_rewards)/test_episodes
avg_steps = sum(test_steps)/test_episodes

print("Average Rewards {} & Average Steps {}".format(avg_rewards,avg_steps))

env.close()