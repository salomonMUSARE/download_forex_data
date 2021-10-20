#================================================================
#
#   File name   : RL-Bitcoin-trading-bot_7.py
#   Author      : PyLessons
#   Created date: 2021-02-25
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/RL-Bitcoin-trading-bot
#   Description : Trading Crypto with Reinforcement Learning #7
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import copy
import pandas as pd
import numpy as np
import random
from collections import deque
from tensorboardX import SummaryWriter
from tensorflow.keras.optimizers import Adam, RMSprop
from model import Actor_Model, Critic_Model, Shared_Model
from utils import TradingGraph, Write_to_file, Normalizing
import matplotlib.pyplot as plt
from datetime import datetime
from indicators import *
from multiprocessing_env import train_multiprocessing, test_multiprocessing
import json

salomon = 0
class CustomAgent:
    # A custom Bitcoin trading agent
    def __init__(self, Actor , Critic, lookback_window_size=50, lr=0.00005, epochs=1, optimizer=Adam, batch_size=32, model="", depth=0, comment=""):
        self.lookback_window_size = lookback_window_size
        self.model = model
        self.comment = comment
        self.depth = depth
        
        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1,2])

        # folder to save models
        self.log_name = datetime.now().strftime("%Y_%m_%d_%H_%M")+"_Crypto_trader"
        
        # State size contains Market+Orders+Indicators history for the last lookback_window_size steps
        # self.state_size = (lookback_window_size, depth + 2) # 5 standard OHCL information + market and indicators
        self.state_size = (1, depth ) # 5 standard OHCL information + market and indicators

        # Neural Networks part bellow
        self.lr = lr
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size
        print("state_size shape ", "*"*40)
        print(self.state_size)

        # Create shared Actor-Critic network model
        global salomon
        if salomon == 0:
            # print("called first time salomon is == ",salomon)
            # print("actor is == ", Actor)
            self.Actor = self.Critic = Shared_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer, model=self.model)
            

            salomon  = 1
            # print("self.actor is == ", self.Actor)

        else:
            # self.Actor = Actor
            # self.Critic = Critic

            self.Actor = self.Critic = Shared_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer, model=self.model)

        # Create Actor-Critic network model
        #self.Actor = Actor_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
        #self.Critic = Critic_Model(input_shape=self.state_size, action_space = self.action_space.shape[0], lr=self.lr, optimizer = self.optimizer)
        
    # create tensorboard writer
    def create_writer(self, initial_balance, normalize_value, train_episodes):
        self.replay_count = 0
        self.writer = SummaryWriter('runs/'+self.log_name)

        # Create folder to save models
        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)

        self.start_training_log(initial_balance, normalize_value, train_episodes)
            
    def start_training_log(self, initial_balance, normalize_value, train_episodes):      
        # save training parameters to Parameters.json file for future
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        params = {
            "training start": current_date,
            "initial balance": initial_balance,
            "training episodes": train_episodes,
            "lookback window size": self.lookback_window_size,
            "depth": self.depth,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch size": self.batch_size,
            "normalize value": normalize_value,
            "model": self.model,
            "comment": self.comment,
            "saving time": "",
            "Actor name": "",
            "Critic name": "",
        }
        with open(self.log_name+"/Parameters.json", "w") as write_file:
            json.dump(params, write_file, indent=4)


    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        rewards = np.vstack(rewards)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions 
        values = self.Critic.critic_predict(states)
        next_values = self.Critic.critic_predict(next_states)
        
        # Compute advantages
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
        '''
        plt.plot(target,'-')
        plt.plot(advantages,'.')
        ax=plt.gca()
        ax.grid(True)
        plt.show()
        '''
        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])
        
        # training Actor and Critic networks
        # print("states shape ", "*"*40)
        # print(states.shape)
        # print(y_true.shape)
        # print(rewards.shape)
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)
        c_loss = self.Critic.Critic.fit(states, target, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)
        # time.sleep("hdshbds")

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1

        return np.sum(a_loss.history['loss']), np.sum(c_loss.history['loss'])

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.actor_predict(np.expand_dims(state, axis=0))[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction
        
    def save(self, name="Crypto_trader", score="", args=[]):
        # directory = f"./{self.log_name}"
        # files_in_directory = os.listdir(directory)
        # filtered_files = [file for file in files_in_directory if file.endswith(".h5")]
        # for file in filtered_files:
        #     path_to_file = os.path.join(directory, file)
        #     os.remove(path_to_file)
        # save keras model weights
        self.Actor.Actor.save_weights(f"{self.log_name}/{score}_{name}_Actor.h5")
        self.Critic.Critic.save_weights(f"{self.log_name}/{score}_{name}_Critic.h5")
        

        

        # update json file settings
        if score != "":
            with open(self.log_name+"/Parameters.json", "r") as json_file:
                params = json.load(json_file)
            params["saving time"] = datetime.now().strftime('%Y-%m-%d %H:%M')
            params["Actor name"] = f"{score}_{name}_Actor.h5"
            params["Critic name"] = f"{score}_{name}_Critic.h5"
            with open(self.log_name+"/Parameters.json", "w") as write_file:
                json.dump(params, write_file, indent=4)

        # log saved model arguments to file
        if len(args) > 0:
            with open(f"{self.log_name}/log.txt", "a+") as log:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                arguments = ""
                for arg in args:
                    arguments += f", {arg}"
                log.write(f"{current_time}{arguments}\n")

    def load(self, folder, name):
        # load keras model weights
        self.Actor.Actor.load_weights(os.path.join(folder, f"{name}_Actor.h5"))
        self.Critic.Critic.load_weights(os.path.join(folder, f"{name}_Critic.h5"))

    def export(self):
        return self.Actor, self.Critic

        
class CustomEnv:
    # A custom Bitcoin trading environment
    def __init__(self, df, df_normalized, initial_balance=1000, lookback_window_size=50, Render_range=100, Show_reward=True, Show_indicators=False, normalize_value=40000):
        # Define action space and state size and other custom parameters

        self.df = df.reset_index()#.reset_index()#.dropna().copy().reset_index()
        self.df_normalized = df_normalized.reset_index()#.reset_index()#.copy().dropna().reset_index() /// check use of reset index
        self.df_total_steps = len(self.df)-1
        self.initial_balance = initial_balance
        
        self.lookback_window_size = lookback_window_size
        self.Render_range = Render_range # render range in visualization
        self.Show_reward = Show_reward # show order reward in rendered visualization
        self.Show_indicators = Show_indicators # show main indicators in rendered visualization

        # Orders history contains the balance, net_worth, crypto_held values for the last lookback_window_size steps
        # self.orders_history = deque(maxlen=self.lookback_window_size)
        self.orders_history = deque(maxlen=1)
        
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=1)

        self.normalize_value = normalize_value

        self.fees = 0.001 # default Binance 0.1% order fees

        self.columns = list(self.df_normalized.columns[2:7])
        self.columns2 = list(self.df_normalized.columns[7:])
        self.trades = deque(maxlen=self.Render_range) # limited orders memory for visualization
        self.rewards = deque(maxlen=self.Render_range)

        self.delimiter = 0

    # Reset the state of the environment to an initial state
    def reset_few(self):
        
        self.balance = 0
        self.net_worth = 0
        
        self.reward = 0
        self.ggoods = 0
        self.bbads = 0
        self.crypto_held = 0
        self.real_reward = 0
        self.episode_orders = 0 # track episode orders count
        self.prev_episode_orders = 0 # track previous episode orders count
        self.punish_value = 0

    def reset(self, env_steps_size = 0):
        self.visualization = TradingGraph(Render_range=self.Render_range, Show_reward=self.Show_reward, Show_indicators=self.Show_indicators) # init visualization
        
        self.balance = 0
        self.net_worth = 0
        
        self.reward = 0
        self.ggoods = 0
        self.bbads = 0
        self.crypto_held = 0
        self.real_reward = 0
        self.episode_orders = 0 # track episode orders count
        self.prev_episode_orders = 0 # track previous episode orders count
        self.env_steps_size = env_steps_size
        self.punish_value = 0
        if env_steps_size > 0: # used for training dataset
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - 205)
            # self.start_step = self.lookback_window_size + 5
            self.end_step = self.start_step + 205
        else: # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps - 2
            
        self.current_step = self.start_step

        
        
        # if env_steps_size > 0: # used for training dataset
        #     self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
        #     self.end_step = self.start_step + env_steps_size -2 
        # else: # used for testing dataset
        #     self.start_step = self.lookback_window_size
        #     self.end_step = self.df_total_steps - 2
            
        # self.current_step = self.start_step
        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i

            # one line for loop to fill market history withing reset call
            self.market_history.append([self.df_normalized.loc[current_step, column] for column in self.columns])
            self.orders_history.append([self.df_normalized.loc[current_step, column] for column in self.columns2])
            
        state = np.concatenate((self.orders_history, self.market_history), axis=1)
        # self.current_step += 1
        return state


        
    # Get the data points for the given current_step
    def next_observation(self):
        self.current_step += 1
        self.market_history.append([self.df_normalized.loc[self.current_step, column] for column in self.columns])
        self.orders_history.append([self.df_normalized.loc[self.current_step, column] for column in self.columns2])

        obs = np.concatenate((self.orders_history, self.market_history), axis=1)
        
        return obs

    # Execute one time step within the environment
    def step(self, action):
        current_price = self.df.loc[self.current_step, 'Close']
        next_close = self.df.loc[self.current_step +1 , 'Close']
        Date = self.df.loc[self.current_step, 'Date'] # for visualization
        High = self.df.loc[self.current_step, 'High'] # for visualization
        Low = self.df.loc[self.current_step, 'Low'] # for visualization

        # if action == 0: # Hold
        #     self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low,  'type': "none", 'current_price': current_price, "next_close" : next_close})
        #     pass

        if action == 0 :
            self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low,  'type': "none", 'current_price': current_price, "next_close" : next_close})
            # self.episode_orders += 1
            

        elif action == 1 :
            self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'type': "buy", 'current_price': current_price, "next_close" : next_close})
            self.episode_orders += 1
            
        elif action == 2:
            self.trades.append({'Date' : Date, 'High' : High, 'Low' : Low, 'type': "sell", 'current_price': current_price, "next_close" : next_close})
            self.episode_orders += 1
            

         
        self.net_worth = self.net_worth + self.get_reward()

        reward = self.trades[-1]["Reward"] 
        self.reward = reward       
        
        done = False
        obs = self.next_observation()  

         
        return obs, reward, done

    # Calculate reward
    def get_reward(self):
        self.punish_value += 0.001
        # print("self.trades[-1]['type']","0"*50,self.trades[-1]['type'])
        if self.episode_orders >= 1 and self.episode_orders > self.prev_episode_orders:
            self.prev_episode_orders = self.episode_orders
            # if self.trades[-1]['type'] == "buy" and self.trades[-2]['type'] == "sell":
            if self.trades[-1]['type'] == "buy" :
                self.punish_value  = 0
                reward = self.trades[-1]['next_close']  - self.trades[-1]['current_price']
                if reward > 0:
                    self.ggoods += 1
                    self.real_reward = 11 + reward/300
                elif reward < 0:
                    self.bbads += 1
                    self.real_reward = -13 + reward/300
                self.trades[-1]["Reward"] = self.real_reward
                return self.real_reward

            elif self.trades[-1]['type'] == "sell" :
                self.punish_value  = 0
                reward = self.trades[-1]['current_price'] - self.trades[-1]['next_close'] 
                if reward > 0:
                    self.ggoods += 1
                    self.real_reward = 11 + reward/300
                elif reward < 0:
                    self.bbads += 1   
                    self.real_reward =    -13 + reward/300   
                self.trades[-1]["Reward"] = self.real_reward
                return self.real_reward
            else:
                reward = -0.001 - self.punish_value 
                
                self.trades[-1]["Reward"] = reward
                # print("reward",reward)
                return reward
        else:
            reward = -0.001 - self.punish_value 
            
            self.trades[-1]["Reward"] = reward
            # print("reward",reward)
            return reward


    # render environment
    def render(self, visualize = False):
        #print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
        if visualize:
            # Render the environment to the screen
            img = self.visualization.render(self.df.loc[self.current_step], self.net_worth, self.trades)
            return img

        
def Random_games(env, visualize, test_episodes = 50, comment=""):
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action = np.random.randint(3, size=1)[0]
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                average_orders += env.episode_orders
                if env.net_worth < env.initial_balance: no_profit_episodes += 1 # calculate episode count where we had negative profit through episode
                print("episode: {}, net_worth: {}, average_net_worth: {}, orders: {}".format(episode, env.net_worth, average_net_worth/(episode+1), env.episode_orders))
                break

    print("average {} episodes random net_worth: {}, orders: {}".format(test_episodes, average_net_worth/test_episodes, average_orders/test_episodes))
    # save test results to test_results.txt file
    with open("test_results.txt", "a+") as results:
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        results.write(f'{current_date}, {"Random games"}, test episodes:{test_episodes}')
        results.write(f', net worth:{average_net_worth/(episode+1)}, orders per episode:{average_orders/test_episodes}')
        results.write(f', no profit episodes:{no_profit_episodes}, comment: {comment}\n')

def train_agent(env, agent, visualize=False, train_episodes = 50, training_batch_size=500):
    agent.create_writer(env.initial_balance, env.normalize_value, train_episodes) # create TensorBoard writer
    total_average = deque(maxlen=100) # save recent 100 episodes net worth
    best_average = 0 # used to track best average net worth
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        for t in range(training_batch_size):
            env.render(visualize)
            action, prediction = agent.act(state)
            next_state, reward, done = env.step(action)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(3)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state

        a_loss, c_loss = agent.replay(states, actions, rewards, predictions, dones, next_states)
        total_average.append(env.net_worth)
        average = np.average(total_average)
        
        agent.writer.add_scalar('Data/average net_worth', average, episode)
        agent.writer.add_scalar('Data/episode_orders', env.episode_orders, episode)
        
        print("episode: {:<5} net worth {:<7.2f} average: {:<7.2f} orders: {}".format(episode, env.net_worth, average, env.episode_orders))
        if episode > len(total_average):
            if best_average < average:
                best_average = average + 50
                print("Saving model")
                # agent.save(score="{:.2f}".format(best_average), args=[episode, average, env.episode_orders, a_loss, c_loss]) =====
                # agent.save(score="{:.2f}".format(best_average), args=[episode, average, env.episode_orders, a_loss, c_loss])
            agent.save()
    

def _test_agent(env, agent, visualize=True, test_episodes=10, folder="", name="Crypto_trader", comment=""):
    agent.load(folder, name)
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action, prediction = agent.act(state)
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                average_orders += env.episode_orders
                if env.net_worth < env.initial_balance: no_profit_episodes += 1 # calculate episode count where we had negative profit through episode
                print("episode: {:<5}, net_worth: {:<7.2f}, average_net_worth: {:<7.2f}, orders: {}".format(episode, env.net_worth, average_net_worth/(episode+1), env.episode_orders))
                break
            
    print("average {} episodes agent net_worth: {}, orders: {}".format(test_episodes, average_net_worth/test_episodes, average_orders/test_episodes))
    print("No profit episodes: {}".format(no_profit_episodes))
    # save test results to test_results.txt file
    with open("test_results.txt", "a+") as results:
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        results.write(f'{current_date}, {name}, test episodes:{test_episodes}')
        results.write(f', net worth:{average_net_worth/(episode+1)}, orders per episode:{average_orders/test_episodes}')
        results.write(f', no profit episodes:{no_profit_episodes}, model: {agent.model}, comment: {comment}\n')

    
def test_agent(test_df, test_df_nomalized, visualize=False, test_episodes=10, folder="", name="", comment="", Show_reward=False, Show_indicators=False):
    with open(folder+"/Parameters.json", "r") as json_file:
        params = json.load(json_file)
    if name != "":
        params["Actor name"] = f"{name}_Actor.h5"
        params["Critic name"] = f"{name}_Critic.h5"
    name = params["Actor name"][:-9]

    agent = CustomAgent(Actor = 0 , Critic = 0,lookback_window_size=params["lookback window size"], optimizer=Adam, depth=params["depth"], model=params["model"])

    env = CustomEnv(df=test_df, df_normalized=test_df_nomalized, lookback_window_size=params["lookback window size"], Show_reward=Show_reward, Show_indicators=Show_indicators)

    agent.load(folder, name)
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    good_bad = deque(maxlen = 100)
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            if env.bbads != 0:
                good_bad.append((env.ggoods/env.bbads))
            good_bad_average = np.average(good_bad)
            action, prediction = agent.act(state)
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                average_orders += env.episode_orders
                if env.net_worth < 200: no_profit_episodes += 1 # calculate episode count where we had negative profit through episode
                print("episode: {:<5}, net_worth: {:<7.2f}, average_net_worth: {:<7.2f}, orders: {}, good_bad_average: {:<7.2f}".format(episode, env.net_worth, average_net_worth/(episode+1), env.episode_orders,good_bad_average))
                break
            
    print("average {} episodes agent net_worth: {}, orders: {}".format(test_episodes, average_net_worth/test_episodes, average_orders/test_episodes))
    print("No profit episodes: {}".format(no_profit_episodes))
    # save test results to test_results.txt file
    with open("test_results.txt", "a+") as results:
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        results.write(f'{current_date}, {name}, test episodes:{test_episodes}')
        results.write(f', net worth:{average_net_worth/(episode+1)}, orders per episode:{average_orders/test_episodes}')
        results.write(f', no profit episodes:{no_profit_episodes}, model: {agent.model}, comment: {comment}\n')


if __name__ == "__main__":            
    df = pd.read_csv('./BTCUSD_1h_new.csv')
    df = df[:-(7412+150)]
    print(len(df))
    
    # df=df[['Date', 'Open', 'Close', 'High', 'Low', 'Volume']]
    # df = df.sort_values('Date')
    df = AddIndicators(df) 
    df['7vs25'] = df['sma7'] - df['sma25']
    df['7vs40'] = df['sma7'] - df['sma40']
    df['25vs40'] = df['sma25'] - df['sma40']

    
    df_nomalized = Normalizing(df[99:])[1:]

    df = df[200:]
    df_nomalized = df_nomalized[100:]
    # print(df)
    # print(df_nomalized)
    # print(len(df) )
    # print(len(df_nomalized) )
    depth = len(list(df.columns[1:]))

    


    #======================================

    lookback_window_size = 1
    # test_window = 720 # 3 months
    test_window = 350 # 3 months
    
    # split training and testing datasets
    train_df = df[:-test_window - lookback_window_size] # we leave 100 to have properly calculated indicators
    test_df = df[-test_window - lookback_window_size:]
    # print("#===================================================================")
    # print(test_df)
    
    # split training and testing normalized datasets
    train_df_nomalized = df_nomalized[:-test_window-lookback_window_size] # we leave 100 to have properly calculated indicators
    test_df_nomalized = df_nomalized[-test_window-lookback_window_size:]
    print(test_df)
    print(train_df_nomalized)

    # #===========================
    # # single processing training
    agent = CustomAgent(Actor = 0 , Critic = 0,lookback_window_size=lookback_window_size, lr=0.00001, epochs=5, optimizer=Adam, batch_size = 32, model="CNN", depth=depth, comment="Normalized")
    # train_env = CustomEnv(df=train_df, df_normalized=train_df_nomalized, lookback_window_size=lookback_window_size)
    # train_agent(train_env, agent, visualize=False, train_episodes=50000, training_batch_size=500)
    test_agent(test_df, test_df_nomalized, visualize=False, test_episodes=10, folder="2021_10_18_21_07_Crypto_trader", name="12384.45_Crypto_trader", comment="3 months", Show_reward=False, Show_indicators=False)
    # agent.load(folder="2021_10_08_07_38_Crypto_trader", name="34259.24_Crypto_trader")
    # # multiprocessing training/testing. Note - run from cmd or terminal
    # Actorr, Criticc = agent.export()


    # print("done loading", "@"*12)

    #=======================================
    # agent = CustomAgent(Actor = Actorr , Critic = Criticc,lookback_window_size=lookback_window_size, lr=0.00001, epochs=5, optimizer=Adam, batch_size=32, model="CNN", depth=depth, comment="Normalized")
    # agent = CustomAgent(Actor = 0 , Critic = 0,lookback_window_size=lookback_window_size, lr=0.00001, epochs=5, optimizer=Adam, batch_size=32, model="CNN", depth=depth, comment="Normalized")
    # train_multiprocessing(CustomEnv, agent, train_df, train_df_nomalized, num_worker = 1, training_batch_size=100, visualize=False, EPISODES=40000)

    # test_multiprocessing(CustomEnv, CustomAgent, test_df, test_df_nomalized, num_worker = 16, visualize=False, test_episodes=1000, folder="2021_02_18_21_48_Crypto_trader", name="3906.52_Crypto_trader", comment="3 months")
    # test_multiprocessing(CustomEnv, CustomAgent, test_df, test_df_nomalized, num_worker = 16, visualize=True, test_episodes=1000, folder="2021_02_21_17_54_Crypto_trader", name="3263.63_Crypto_trader", comment="3 months")
