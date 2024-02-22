import GameEnv
import pygame
import numpy as np
from ddqn_keras import DDQNAgent
from collections import deque
import random, math
import matplotlib.pyplot as plt

TOTAL_GAMETIME = 100000000000 # Max game time for one episode
N_EPISODES = 10000
REPLACE_TARGET = 50 

game = GameEnv.RacingEnv()
game.fps = 60

GameTime = 0 
GameHistory = []
renderFlag = False

ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=1.00, epsilon_end=0.01, epsilon_dec=0.9995, replace_target= REPLACE_TARGET, batch_size=1024, input_dims=19)

# if you want to load the existing model uncomment this line.
# careful an existing model might be overwritten
#ddqn_agent.load_model()

ddqn_scores = []
eps_history = []
max_steps_history = []
avg_steps_history = []
maxscore = 0

def run():

    global maxscore
    for e in range(N_EPISODES):
        
        game.reset() #reset env 

        done = False
        score = 0
        counter = 0
        
        observation_, reward, done = game.step(0)
        observation = np.array(observation_)

        gtime = 0 # set game time back to 0
        
        renderFlag = True # if you want to render every episode set to true

        # if e % 10 == 0 and e > 0: # render every 10 episodes
        #     renderFlag = True

        while not done:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: 
                    return

            action = ddqn_agent.choose_action(observation)
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)

            # This is a countdown if no reward is collected the car will be done within 100 ticks
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward

            ddqn_agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            ddqn_agent.learn()
            
            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True

            if renderFlag:
                game.render(action)

        if score > 3 and score > maxscore:
            maxscore = score
            ddqn_agent.save_model_max()

        eps_history.append(e)
        ddqn_scores.append(score)
        max_steps_history.append(score)
        avg_score = np.mean(ddqn_scores[max(0, e - 100):(e + 1)])
        if len(max_steps_history) >= 100:
            avg_steps = np.mean(max_steps_history[-100:])
            avg_steps_history.append(avg_steps)

        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            ddqn_agent.update_network_parameters()

        if e % 40 == 0 and e > 10:
            ddqn_agent.save_model()
            print("save model")
            
        print('episode: ', e,'score: %.2f' % score,
              ' average score %.2f' % avg_score,
              ' epsolon: ', ddqn_agent.epsilon,
              ' memory size', ddqn_agent.memory.mem_cntr % ddqn_agent.memory.mem_size)   

        # Vẽ biểu đồ sau mỗi episode
        plt.plot(eps_history, ddqn_scores, marker='o', linestyle='-')
        if e>=100:
            plt.plot(eps_history[100:], avg_steps_history[1:], label='Avg step', linestyle='--')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('BIỂU ĐỒ REWARD ĐẠT ĐƯỢC TRÊN MỖI EPISODE ')
        plt.legend(['Reward','Avg_step'])
        plt.grid(True)
        plt.pause(0.05)  # Tạo độ trễ ngắn để cập nhật biểu đồ
        # if e % 1000 ==0:
        #     plt.show()
        #     break
        #     plt.savefig(f'plot_{e + 1}_episodes.png')
        # plt.clf()  # Xóa biểu đồ sau mỗi lần vẽ


run()
