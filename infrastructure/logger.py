import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time

#inspired by Berkeley CS 182's (who took it from 285)
#logz.py file (mainly just the report formatting)

class Logger():

    def __init__(self, config):
        self.config = config
        self.on_policy = config.params["model"] != "DQN"
        self.plot_value = self.config.params.get("plot_value", None)
        self.episode_rewards = []
        self.episode_lengths = []
        self.goals_reached = []
        self.statistics = []
        self.best = float("-inf")
        self.start_time = time.time()
        self.reported = []
        self.mkdir()

    def mkdir(self):
        self.experiment_filepath = "./experiments/" + self.config.params["name"] + "/" + "seed" + "_" + str(self.config.params["seed"])
        if not os.path.exists(self.experiment_filepath):
            os.makedirs(self.experiment_filepath)

    def add_reward(self, episode_reward, episode_length, goals_reached = None):
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        if goals_reached != None:
            self.goals_reached.append(goals_reached)

    def compute_statistics(self, t):
        labels = ["t", "Avg_Last_100_Episodes", "Std_Last_100_Episodes",
        "Best_Avg_100_Episodes", "Num_Episodes", "Avg_Episode_Len", "%_Max_Len", "Elapsed_Time_Hours"]
        avg = np.mean(self.episode_rewards[-100:])
        std = np.std(self.episode_rewards[-100:])
        if avg > self.best:
            self.best = avg
        best_avg = self.best
        num_episodes = len(self.episode_rewards)
        avg_episode_len = np.mean(self.episode_lengths[-100:])
        percent_max_len = len([i for i in self.episode_lengths[-100:] if i >= 995])
        duration = (time.time() - self.start_time) / 3600
        values = [t, avg, std, best_avg, num_episodes, avg_episode_len, percent_max_len, duration]
        if len(self.goals_reached) > 0:
            percent_1st_goal_reached = len([i for i in self.goals_reached[-100:] if len(i) >= 1])
            labels.append("%_1st_Goal_Reached")
            values.append(percent_1st_goal_reached)

            percent_2nd_goal_reached = len([i for i in self.goals_reached[-100:] if len(i) >= 2])
            labels.append("%_2nd_Goal_Reached")
            values.append(percent_2nd_goal_reached)

            percent_3rd_goal_reached = len([i for i in self.goals_reached[-100:] if len(i) >= 3])
            labels.append("%_3rd_Goal_Reached")
            values.append(percent_3rd_goal_reached)

        return labels, values


    def report(self, t, eps):
        reported_t = [t - t_reported for t_reported in self.reported]
        if len(reported_t) > 0 and min(reported_t) <= 200: #hacky way to fix the double reporting issue with the jump_outcome code, should also be <= episode_length not a hardcoded 200
            return
        self.reported.append(t)
        labels, values = self.compute_statistics(t)
        labels.append("Epsilon")
        values.append(eps)
        self.statistics.append(dict(zip(labels, values)))
        labels_width = max(12, max([len(label) for label in labels]))
        keystr = '%' + '%d' % labels_width
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + labels_width
        print("-" * n_slashes)
        for label, value in zip(labels, values):
            if isinstance(value, float):
                valstr = "%3.3g"%value
            elif isinstance(value, int) and value > 1000:
                valstr = "%8.2e"%value
            else:
                valstr = value
            print(fmt%(label, valstr))
        print("-" * n_slashes)


    def graph(self):
        if self.on_policy:
            x = [t["t"] / 1e3 for t in self.statistics]
            plt.xlabel("Episodes (in Thousands)")
        else:
            x = [t["t"] / 1e6 for t in self.statistics]
            plt.xlabel("Training Steps (Millions)")
        y = [t[self.plot_value] for t in self.statistics]
        plt.plot(x,y)
        plt.ylabel(self.plot_value)
        plt.title(self.config.params["name"])
        plt.savefig(self.experiment_filepath + "/" + "graph.png")
        plt.show()


    def save_experiment(self):
        filepath = self.experiment_filepath + "/" + "episode_rewards"
        file = open(filepath, 'wb')
        pickle.dump(self.episode_rewards, file)
        file.close()

        filepath = self.experiment_filepath + "/" + "experiment_statistics"
        file = open(filepath, 'wb')
        pickle.dump(self.statistics, file)
        file.close()
