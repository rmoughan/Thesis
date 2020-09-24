#The goal of this file is just to see if I can make a controller that can learn
#to reach a simple, close subgoal

#next thing to do is to wrap all of this in something I can run and then work on
#configs



def montezuma_train_off_policy(self):
    print("training")
    self.last_obs = self.env.reset()
    env.current_goal = [111,121]
    while self.t < self.num_timesteps:
        self.done, self.at_goal = False, False
        self.episode_reward = 0
        self.intrinsic_reward = 0
        while not self.done and not self.at_goal:
            self.step()
            self.update()
            if self.t % self.report_freq == 0:
                self.logger.report(self.t)
        if self.at_goal:
            print("reached goal!!!")
            self.intrinsic_reward = 1
        self.logger.add_reward(self.episode_reward + self.intrinsic_reward)
    if self.plot_experiment:
        self.logger.graph()
    self.logger.save_experiment()
