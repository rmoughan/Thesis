


class HierarchicalBaseAgent:

    def __init__(self, config, logger, device):
        pass


    def train_off_policy(self):
        #want to check the paper to make sure that the intrinsic critic reward
        #that it is referring to is just +1 for getting to a subgoal

        print("training hierarchical algorithm")
        #probably need some kind of start_state for the meta_controller buffer
        controller.last_obs = self.env.reset()
        #should do some pre-training
        while self.t < self.num_timesteps:
            controller.done = False
            controller.episode_reward = 0
            self.intrinsic_reward = 0
            self.start_obs = controller.last_obs
            self.current_goal = meta_controller.step()
            while not controller.done or not self.env.at_goal(self.last_obs, self.current_goal):
                controller.step() #(need to modify step function)
                controller.update()
                meta_controller.update()

            self.total_reward = self.intrinsic_reward + controller.episode_reward
            #should I make a separate DQN controller and meta_controller class? Or maybe just separate methods in the DQN class...

            #store entire trajectory in meta_controller buffer
            #populate goals with any new findings
            self.env.add_goal() #need to be careful because right now these envs don't all match...
            #kinda want to make a meta_controller_buffer class. Should I though... I guess this also ties in to the above determinatino of how I implement the metacontroller
            if not controller.done:
                meta_controller.step() #don't know how I want to have return value

            self.logger.add_reward(self.total_reward) #may want to modify to break down intrinsic vs. extrinsic reward
        if self.plot_experiment:
            self.logger.graph()
        self.logger.save_experiment()
