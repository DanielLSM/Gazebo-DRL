'''A complete DDPG agent, everything running on tensorflow should just run 
in this class for sanity and simplicity. Moreoever, every variable and
hyperparameter should be stored within the tensorflow graph to grant
increased performance'''
#agent=DDPG_agent(something,something)....
from canton.misc import get_session
import tensorflow as tf
from copy import copy
from memoryNIPS import Memory
import tensorflow.contrib as tc
from models import *

from copy import copy

from math import *
import random
import time

import numpy as np
from noise import one_fsq_noise
from observation_processor import process_observation as po
from observation_processor import generate_observation as go

from plotter import interprocess_plotter as plotter

from triggerbox import TriggerBox

import traceback

from multi import fastenv


class DDPG_agent(object):
    
    def __init__(self,observation_dims, action_space, param_noise=None,param_noise_adaption_interval=50,adaptive_param_noise_policy_threshold=.1,
        alpha=0.9,gamma=0.99,memory_size=1000000,batch_size=64,tau=5e-4,
        actor_l2_reg=1e-7,critic_l2_reg=1e-7,train_multiplier=1):

        #Pre-processing
        self.render = False
        self.training = True
        self.plotter = plotter(num_lines=3)
        low = action_space.low
        high = action_space.high
        self.action_dims = action_space.shape[0]
        self.action_bias = high/2. + low/2.
        self.action_multiplier = high - self.action_bias
        def clamper(actions):
            return np.clip(actions,a_max=action_space.high,a_min=action_space.low)
        self.clamper = clamper
        observation_shape = (None,observation_dims)
        action_shape = (None,self.action_dims)
        print('inputdims:{}, outputdims:{}'.format(observation_dims,self.action_dims))

        #Input tensorflow nodes
        self.observation = tf.placeholder(tf.float32, shape=observation_shape, name='observation')
        self.action = tf.placeholder(tf.float32, shape=action_shape, name='action')        
        self.observation_after = tf.placeholder(tf.float32, shape=observation_shape, name='observation_after')
        self.reward = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')        
        self.terminals1 = tf.placeholder(tf.float32, shape=(None, 1), name='terminals1')
        self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')
        
        #self.critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')
        #self.param_noise_stddev = tf.placeholder(tf.float32, shape=(), name='param_noise_stddev')
 
        #Hyper Parameters
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tf.Variable(tau)
        self.actor_l2_reg = actor_l2_reg
        self.critic_l2_reg = critic_l2_reg
        self.batch_size = batch_size
        self.train_multiplier = train_multiplier

        #Noise
        self.param_noise = param_noise
        self.param_noise_adaption_interval = param_noise_adaption_interval
        self.adaptive_param_noise_policy_threshold = adaptive_param_noise_policy_threshold
        #self.noise = noise

        #Replay Memory
        self.memory_replay = Memory(limit=memory_size,action_shape=(self.action_dims,),observation_shape=(observation_dims,))  
        
        #Networks
        self.actor = Actor(self.action_dims,self.action_multiplier,self.action_bias)
        self.target_actor = copy(self.actor)
        self.target_actor.name = 'target_actor'
        self.critic = Critic(observation_dims,self.action_dims)
        self.target_critic = copy(self.critic)
        self.target_critic.name = 'target_critic'

        #Expose nodes from the tf graph to be used

        # Critic Nodes
        self.a2 = self.target_actor(self.observation_after)
        self.q2 = self.target_critic(self.observation_after , self.a2)
        self.q1_target = self.reward + (1-self.terminals1) * self.gamma * self.q2
        self.q1_predict = self.critic(self.observation,self.action)
        self.critic_loss = tf.reduce_mean((self.q1_target - self.q1_predict)**2)

        # Actor Nodes
        self.a1_predict = self.actor(self.observation)
        self.q1_predict = self.critic(self.observation,self.a1_predict,reuse=True)
        self.actor_loss = tf.reduce_mean(- self.q1_predict) 

        # Infer
        self.a_infer = self.actor(self.observation,reuse=True)
        self.q_infer = self.critic(self.observation,self.a_infer,reuse=True)

        # Setting Nodes to Sync target networks
        self.setup_target_network_updates()

        # Train Boosters
        self.traincounter = 0

        # Optimzers
        self.opt_actor = tf.train.AdamOptimizer(1e-4)
        self.opt_critic = tf.train.AdamOptimizer(3e-4) #me it was 3e-4

        #L2 weight loss
        #critic_reg_vars = [var for var in self.critic.trainable_vars if 'kernel' in var.name and 'output' not in var.name]
        #self.critic_reg = tc.layers.apply_regularization(
        #    tc.layers.l2_regularizer(self.critic_l2_reg),
        #    weights_list=critic_reg_vars
        #)

        #actor_reg_vars = [var for var in self.actor.trainable_vars if 'kernel' in var.name and 'output' not in var.name]
        #self.actor_reg = tc.layers.apply_regularization(
        #    tc.layers.l2_regularizer(self.actor_l2_reg),
        #    weights_list=actor_reg_vars
        #)

        # Nodes to run one backprop step on the actor and critic
        #self.cstep = self.opt_critic.minimize(self.critic_loss+self.critic_reg,
        #    var_list=self.critic.trainable_vars)
        #self.astep = self.opt_actor.minimize(self.actor_loss+self.actor_reg,
        #    var_list=self.actor.trainable_vars)

        # Nodes to run one backprop step on the actor and critic
        self.cstep = self.opt_critic.minimize(self.critic_loss,var_list=self.critic.trainable_vars)
        self.astep = self.opt_actor.minimize(self.actor_loss,var_list=self.actor.trainable_vars)

        #Setup parameter noise
        self.setup_param_noise()

        #Saver
        self.saver = tf.train.Saver()
        
        # Initialize and Sync Networks
        self.initialize()
        self.sync_target()

        #A thread lock for all this proxys messing with out agent :) (our?)
        import threading as th
        self.lock = th.Lock()

        tf.summary.FileWriter(logdir='underworld_dumpster/graph_model', graph=tf.get_default_graph())
        print('agent initialized :>')
    
    def __call__(self,obs):
        input_observation = np.reshape(obs,(1,len(obs)))
        feed_dict = {self.observation:input_observation}
        #actor = self.actor
        #obs = np.reshape(observation,(1,len(observation)))###############
        sess = get_session()
        #res = sess.run(self.a_infer,self.q_infer,feed_dict=feed_dict)
        [a,q] = sess.run([self.perturbed_actor_tf,self.q_infer],feed_dict=feed_dict)
        actions,q = a[0],q[0]

        #if curr_noise is not None:
        #    disp_actions = (actions-self.action_bias) / self.action_multiplier
        #    disp_actions = disp_actions * 5 + np.arange(self.action_dims) * 12.0 + 30

        #    noise = curr_noise * 5 - np.arange(self.action_dims) * 12.0 - 30

            # self.lock.acquire()
            #self.loggraph(np.hstack([disp_actions,noise,q]))
            # self.lock.release()
            # temporarily disabled.

        #action, q = self.sess.run([actor_tf, self.critic_with_actor_tf], feed_dict=feed_dict)
        #action = action.flatten()
        #if self.action_noise is not None and apply_noise:
        #    noise = self.action_noise()
        #    assert noise.shape == action.shape
        #    action += noise
        #action = np.clip(action, self.action_range[0], self.action_range[1])
        return actions #action, q

        

    def __len__(self):
        #return memory_replay_size and/or number of episodes
        return self.memory_replay.nb_entries
         
    def initialize(self):
        sess = get_session()
        sess.run(tf.global_variables_initializer())
        sess.graph.finalize()
        #self.actor_optimizer.sync()
        #self.critic_optimizer.sync()
        #self.sess.run(self.target_init_updates)

    def feed_experience(self,obs0, action, reward, obs1, terminal1):
        #it will be made thread safe
        self.memory_replay.append(obs0, action, reward, obs1, terminal1)

    def train(self):
        mem_replay = self.memory_replay
        batch_size = self.batch_size

        if len(self) > batch_size * 64:

            for i in range(self.train_multiplier):
                batch = mem_replay.sample(batch_size)
                sess = get_session()
                res = sess.run([self.critic_loss,
                    self.actor_loss,
                    self.cstep,
                    self.astep,
                    self.target_soft_updates],
                    feed_dict={
                    self.observation:batch['obs0'],
                    self.action:batch['actions'],
                    self.observation_after:batch['obs1'],
                    self.reward:batch['rewards'],
                    self.terminals1:batch['terminals_1'],
                    self.tau:5e-4})
                #self.sync_target(update='soft')
                self.traincounter += 1
                if self.traincounter%20==0:
                    print(' '*30, 'closs: {:6.4f} aloss: {:6.4f}'.format(
                res[0],res[1]),end='\r')
        #return res


        
    def load_agent(self,i):
        sess = get_session()
        self.saver.restore(sess,"/home/daniel/Videos/underworld/underworld_dumpster/model/model-"+str(i))
        self.memory_replay.load("/home/daniel/Videos/underworld/underworld_dumpster/mem.pickle"+str(i))
    
    def save_agent(self,i):
        sess = get_session()
        self.saver.save(sess,"/home/daniel/Videos/underworld/underworld_dumpster/model/model",global_step=i)
        self.memory_replay.save("/home/daniel/Videos/underworld/underworld_dumpster/mem.pickle"+str(i))
    
    #def load_hyper_parameters(self):
        #pass
    
    #def save_hyper_parameters(self):
        #pass
    
    def setup_target_network_updates(self):
        actor_init_updates, actor_soft_updates = get_target_updates(self.actor.vars, self.target_actor.vars, self.tau)
        critic_init_updates, critic_soft_updates = get_target_updates(self.critic.vars, self.target_critic.vars, self.tau)
        self.target_init_updates = [actor_init_updates, critic_init_updates]
        self.target_soft_updates = [actor_soft_updates, critic_soft_updates]

    def sync_target(self,update='hard'):
        sess = get_session()
        if update=='hard':
            sess.run(self.target_init_updates)
        else:
            sess.run(self.target_soft_updates,feed_dict={self.tau:5e-4})

    def play(self,env,max_steps=-1,realtime=False): # play 1 episode
        timer = time.time()
        #noise_source = one_fsq_noise()

        #for j in range(200):
        #    noise_source.one((self.action_dims,),noise_level)

        max_steps = max_steps if max_steps > 0 else 50000
        steps = 0
        total_reward = 0
        episode_memory = []


        # removed: state stacking
        # moved: observation processing

        try:
            observation = env.reset()
            if len(self) > self.batch_size * 64: #me it was 64
                self.adapt_param_noise() #me
            self.change_is_goodz() #me

        except Exception as e:
            print('(agent) something wrong on reset(). episode terminates now')
            traceback.print_exc()
            print(e)
            return

        while True and steps <= max_steps:
            steps +=1

            observation_before_action = observation # s1

            # exploration_noise = noise_source.one((self.action_dims,),noise_level)
            # exploration_noise -= noise_level * 1

            # self.lock.acquire() # please do not disrupt.
            action = self(observation_before_action) # a1
            # self.lock.release()


            # add noise to our actions, since our policy by nature is deterministic
            # exploration_noise *= self.action_multiplier
            # print(exploration_noise,exploration_noise.shape)
            # action += exploration_noise
            action = self.clamper(action)
            action_out = action

            # o2, r1,
            try:
                observation, reward, done, _info = env.step(action_out) # take long time
                #reward_makro = 10 * reward
            except Exception as e:
                print('(agent) something wrong on step(). episode teminates now')
                traceback.print_exc()
                print(e)
                return

            # d1
            isdone = 1 if done else 0
            total_reward += reward

            # feed into replay memory
            if self.training == True:
                episode_memory.append([
                    observation_before_action,action,reward,observation,isdone
                ])

                # don't feed here since you never know whether the episode will complete without error.
                # self.feed_one((
                #     observation_before_action,action,reward,isdone,observation
                # )) # s1,a1,r1,isdone,s2
                # self.lock.acquire()
                #self.train(verbose=2 if steps==1 else 0)
                # self.lock.release()
                self.train()
                if len(self) > self.batch_size * 64 and steps % self.param_noise_adaption_interval == 0:
                    self.adapt_param_noise() #me

            #if self.render==True:
            #     env.render()
            if done :
                break

        # print('episode done in',steps,'steps',time.time()-timer,'second total reward',total_reward)
        totaltime = time.time()-timer
        print('episode done in {} steps in {:.2f} sec, {:.4f} sec/step, got reward :{:.2f}'.format(
        steps,totaltime,totaltime/steps,total_reward
        ))
        self.lock.acquire()
        # cause thread safe
        for step_memory in episode_memory:
            self.feed_experience(step_memory[0],step_memory[1],step_memory[2],step_memory[3],step_memory[4])

        self.plotter.pushys([total_reward,self.param_noise.current_stddev,(time.time()%3600)/3600-2])
        # self.noiseplotter.pushy(noise_level)
        self.lock.release()

        return

    def fetch_all_tensors(self):
        lista = tf.contrib.graph_editor.get_tensors(tf.get_default_graph())
        #print(lista)
        print('A lista tem tamanho: ',len(lista))
    #def loggraph(self,waves):
    #    wg = self.wavegraph
    #    wg.one(waves.reshape((-1,)))

    #All across the hype train of param noise

    def get_perturbed_actor_updates(self,actor, perturbed_actor, param_noise_stddev):
        assert len(actor.vars) == len(perturbed_actor.vars)
        assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)
        # Falta layer norm for sure
        updates = []
        for var, perturbed_var in zip(actor.vars, perturbed_actor.vars):
            if var in actor.perturbable_vars:
                updates.append(tf.assign(perturbed_var, var + tf.random_normal(tf.shape(var), mean=0., stddev=param_noise_stddev)))
            else:
                updates.append(tf.assign(perturbed_var, var))
        assert len(updates) == len(actor.vars)
        return tf.group(*updates)


    def setup_param_noise(self):

        # Configure perturbed actor.
        param_noise_actor = copy(self.actor)
        #print('Actor len is {} and param_noise_actor is {}'.format(len(self.actor.perturbable_vars),len(param_noise_actor.perturbable_vars)))
        param_noise_actor.name = 'param_noise_actor'
        self.perturbed_actor_tf = param_noise_actor(self.observation)
        #print('setting up param noise')
        self.perturb_policy_ops = self.get_perturbed_actor_updates(self.actor, param_noise_actor, self.param_noise_stddev)

        # Configure separate copy for stddev adoption.
        adaptive_param_noise_actor = copy(self.actor)
        adaptive_param_noise_actor.name = 'adaptive_param_noise_actor'
        self.adaptive_actor_tf = adaptive_param_noise_actor(self.observation)
        self.perturb_adaptive_policy_ops = self.get_perturbed_actor_updates(self.actor, adaptive_param_noise_actor, self.param_noise_stddev)
        self.adaptive_policy_distance = tf.sqrt(tf.reduce_mean(tf.square(self.a_infer - self.adaptive_actor_tf)))
        print('setting up parameter noise :>')
    
    #Change this one
    def adapt_param_noise(self):
        
        sess = get_session()
        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        batch = self.memory_replay.sample(batch_size=self.batch_size)
        sess.run(self.perturb_adaptive_policy_ops, feed_dict={
            self.param_noise_stddev: self.param_noise.current_stddev,
        })
        distance = sess.run(self.adaptive_policy_distance, feed_dict={
            self.observation: batch['obs0'],
            self.param_noise_stddev: self.param_noise.current_stddev,
        })

        #mean_distance = mpi_mean(distance)
        self.param_noise.adapt(distance)
        #return mean_distance

    def change_is_goodz(self):

        sess = get_session()
        sess.run(self.perturb_policy_ops, feed_dict={
                self.param_noise_stddev: self.param_noise.current_stddev,
            })

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean
