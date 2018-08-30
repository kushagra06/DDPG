import numpy as np

import tensorflow as tf
import gym

from DDPG.src.gym_utils import get_action_bound, get_state_dimension, get_action_dimension
from DDPG.src.model import Actor, Critic
from DDPG.src.replay_buffer import ReplayBuffer
from DDPG.src.stochastic_process import OUProcess


def run_episode(actor,
                buffer,
                critic,
                env,
                gamma,
                freq,
                start_step,
                global_step,
                theta,
                sigma,
                evaluation=False):

    action_bound = get_action_bound(env)

    state        = env.reset()
    done         = False
    process      = OUProcess(a_dim=get_action_dimension(env), theta=theta, sigma=sigma)
    total        = 0
    test_rewards = []
    while not done:
        noise  = process.sample() if not evaluation else 0.
        action = actor.act([state.reshape(1, -1)])[0].flatten() + noise
        action = np.clip(action, action_bound[0], action_bound[1])

        next_state, reward, done, _ = env.step(action)

        if not evaluation:
            buffer.add(state, action, reward, next_state, done)

            if start_step < len(buffer):
                s_b, a_b, r_b, ns_b, done_t = buffer.sample_batch(64)
                a_t_b = actor.target_act([ns_b])[0]
                q_t_b = r_b[:, None] + gamma * (1. - done_t[:, None]) * critic.target_q_val([ns_b, a_t_b])[0]

                actor.train([s_b, critic.dq_da([s_b, a_b])[0]])
                critic.train([s_b, a_b, q_t_b])

                actor.target_update()
                critic.target_update()

        global_step += 1
        state  = next_state
        total += reward

        if global_step % freq == 0 and not evaluation:
            reward, _, _ = run_episode(actor=actor,
                                       buffer=buffer,
                                       critic=critic,
                                       env=env,
                                       gamma=gamma,
                                       freq=freq,
                                       start_step=start_step,
                                       global_step=global_step,
                                       evaluation=True,
                                       theta=theta,
                                       sigma=sigma)

            test_rewards.append(reward)

        if done:
            break

    return total, test_rewards, global_step


def run_experiment(actor,
                   buffer,
                   critic,
                   env,
                   freq,
                   gamma=0.99,
                   max_step=1000,
                   start_step=1000,
                   theta=0.15,
                   sigma=0.2):

    global_step = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        actor.target_init()
        critic.target_init()

        test_rewards = [run_episode(actor=actor,
                                    buffer=buffer,
                                    critic=critic,
                                    env=env,
                                    gamma=gamma,
                                    freq=freq,
                                    start_step=start_step,
                                    global_step=global_step,
                                    evaluation=True,
                                    theta=theta,
                                    sigma=sigma)[0]]
        train_rewards = []
        num_steps     = [0]
        while global_step < max_step:
            reward, test_reward, global_step = run_episode(actor=actor,
                                                           buffer=buffer,
                                                           critic=critic,
                                                           env=env,
                                                           gamma=gamma,
                                                           freq=freq,
                                                           start_step=start_step,
                                                           global_step=global_step,
                                                           evaluation=False,
                                                           theta=theta,
                                                           sigma=sigma)

            train_rewards.append(reward)
            test_rewards.extend(test_reward)
            num_steps.append(global_step)

        return train_rewards, np.asarray(test_rewards), num_steps


def main(arg):
    train_rewards = []
    num_steps     = []
    test_rewards  = np.zeros(shape=(arg.n_exp, arg.max // arg.freq + 1), dtype=np.float64)
    for i in range(arg.n_exp):
        environment = gym.make(arg.env)
        environment.seed(np.random.randint(12345))
        action_dim = get_action_dimension(environment)
        state_dim  = get_state_dimension(environment)
        act_bound  = np.asarray(get_action_bound(environment)).astype(np.float32)

        buffer = ReplayBuffer(size=arg.buffer,
                              a_dim=action_dim,
                              a_dtype=np.float32,
                              s_dim=state_dim,
                              s_dtype=np.float32,
                              store_mu=False)

        actor = Actor(a_dim=action_dim,
                      a_scale=act_bound,
                      s_dim=state_dim)

        critic = Critic(a_dim=action_dim, s_dim=state_dim)

        reward, test_rewards[i], steps = run_experiment(actor=actor,
                                                        buffer=buffer,
                                                        critic=critic,
                                                        env=environment,
                                                        freq=arg.freq,
                                                        gamma=arg.discount,
                                                        max_step=arg.max,
                                                        start_step=arg.start,
                                                        theta=arg.theta,
                                                        sigma=arg.sigma)

        tf.reset_default_graph()

        train_rewards.append(reward)
        num_steps.append(steps)

    attr = [(attr, getattr(arg, attr)) for attr in dir(arg) if attr[0] is not '_']
    post = ''.join(['_{}_{}'.format(a, v) for a, v in attr])

    np.save('test{}.npy'.format(post), test_rewards)

    import pickle as pkl
    with open('train{}.pkl'.format(post), mode='wb') as f:
        pkl.dump(train_rewards, f, protocol=-1)

    with open('steps{}.pkl'.format(post), mode='wb') as f:
        pkl.dump(num_steps, f, protocol=-1)


if __name__ == '__main__':
    from argparse import ArgumentParser
    argparser = ArgumentParser()
    argparser.add_argument('-b', '--buffer',
                           type=int,
                           default=1000000,
                           help='Buffer size (default: 1000000).')
    argparser.add_argument('-d', '--discount',
                           type=float,
                           default=0.99,
                           help='The discount factor (default: 0.99).')
    argparser.add_argument('-e', '--env',
                           type=str,
                           default='Pendulum-v0',
                           help='ID of gym environment (default: Pendulum-v0).')
    argparser.add_argument('-f', '--freq',
                           type=int,
                           default=2000,
                           help='Every this step, an evaluation episode is done.')
    argparser.add_argument('-m', '--max',
                           type=int,
                           default=2000000,
                           help='Maximum total number of steps (default: 2000000).')
    argparser.add_argument('-n', '--n-exp',
                           type=int,
                           default=20,
                           help='Number of experiments (default: 20).')
    argparser.add_argument('-s', '--start',
                           type=int,
                           default=2000,
                           help='Number of steps until the learning starts (default: 2000).')
    argparser.add_argument('--sigma',
                           type=float,
                           default=0.2,
                           help='sigma of OU process (default: 0.2).')
    argparser.add_argument('--theta',
                           type=float,
                           default=0.15,
                           help='theta of OU process (default: 0.15).')
    arg = argparser.parse_args()

    main(arg)