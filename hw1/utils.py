import operator
import typing

import joblib
import numba
import numpy as np
from IPython.display import clear_output
from matplotlib import pyplot as plt


def show_progress(rewards_batch, log, percentile, reward_range=[-990, +10], show_percentile=True):
    """
    A convenience function that displays training progress.
    No cool math here, just charts.
    """

    mean_reward = np.mean(rewards_batch)
    threshold = np.percentile(rewards_batch, percentile)
    log.append([mean_reward, threshold])

    plt.figure(figsize=[8, 4 if show_percentile else 8])
    if show_percentile:
        plt.subplot(1, 2, 1)
    plt.plot(list(zip(*log))[0], label='Mean rewards')
    plt.plot(list(zip(*log))[1], label='Reward thresholds')
    plt.legend()
    plt.grid()

    if show_percentile:
        plt.subplot(1, 2, 2)
        plt.hist(rewards_batch, range=reward_range)
        plt.vlines([np.percentile(rewards_batch, percentile)],
                   [0], [100], label="percentile", color='red')
        plt.legend()
        plt.grid()

    clear_output(True)
    plt.show()


def generate_session(env, n_actions, agent, t_max=1000, test=False):
    """
    Play a single game using agent neural network.
    Terminate when game finishes or after :t_max: steps
    """
    states, actions = [], []
    total_reward = 0

    s, _ = env.reset()

    for t in range(t_max):
        # use agent to predict a vector of action probabilities for state :s:
        probs = agent.predict_proba(s.reshape(1, -1)).reshape(-1)

        assert probs.shape == (n_actions,), "make sure probabilities are a vector (hint: np.reshape)"

        # use the probabilities you predicted to pick an action
        if test:
            # on the test use the best (the most likely) actions at test
            # experiment, will it work on the train and vice versa?
            a = np.arange(0, n_actions)[np.argmax(probs)]
            # ^-- hint: try np.argmax
        else:
            # sample proportionally to the probabilities,
            # don't just take the most likely action at train
            a = np.random.choice(a=np.arange(0, n_actions), size=1, p=probs)[-1]
            # ^-- hint: try np.random.choice

        new_s, r, done, info, _ = env.step(a)

        # record sessions like you did before
        states.append(s)
        actions.append(a)
        total_reward += r

        s = new_s
        if done:
            break
    return np.array(states), np.array(actions), total_reward


def make_batches(sessions: typing.List[typing.Tuple[np.ndarray, np.ndarray, float]]):
    """Transforms list of session tuples into np.arrays of states, actions and rewards"""
    states = None
    actions = None
    rewards = []

    for state, action, reward in sessions:
        if states is None:
            states = state[None, ...]
        else:
            states = append3(states, state)
        if actions is None:
            actions = action[None, ...]
        else:
            actions = append2(actions, action)
        rewards.append(reward)

    return states, actions, np.array(rewards)


__pad__ = np.iinfo(np.int32).max


def append2(a: np.ndarray, b: np.ndarray, padding=__pad__):
    """Appends 1D or 2D array `b` to 2D array `a` with padding (default: maximum int32)"""
    if len(b.shape) < 2:
        pad_width = a.shape[1] - b.shape[0]
        b = b[None, ...]
    else:
        pad_width = a.shape[1] - b.shape[1]

    if pad_width > 0:
        b = np.pad(b, ((0, 0), (0, pad_width)), constant_values=padding)
        c = np.vstack([a, b])
    else:
        a = np.pad(a, ((0, 0), (0, -pad_width)), constant_values=padding)
        c = np.vstack([a, b])

    return c


def append3(a: np.ndarray, b: np.ndarray, padding=__pad__):
    """Appends 2D or 3D array `b` to 3D array `a` with padding (default: maximum int32)"""
    if len(b.shape) < 3:
        pad_width = a.shape[1] - b.shape[0]
        b = b[None, ...]
    else:
        pad_width = a.shape[1] - b.shape[1]

    if pad_width > 0:
        b = np.pad(b, ((0, 0), (0, pad_width), (0, 0)), constant_values=padding)
        c = np.vstack([a, b])
    else:
        a = np.pad(a, ((0, 0), (0, -pad_width), (0, 0)), constant_values=padding)
        c = np.vstack([a, b])

    return c


def select_elites(states_batch, actions_batch, rewards_batch, percentile=50, strict: bool = False):
    """
    Select states and actions from games that have rewards >= percentile
    :param strict:
    :param states_batch: list of lists of states, states_batch[session_i][t]
    :param actions_batch: list of lists of actions, actions_batch[session_i][t]
    :param rewards_batch: list of rewards, rewards_batch[session_i]
    :param percentile: percentile to cut off rewards

    :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions

    Please return elite states and actions in their original order
    [i.e. sorted by session number and timestep within session]

    If you are confused, see examples below. Please don't assume that states are integers
    (they will become different later).
    """
    reward_threshold = np.percentile(rewards_batch, q=percentile)

    op = operator.gt if strict else operator.ge

    mask: np.ndarray = op(rewards_batch, reward_threshold)

    elite_states, elite_actions = states_batch[mask, :, :], actions_batch[mask, :]

    elite_states = elite_states[elite_states != float(__pad__)].reshape(-1, elite_states.shape[-1])

    if elite_actions.dtype == np.float:
        elite_actions = elite_actions[elite_actions != float(__pad__)]
    else:
        elite_actions = elite_actions[elite_actions != __pad__]

    return elite_states, elite_actions


def train(env, agent, n_sessions, t_max, iters, buffer_size, percentile, strict=True, log=[]):
    n_actions = env.action_space.n

    agent.partial_fit([env.reset()[0]] * n_actions, range(n_actions), range(n_actions))

    states_buffer, actions_buffer, rewards_buffer = None, None, None

    for i in range(iters):
        # generate new sessions
        sessions = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(generate_session)(env, n_actions, agent, t_max, False) for _ in range(n_sessions)
        )

        states_batch, actions_batch, rewards_batch = make_batches(sessions)

        if states_buffer is None:
            states_buffer = states_batch
        else:
            states_buffer = append3(states_buffer, states_batch)

        if actions_buffer is None:
            actions_buffer = actions_batch
        else:
            actions_buffer = append2(actions_buffer, actions_batch)

        if rewards_buffer is None:
            rewards_buffer = rewards_batch
        else:
            rewards_buffer = np.concatenate([rewards_buffer, rewards_batch])

        size = buffer_size * n_sessions

        elite_states, elite_actions = select_elites(
            states_batch if states_buffer.shape[0] < size else states_buffer[-size:, :, :],
            actions_batch if actions_buffer.shape[0] < size else actions_buffer[-size:, :],
            rewards_batch if rewards_buffer.shape[0] < size else rewards_buffer[-size:],
            percentile=percentile,
            strict=strict
        )

        agent.partial_fit(elite_states, elite_actions)

        yield rewards_batch, log
