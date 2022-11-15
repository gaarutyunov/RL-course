
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """

    Q = 0

    for s, _ in mdp.get_next_states(state, action).items():
        v = float(state_values[s])
        p = mdp.get_transition_prob(state, action, s)
        r = mdp.get_reward(state, action, s)
        Q += (p * (r + gamma * v))

    return Q
