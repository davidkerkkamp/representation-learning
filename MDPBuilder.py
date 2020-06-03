import json
import random
import pickle

'''
The MDPBuilder class can be used to collect observations and build a Markov Decision Process. The information needed to
create an MDP is stored in a structure of dicts where every distinct observation (in the form of a dict with 
label - value pairs) is treated as a state and assigned a unique numeric ID. The MDPBuilder keeps a count of how many 
times the same transitions occur between states. The MDPBuilder state can be directly saved to a file (using pickle),
so building of an MDP can be continued later. An MDP can be generated using either PRISM or JANI (JSON) format. The 
probability of a transition is calculated as the fraction of the total number of transitions of the action. The 
accuracy of these probabilities ofcourse depends on the number of observations used to create the MDP. For usage 
examples, see the run_atari.py file.
'''


class MDPBuilder:
    def __init__(self, labels, actions, log=True, probability_decimals=4):
        self.fresh_state_id = 0
        self.prev_state_id = None
        self.states_ids = {}
        self.states = {}
        self.initial_states = set()
        self.labels = labels
        self.actions = actions
        self.log = log
        self.probability_decimals = probability_decimals

    def num_states(self):
        return len(self.states)

    def get_random_action(self):
        return random.choice(self.actions)

    def get_fresh_state_id(self):
        r = self.fresh_state_id
        self.fresh_state_id += 1
        return r

    def insert_state(self, label_dict, state_rep_hash):
        fid = self.get_fresh_state_id()
        self.log_message(f'Adding new state with ID {fid}')
        self.states_ids[state_rep_hash] = fid
        s = {
            'labels': label_dict,
            'transitions': {
                a: {} for a in self.actions
            }
        }
        self.states[fid] = s
        return fid

    def get_state_id(self, label_dict):
        state_rep_hash = hash(frozenset(label_dict.items()))
        s_id = self.states_ids.get(state_rep_hash)
        return s_id if s_id is not None else self.insert_state(label_dict, state_rep_hash)

    def add_transition(self, from_id, to_id, action):
        if action not in self.actions:
            raise Exception(f'Action value {action} was provided, but does not occur in MDP actions')
        s = self.states.get(from_id)
        if to_id in s['transitions'][action]:
            s['transitions'][action][to_id] += 1
        else:
            s['transitions'][action][to_id] = 1

    def add_state_info(self, label_dict, action):
        label_dict = {k: v for k, v in label_dict.items() if k in self.labels}
        s_id = self.get_state_id(label_dict)
        if self.prev_state_id is not None:
            self.add_transition(self.prev_state_id, s_id, action)
        else:
            self.initial_states.add(s_id)
        self.prev_state_id = s_id

    def log_message(self, msg):
        if self.log:
            print(msg)

    def save_builder_to_file(self, path):
        info = {
            'fresh_state_id': self.fresh_state_id,
            'states_ids': self.states_ids,
            'states': self.states,
            'initial_states': self.initial_states,
            'labels': self.labels,
            'actions': self.actions
        }
        pickle.dump(info, open(path, 'wb'))

    def load_from_file(self, path):
        info = pickle.load(open(path, 'rb'))
        self.fresh_state_id = info['fresh_state_id']
        self.states_ids = info['states_ids']
        self.states = info['states']
        self.initial_states = info['initial_states']
        self.labels = info['labels']
        self.actions = info['actions']
        self.restart()

    def restart(self):
        self.prev_state_id = None

    def get_prism_commands(self):
        commands = []
        for s_id, info in self.states.items():
            guard_str = '[] '
            guards = []
            for l, v in info['labels'].items():
                guards.append(f'{l}={v}')
            guard_str += (' & '.join(guards))
            guard_str += ' -> '

            for action, transitions in info['transitions'].items():
                n_transitions = len(transitions)
                total_n = 1
                if n_transitions == 0:
                    continue
                elif n_transitions > 1:
                    total_n = sum([n for n in transitions.values()])

                update_probs = []
                for t_id, n in transitions.items():
                    updates = []
                    for l, v in self.states[t_id]['labels'].items():
                        updates.append(f"({l}'={v})")

                    prob_str = ''
                    if n_transitions > 1:
                        prob_str = f'{round(n / total_n, self.probability_decimals)} : '
                    update_probs.append(f'{prob_str}{" & ".join(updates)}')
                commands.append(f'{guard_str}{" + ".join(update_probs)};')
        return commands

    def get_jani_guard(self, label_list):
        if len(label_list) == 0:
            return {}
        guard = {
            'op': '=',
            'left': label_list[0][0],
            'right': int(label_list[0][1])
        }

        if len(label_list) == 1:
            return guard
        return {
            'op': '∧',
            'left': self.get_jani_guard(label_list[1:]),
            'right': guard
        }

    def get_jani_automaton(self):
        aut = {
            'name': 'atari_game',
            'locations': [{'name': 'l'}],
            'initial-locations': ['l'],
            'edges': []
        }

        for s_id, info in self.states.items():
            guards = {'exp': self.get_jani_guard(list(info['labels'].items()))}
            for action, transitions in info['transitions'].items():
                n_transitions = len(transitions)
                total_n = 1
                if n_transitions == 0:
                    continue
                elif n_transitions > 1:
                    total_n = sum([n for n in transitions.values()])

                edge = {
                    'location': 'l',
                    'action': str(action),
                    'guard': guards,
                    'destinations': []
                }
                for t_id, n in transitions.items():
                    dest = {
                        'location': 'l',
                        'probability': {
                            'exp': round(n / total_n, self.probability_decimals) if n_transitions > 1 else 1
                        },
                        'assignments': []
                    }
                    for l, v in self.states[t_id]['labels'].items():
                        dest['assignments'].append({
                            'ref': l,
                            'value': int(v)
                        })
                    edge['destinations'].append(dest)
                aut['edges'].append(edge)
        return aut

    def build_jani_model(self, file_path):
        init_state = self.states[next(iter(self.initial_states))]['labels']
        jani = {
            'jani-version': 1,
            'name': 'atari-jani-model',
            'type': 'mdp',
            'features': ['derived-operators'],
            'actions': [{'name': str(action)} for action in self.actions],
            'variables': [{'name': label,
                           'type': {'kind': 'bounded', 'base': 'int', 'lower-bound': 0, 'upper-bound': 256}}
                          for label in self.labels],
            'restrict-initial': {'exp': self.get_jani_guard(list(init_state.items()))},
            'properties': [],
            'automata': [self.get_jani_automaton()],
            'system': {
                'elements': [{'automaton': 'atari_game'}]
            }
        }
        json.dump(jani, open(file_path, "w"))

    def build_prism_model(self, file_path):
        f = open(file_path, "w")
        f.write('mdp\n\n')
        f.write('module atari_game\n')
        indent_fmt = '    {}\n'
        var_fmt = indent_fmt.format('{} : [0..256] init {};')
        init_state = self.states[next(iter(self.initial_states))]['labels']
        for l, v in init_state.items():
            f.write(var_fmt.format(l, v))
        commands = self.get_prism_commands()
        for c in commands:
            f.write(indent_fmt.format(c))
        f.write('endmodule')
        f.close()

    def build_model_file(self, file_path, format='prism'):
        if format == 'prism':
            self.build_prism_model(file_path)
        elif format == 'jani':
            self.build_jani_model(file_path)
        else:
            raise Exception(f"Model format '{format}' is not supported")
