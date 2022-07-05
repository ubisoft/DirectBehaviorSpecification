import copy
import pickle
import numpy as np
import time


def remove_nones(input_list):
    return [x for x in input_list if x is not None]


class Recorder(object):
    def __init__(self, metrics_to_record):
        """
        Simple object consisting of a recording tape (dictionary) that can be extended and saved
        Keys are strings and values are lists of recorded quantities
        (could be reward, loss, action, parameters, gradients, evaluation metric, etc.)
        """
        self.tape = {}

        for metric_name in metrics_to_record:
            self.tape[metric_name] = []

    def write_to_tape(self, new_values_dict):
        """
        Appends to tape all values for corresponding keys defined in dict
        If some keys present on tape do not have a new value in 'new_values_dict',
        we add None instead (so that all lists have the same length.
        """

        # new_values_dict is not allowed to contain un-initialised keys
        assert all([key in self.tape.keys() for key in new_values_dict.keys()]), \
            f"self.tape.keys()={self.tape.keys()}\nnew_values_dict.keys()={new_values_dict.keys()}"

        for key in self.tape.keys():
            if key in new_values_dict.keys():
                self.tape[key].append(copy.deepcopy(new_values_dict[key]))
            else:
                self.tape[key].append(None)

    def save(self, filename):
        """
        Saves the tape (dictionary) in .pkl file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.tape, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def init_from_pickle_file(cls, filename):
        """
        Initialises Recorder() from a .pkl file containing a tape (dictionary)
        """
        with open(filename, 'rb') as f:
            loaded_tape = pickle.load(f)
        instance = cls(metrics_to_record=loaded_tape.keys())
        instance.tape = loaded_tape
        return instance


class TrainingIterator(object):
    def __init__(self, max_itr, heartbeat_ite=np.inf, heartbeat_time=np.inf):
        """
        Container that allows to store temporary values in a dictionary self._data
        Typically used as iterable in a training loop to collect data from each iteration
        We then check the heartbeat of the TrainingIterator and when heartbeat==True, we use self.pop_all_means() to
        average the different data for each keys in self._data() and make them one datapoint on their respective plot.
        :param max_itr: (int) maximum number of iteration
        :param heartbeat_ite: (int) number of iterations between heartbeats
        :param heartbeat_time: (float) number of seconds between heartbeats (only works when used as iterable)
        """
        self._heartbeat = False
        self._itr = 0

        self.max_itr = max_itr
        self.heartbeat_ite = heartbeat_ite
        self.heartbeat_time = heartbeat_time

        self._data = {}

    @property
    def itr(self):
        return self._itr

    @property
    def heartbeat(self):
        if self._heartbeat:
            self._heartbeat = False
            return True
        else:
            return False

    @property
    def elapsed(self):
        return self._elapsed

    def itr_message(self):
        return f'==> Itr {self.itr + 1}/{self.max_itr} (elapsed:{self.elapsed:.2f})'

    def record(self, key, value):
        if key in self._data:
            self._data[key].append(value)
        else:
            self._data[key] = [value]

    def update(self, dict):
        for (key, value) in dict.items():
            self.record(key, value)

    def pop(self, key):
        vals = self._data.get(key, [])
        del self._data[key]
        return vals

    def pop_mean(self, key):
        return np.mean(self.pop(key))

    def pop_all_means(self):
        data_points = {}
        for key in dict(self._data):
            data_points.update({key: self.pop_mean(key)})
        return data_points

    def __iter__(self):
        """
        Iterate until self.mex_itr is reached (called automatically in a loop)
        """
        prev_time = time.time()
        self._heartbeat = False
        for i in range(self.max_itr):
            self._itr = i
            cur_time = time.time()

            if (cur_time - prev_time) > self.heartbeat_time \
                    or (self.itr % self.heartbeat_ite == 0) \
                    or (i == self.max_itr - 1):

                self._heartbeat = True
                self._elapsed = cur_time - prev_time
                prev_time = cur_time

            # self.itr_message()
            yield self
            self._heartbeat = False

    def touch(self):
        """
        Increase iteration by one (called by hand)
        """
        self._itr += 1

        if (self.itr == self.max_itr) or (self.itr % self.heartbeat_ite == 0):
            self._heartbeat = True
