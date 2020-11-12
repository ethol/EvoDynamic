import tensorflow as tf
import numpy as np
import random
import time
import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.connection as connection
import evodynamic.cells.activation as act
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import helpers.helper as helper


class ReservoirMemorySingleExperiment:
    def __init__(self, bits, r, itr, r_width, d_period, ca_rule):

        self.recurrence = r
        self.iterations_between_input = itr
        self.reservoir_height = itr
        self.recurrence_width = r_width
        self.distractor_period = d_period
        self.distractor_period_input_output = d_period + (2 * bits)
        self.number_of_bits = bits
        self.ca_rule = ca_rule
        self.reg = SVC(kernel="linear")
        self.input_channels = 4

        self.ca_height = self.distractor_period_input_output * self.iterations_between_input
        self.ca_width = self.recurrence * self.recurrence_width

        self.input_true_locations = self.create_input_locations()
        evo = self.set_up_evodynamics()
        self.evo_exp = evo[0]
        self.input_connections = evo[1]

        self.x_training = []
        self.x_labels = []
        self.exp_history = []
        self.exp_memory_history = []

    def create_input_locations(self):
        input_true_locations = []
        for i in range(self.recurrence):
            input_locations = np.add(random.sample(range(self.recurrence_width), self.input_channels),
                                     i * self.recurrence_width)
            input_true_locations.extend(input_locations)
        return input_true_locations

    def create_input_streams(self, input_array):
        input_streams = []

        input_stream = np.zeros(self.distractor_period_input_output, dtype=int)
        input_stream[:len(input_array)] = input_array
        input_streams.append(input_stream.tolist())

        input_stream = np.zeros(self.distractor_period_input_output, dtype=int)
        input_stream[:len(input_array)] = np.bitwise_xor(input_array, 1)
        input_streams.append(input_stream.tolist())

        input_stream = np.ones(self.distractor_period_input_output, dtype=int)
        input_stream[:len(input_array)] = np.zeros(5)
        input_stream[self.distractor_period_input_output - len(input_array) - 1] = 0
        input_streams.append(input_stream.tolist())

        input_stream = np.zeros(self.distractor_period_input_output, dtype=int)
        input_stream[self.distractor_period_input_output - len(input_array) - 1] = 1
        input_streams.append(input_stream.tolist())

        return input_streams

    def create_output_streams(self, input_array):
        output_streams = []

        output_stream = np.zeros(self.distractor_period_input_output, dtype=int)
        output_stream[-len(input_array):] = input_array
        output_streams.append(output_stream.tolist())

        output_stream = np.zeros(self.distractor_period_input_output, dtype=int)
        output_stream[-len(input_array):] = np.bitwise_xor(input_array, 1)
        output_streams.append(output_stream.tolist())

        output_stream = np.ones(self.distractor_period_input_output, dtype=int)
        output_stream[-len(input_array):] = np.zeros(5)
        output_streams.append(output_stream.tolist())

        return output_streams

    def set_up_evodynamics(self):
        fargs_list = [(a,) for a in [self.ca_rule]]

        exp = experiment.Experiment(input_start=0, input_delay=0, training_delay=5)
        g_ca = exp.add_group_cells(name="g_ca", amount=self.ca_width)
        neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
        g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin', init="zeros")
        g_ca_bin_conn = ca.create_conn_matrix_ca1d(
            'g_ca_bin_conn',
            self.ca_width,
            neighbors=neighbors,
            center_idx=center_idx)

        input_connection = exp.add_input(tf.float64, [self.ca_width], "input_connection")

        exp.add_connection(
            "input_conn",
            connection.IndexConnection(
                input_connection,
                g_ca_bin,
                np.arange(self.ca_width)))

        exp.add_connection(
            "g_ca_conn",
            connection.WeightedConnection(
                g_ca_bin,
                g_ca_bin,
                act.rule_binary_ca_1d_width3_func,
                g_ca_bin_conn,
                fargs_list=fargs_list))

        # exp.add_monitor("g_ca", "g_ca_bin")

        exp.initialize_cells()
        return exp, input_connection

    def run(self):
        for bits in range(0, 32):
            input_array = helper.int_to_binary_string(bits, self.number_of_bits)
            self.run_bit_string(input_array)

        self.reg.fit(self.x_training, self.x_labels)

        this_score = self.reg.score(self.x_training, self.x_labels)

        self.evo_exp.close()

        return this_score

    def run_bit_string(self, input_array):
        short_term_history = np.zeros((self.reservoir_height, self.ca_width), dtype=int).tolist()
        run_ca = np.zeros((self.ca_height, self.ca_width))

        input_streams = self.create_input_streams(input_array)
        output_streams_labels = self.create_output_streams(input_array)

        for i in range(0, self.ca_height):
            self.run_step(i, input_streams, output_streams_labels, short_term_history, run_ca)

    def run_step(self, i, input_streams, output_streams_labels, short_term_history, run_ca):
        g_ca_bin_current = self.evo_exp.get_group_cells_state("g_ca", "g_ca_bin")
        step = g_ca_bin_current[:, 0]

        if i % self.iterations_between_input == 0:
            input_bits = helper.pop_all_lists(input_streams)
            for j in range(len(self.input_true_locations)):
                input_bit = input_bits[j % 4]
                step[self.input_true_locations[j]] = float(int(step[self.input_true_locations[j]]) ^ input_bit)

        short_term_history.append(step)
        short_term_history = short_term_history[-self.reservoir_height:]

        if i % self.iterations_between_input == 0:
            correct_answer = helper.pop_all_lists(output_streams_labels)

            self.x_training.append(helper.flatten_list_of_lists(short_term_history))

            if correct_answer[0] == 1:
                self.x_labels.append(0)
            elif correct_answer[1] == 1:
                self.x_labels.append(1)
            else:
                self.x_labels.append(2)

        if i < self.ca_height:
            run_ca[i] = step
        else:
            run_ca = np.vstack((run_ca[1:], step))
        # self.exp_history.append(run_ca.copy())
        # self.exp_memory_history.append(short_term_history.copy())

        self.evo_exp.run_step(feed_dict={self.input_connections: step.reshape((-1, 1))})

        _ = self.evo_exp.get_group_cells_state("g_ca", "g_ca_bin")

    def show_visuals(self):

        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.animation as animation

        def updatefig(*args):
            im.set_array(self.exp_history[self.idx_anim])
            im2.set_array(self.exp_memory_history[self.idx_anim])
            if self.idx_anim % self.iterations_between_input == 0:
                pred = self.reg.predict([self.x_training[self.idx_anim // self.iterations_between_input]])
                # print(pred)
                # print(x_labels[200])
                if pred == 0:
                    im3.set_array([[1, 0, 0]])
                elif pred == 1:
                    im3.set_array([[0, 1, 0]])
                else:
                    im3.set_array([[0, 0, 1]])
                # im3.set_array([list(map(round, pred[0]))])
                ax3.set_title("model prediction: " + str(pred))
                cor = self.x_labels[self.idx_anim // self.iterations_between_input]
                if cor == 0:
                    im4.set_array([[1, 0, 0]])
                elif cor == 1:
                    im4.set_array([[0, 1, 0]])
                else:
                    im4.set_array([[0, 0, 1]])
                # im4.set_array([x_labels[idx_anim // (iterations_between_input + 1)]])
            fig.suptitle(
                'Step: ' + str(self.idx_anim % self.ca_height) + " Exp: " + str(self.idx_anim // self.ca_height))
            self.idx_anim += 1

        fig = plt.figure()
        gs = fig.add_gridspec(4, 8)
        ax1 = fig.add_subplot(gs[:-1, :-1])
        ax1.set_title("reservoir full history")
        ax2 = fig.add_subplot(gs[3, :-1])
        ax2.set_title("model perceived history")
        ax3 = fig.add_subplot(gs[:-2, 7])
        ax3.set_title("model prediction")
        ax3.axis("off")
        ax4 = fig.add_subplot(gs[2:, 7])
        ax4.set_title("model desired output")
        ax4.axis("off")

        im_ca = np.zeros((self.ca_height, self.ca_width))

        shortTermHistory = np.zeros((self.reservoir_height, self.ca_width), dtype=int).tolist()

        im = ax1.imshow(im_ca, animated=True, vmax=1)
        im2 = ax2.imshow(shortTermHistory, animated=True, vmax=1)
        im3 = ax3.imshow(np.zeros((1, 3), dtype=int).tolist(), animated=True, vmax=1)
        im4 = ax4.imshow(np.zeros((1, 3), dtype=int).tolist(), animated=True, vmax=1)

        fig.suptitle('Step: 0 Exp: 0')

        print(self.input_true_locations)

        # implement as list of arrays instead?

        self.idx_anim = 0
        ani = animation.FuncAnimation(
            fig,
            updatefig,
            frames=(self.ca_height - 1) * 32,
            interval=200,
            blit=False,
            repeat=False
        )

        plt.show()

        # plt.connect('close_event', self.exp.close())


start_time = time.time()
scores = []
for expRun in range(0, 100):
    print("starting exp nr" + str(expRun))
    start_time_sub = time.time()
    exp = ReservoirMemorySingleExperiment(bits=5, r=4, itr=2, r_width=40, d_period=200, ca_rule=90)
    score = exp.run()
    # exp.show_visuals()

    print(score)
    scores.append(score)
    this_runtime = time.time() - start_time_sub
    print(this_runtime)

print(time.time() - start_time)
print(sum(map(lambda i: i == 1.0, scores)))
