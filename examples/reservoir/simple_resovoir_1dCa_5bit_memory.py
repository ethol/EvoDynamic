import tensorflow as tf
import numpy as np
import math
import random
import time
import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.connection as connection
import evodynamic.connection.random as randon_conn
import evodynamic.cells.activation as act
import evodynamic.cells as cells
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC

runtime = []
score = []
for expRun in range(0, 100):
    start_time = time.time()
    reg = SVC(kernel="linear")
    x_training = []
    x_labels = []
    exp_history = []
    exp_memory_history = []

    number_of_reservoirs = 8
    reservoir_width = 40
    iterations_between_input = 4
    historyMemory = iterations_between_input
    input_channels = 4
    input_array = [0, 0, 0, 0, 0]
    # distractor_period = 200 //(iterations_between_input+1)
    distractor_period = 200
    distractor_period_input_output = (distractor_period + 2 * len(input_array))

    width = number_of_reservoirs * reservoir_width
    height_fig = distractor_period_input_output * (iterations_between_input + 1)
    timesteps = height_fig

    fargs_list = [(a,) for a in [30]]

    input_true_locations = []
    for i in range(number_of_reservoirs):
        input_locations = np.add(random.sample(range(reservoir_width), input_channels), i * reservoir_width)
        input_true_locations.extend(input_locations)

    #math for 32
    for bits in range(0, 32):

        binArray = list(str(int(bin(bits)[2:])))
        binArray = list(map(int, binArray))
        input_array = np.zeros(5, dtype=int)
        input_array[-len(binArray):] = binArray
        # print(input_array)

        # input_locations = np.random.randint(resovoir_width, size=number_of_resovoirs*4)

        input_streams = []

        input_no_infix = np.zeros(-(-timesteps // iterations_between_input), dtype=int)
        input_no_infix[:len(input_array)] = input_array

        input_stream = np.zeros(timesteps, dtype=int)
        input_stream[::iterations_between_input] = input_no_infix

        input_streams.append(input_stream)

        input_no_infix = np.zeros(-(-timesteps // iterations_between_input), dtype=int)
        input_no_infix[:len(input_array)] = np.bitwise_xor(input_array, 1)

        input_stream = np.zeros(timesteps, dtype=int)
        input_stream[::iterations_between_input] = input_no_infix

        input_streams.append(input_stream)

        input_no_infix = np.ones(-(-timesteps // iterations_between_input), dtype=int)
        input_no_infix[:len(input_array)] = np.zeros(5)

        input_stream = np.zeros(timesteps, dtype=int)
        input_stream[::iterations_between_input] = input_no_infix
        input_stream[timesteps - ((iterations_between_input + 1) * len(input_array)) - 1] = 0

        input_streams.append(input_stream)

        input_stream = np.zeros(timesteps, dtype=int)
        input_stream[timesteps - ((iterations_between_input + 1) * len(input_array)) - 1] = 1

        input_streams.append(input_stream)

        output_streams_correct = []

        output_stream = np.zeros(distractor_period_input_output, dtype=int)
        output_stream[-len(input_array):] = input_array
        output_streams_correct.append(output_stream.tolist())

        output_stream = np.zeros(distractor_period_input_output, dtype=int)
        output_stream[-len(input_array):] = np.bitwise_xor(input_array, 1)
        output_streams_correct.append(output_stream.tolist())

        output_stream = np.ones(distractor_period_input_output, dtype=int)
        output_stream[-len(input_array):] = np.zeros(5)
        output_streams_correct.append(output_stream.tolist())

        exp = experiment.Experiment(input_start=0, input_delay=0, training_delay=5)
        g_ca = exp.add_group_cells(name="g_ca", amount=width)
        neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
        g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin', init="zeros")
        g_ca_bin_conn = ca.create_conn_matrix_ca1d(
            'g_ca_bin_conn',
            width,
            neighbors=neighbors,
            center_idx=center_idx)

        input_connection = exp.add_input(tf.float64, [width], "input_connection")

        exp.add_connection(
            "input_conn",
            connection.IndexConnection(
                input_connection,
                g_ca_bin,
                np.arange(width)))

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

        shortTermHistory = np.zeros((historyMemory, width), dtype=int).tolist()
        run_ca = np.zeros((height_fig, width))

        for i in range(0, timesteps):
            g_ca_bin_current = exp.get_group_cells_state("g_ca", "g_ca_bin")
            step = g_ca_bin_current[:, 0]
            for j in range(len(input_true_locations)):
                step[input_true_locations[j]] = float(int(step[input_true_locations[j]]) ^ input_streams[j % 4][i])

            shortTermHistory.append(step)
            shortTermHistory = shortTermHistory[-historyMemory:]

            if i % (iterations_between_input + 1) == 0:
                correct_answer = np.ones(3, dtype=int)
                correct_answer[0] = output_streams_correct[0].pop(0)
                correct_answer[1] = output_streams_correct[1].pop(0)
                correct_answer[2] = output_streams_correct[2].pop(0)

                flat_list = []
                for sublist in shortTermHistory:
                    for item in sublist:
                        flat_list.append(item)
                # reg.fit([flat_list], [correct_answer])
                x_training.append(flat_list)
                if correct_answer[0] == 1:
                    x_labels.append(0)
                elif correct_answer[1] == 1:
                    x_labels.append(1)
                else:
                    x_labels.append(2)

            if i < height_fig:
                run_ca[i] = step
            else:
                run_ca = np.vstack((run_ca[1:], step))
            # exp_history.append(run_ca.copy())
            # exp_memory_history.append(shortTermHistory.copy())

            exp.run_step(feed_dict={input_connection: step.reshape((-1, 1))})

            g_ca_bin_current = exp.get_group_cells_state("g_ca", "g_ca_bin")

    reg.fit(x_training, x_labels)

    this_score = reg.score(x_training, x_labels)
    score.append(this_score)

    this_runtime = time.time() - start_time
    runtime.append(this_runtime)

    print("run nr " + str(expRun))
    print(this_score)
    print(str(this_runtime) + " seconds")

    exp.close()


print(score)
print(runtime)

print(sum(map(lambda i: i == 1.0, score)))

#
# def updatefig(*args):
#     global idx_anim
#     im.set_array(exp_history[idx_anim])
#     im2.set_array(exp_memory_history[idx_anim])
#     if idx_anim % (iterations_between_input + 1) == 0:
#         pred = reg.predict([x_training[idx_anim // (iterations_between_input + 1)]])
#         # print(pred)
#         # print(x_labels[200])
#         if pred == 0:
#             im3.set_array([[1, 0, 0]])
#         elif pred == 1:
#             im3.set_array([[0, 1, 0]])
#         else:
#             im3.set_array([[0, 0, 1]])
#         # im3.set_array([list(map(round, pred[0]))])
#         ax3.set_title("model prediction: " + str(pred))
#         cor = x_labels[idx_anim // (iterations_between_input + 1)]
#         if cor == 0:
#             im4.set_array([[1, 0, 0]])
#         elif cor == 1:
#             im4.set_array([[0, 1, 0]])
#         else:
#             im4.set_array([[0, 0, 1]])
#         # im4.set_array([x_labels[idx_anim // (iterations_between_input + 1)]])
#     fig.suptitle('Step: ' + str(idx_anim % timesteps) + " Exp: " + str(idx_anim // timesteps))
#     idx_anim += 1
#
#
# # Animation
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.animation as animation
#
# fig = plt.figure()
# gs = fig.add_gridspec(4, 8)
# ax1 = fig.add_subplot(gs[:-1, :-1])
# ax1.set_title("reservoir full history")
# ax2 = fig.add_subplot(gs[3, :-1])
# ax2.set_title("model perceived history")
# ax3 = fig.add_subplot(gs[:-2, 7])
# ax3.set_title("model prediction")
# ax3.axis("off")
# ax4 = fig.add_subplot(gs[2:, 7])
# ax4.set_title("model desired output")
# ax4.axis("off")
#
# im_ca = np.zeros((height_fig, width))
#
# shortTermHistory = np.zeros((historyMemory, width), dtype=int).tolist()
#
# im = ax1.imshow(im_ca, animated=True, vmax=1)
# im2 = ax2.imshow(shortTermHistory, animated=True, vmax=1)
# im3 = ax3.imshow(np.zeros((1, 3), dtype=int).tolist(), animated=True, vmax=1)
# im4 = ax4.imshow(np.zeros((1, 3), dtype=int).tolist(), animated=True, vmax=1)
#
# fig.suptitle('Step: 0 Exp: 0')
#
# print(input_true_locations)
#
# # implement as list of arrays instead?
#
# idx_anim = 0
# ani = animation.FuncAnimation(
#     fig,
#     updatefig,
#     frames=(timesteps - 1) * 32,
#     interval=200,
#     blit=False,
#     repeat=False
# )
#
# plt.show()
#
# plt.connect('close_event', exp.close())
