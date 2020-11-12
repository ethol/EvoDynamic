
import tensorflow as tf
import numpy as np
import math
import evodynamic.experiment as experiment
import evodynamic.connection.cellular_automata as ca
import evodynamic.connection as connection
import evodynamic.connection.random as randon_conn
import evodynamic.cells.activation as act
import evodynamic.cells as cells

number_of_resovoirs = 4
resovoir_width = 20
iterations_between_input= 4
 
input_locations = np.random.randint(resovoir_width, size=number_of_resovoirs)
input_true_locations = []
for i in range(len(input_locations)):
  input_true_locations.append((i*resovoir_width)+(input_locations[i]))

width = number_of_resovoirs * resovoir_width
height_fig = 200
timesteps = 200

input_stream = np.ones(5, dtype=int)

exp = experiment.Experiment(input_delay=0, training_delay=5)
g_ca = exp.add_group_cells(name="g_ca", amount=width)
neighbors, center_idx = ca.create_pattern_neighbors_ca1d(3)
g_ca_bin = g_ca.add_binary_state(state_name='g_ca_bin', init="zeros")
g_ca_bin_conn = ca.create_conn_matrix_ca1d('g_ca_bin_conn',width,\
                                           neighbors=neighbors,\
                                           center_idx=center_idx)

input_connection = exp.add_input(tf.float64, [width], "input_connection")

fargs_list = [(a,) for a in [30]]

exp.add_connection("input_conn", connection.IndexConnection(input_connection,g_ca_bin,
                                                            np.arange(width)))


# exp.add_connection("input_conn", connection.IndexConnection(input_connection,g_ca_bin,
#                                                             input_true_locations))
exp.add_connection("g_ca_conn",
                   connection.WeightedConnection(g_ca_bin,g_ca_bin,
                                                 act.rule_binary_ca_1d_width3_func,
                                                 g_ca_bin_conn, fargs_list=fargs_list))

#exp.add_monitor("g_ca", "g_ca_bin")

exp.initialize_cells()

# Animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

idx_anim = 0  
im_ca = np.zeros((height_fig,width))
# im_ca[0] = exp.get_group_cells_state("g_ca", "g_ca_bin")

im = plt.imshow(im_ca, animated=True, vmax=1)

plt.title('Step: 0')


def updatefig(*args):
    global idx_anim, im_ca
    
    idx_anim += 1
    step = exp.get_group_cells_state("g_ca", "g_ca_bin")
    step_input = [0,0,0,0]
    # step = np.random.randint(2,size=(width,))
    
    if idx_anim % iterations_between_input == 0 and idx_anim/iterations_between_input < len(input_stream):
      step_input = []
      for i in range(len(input_locations)):
        inputPosition = (i*resovoir_width)+(input_locations[i])
        # step_input.append(float(int(step[inputPosition]) ^ input_stream[int(math.floor(idx_anim/iterations_between_input))]))
        step[inputPosition] = float(int(step[inputPosition]) ^ input_stream[int(math.floor(idx_anim/iterations_between_input))])
        

      print(step_input)
      print(input_true_locations)
    print(step.shape)
    exp.run_step(feed_dict={input_connection:step})

    step = exp.get_group_cells_state("g_ca", "g_ca_bin")

    if idx_anim < height_fig:
      im_ca[idx_anim] = step
      im.set_array(im_ca)
    else:
      im_ca = np.vstack((im_ca[1:], step))
      im.set_array(im_ca)

    plt.title('Step: '+str(idx_anim+1))

    return im,# ttl

ani = animation.FuncAnimation(fig, updatefig, frames=timesteps-2,\
                              interval=100, blit=False, repeat=False)

plt.show()

plt.connect('close_event', exp.close())