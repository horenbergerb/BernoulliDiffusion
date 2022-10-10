from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import os


def plot_loss(epochs, losses, save_dir, filename='loss_curve.png'):
    '''Given losses and epoch labels, this creates a nice plot.'''
    plt.plot(epochs, losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(save_dir, filename))
    plt.clf()

def plot_validation_proportions(epochs, proportions, save_dir, filename='validation.png'):
    for key, item in proportions.items():
        plt.plot(epochs, item, label=key)
    plt.title('Validation Proportions')
    plt.xlabel('Epoch')
    plt.ylabel('Proportion')
    plt.legend()
    plt.savefig(os.path.join(save_dir, filename))
    plt.clf()

def plot_evolution(epochs, examples, save_dir, step_name='Step', filename='evolution.gif'):
    '''Given a sequence of batches of samples, this animates their evolution.
    Useful for showing how a particular sample changes during training.
    Also useful for illustrating the reverse process.'''
    fig = plt.figure()
    im = plt.imshow(examples[0], interpolation='none')

    def init():
        fig.suptitle(step_name + ': 0')
        im.set_data(examples[0])
        return [im]

    def animate(i):
        fig.suptitle(step_name + ': {}'.format(epochs[i]))
        im.set_array(examples[i])
        return [im]

    # generate the animation
    ani = FuncAnimation(fig, animate, init_func=init,
                        frames=len(examples), interval=300, repeat=True) 
    
    ani.save(os.path.join(save_dir, filename), writer='imagemagick', fps=2)

    fig.clf()