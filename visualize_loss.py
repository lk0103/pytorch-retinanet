from matplotlib import pyplot as plt

def plot_loss(epoch_train_losses, train_name):
    plt.plot(epoch_train_losses, c='r', label='epoch train losses')
    plt.title(f'{train_name} loss')
    plt.legend()
    plt.savefig(f'{train_name}_loss.png')
    plt.show()


def load_train_loss_log(file):
    epoch_loss = {}
    class_losses = []
    reg_losses = []
    total_losses = []
    last_ep = 0
    with open(file, 'r') as f:
        line = f.readline()

        while line != '':
            line.strip('\n')
            elements = line.split()
            ep = int(elements[1])
            if ep != last_ep:
                epoch_loss[last_ep] = [class_losses, reg_losses, total_losses]
                class_losses = []
                reg_losses = []
                total_losses = []
                last_ep = ep

            class_losses.append(float(elements[8]))
            reg_losses.append(float(elements[12]))
            total_losses.append(float(elements[-1]))
            line = f.readline()
    epoch_loss[last_ep] = [class_losses, reg_losses, total_losses]
    return epoch_loss


epoch_loss = load_train_loss_log('training_loss.txt')
class_losses = []
reg_losses = []
total_losses = []

for epoch in range(0, 4):
    class_losses.extend(epoch_loss[epoch][0])
    reg_losses.extend(epoch_loss[epoch][1])
    total_losses.extend(epoch_loss[epoch][2])
print(len(class_losses))

plot_loss(class_losses, 'classification')
plot_loss(reg_losses, 'bounding_box_regression')
plot_loss(total_losses, 'total')




num_iter_by_epoch = int(len(class_losses) / 4)
class_losses_epochs = []
reg_losses_epochs = []
total_losses_epochs = []

for epoch in range(0, 4):
    class_losses_epochs.append(class_losses[num_iter_by_epoch * epoch])
    reg_losses_epochs.append(reg_losses[num_iter_by_epoch * epoch])
    total_losses_epochs.append(total_losses[num_iter_by_epoch * epoch])

    if epoch == 3:
        class_losses_epochs.append(class_losses[-1])
        reg_losses_epochs.append(reg_losses[-1])
        total_losses_epochs.append(total_losses[-1])

plot_loss(class_losses_epochs, 'classification_epochs')
plot_loss(reg_losses_epochs, 'bounding_box_regression_epochs_')
plot_loss(total_losses_epochs, 'total_epochs')
print(class_losses_epochs)
print(reg_losses_epochs)
print(total_losses_epochs)

