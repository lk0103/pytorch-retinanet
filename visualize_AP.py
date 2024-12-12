from matplotlib import pyplot as plt
import numpy as np

def plot(epoch_train_losses, train_name):
    plt.plot(epoch_train_losses, c='r', label='epoch train losses')
    plt.title(f'{train_name}')
    plt.legend()
    plt.savefig(f'{train_name}.png')
    plt.show()


def load_AP_log(file):
    epoch_loss = {}
    class_AP = []
    class_total_objects = []
    last_ep = 0
    ep = 0
    with open(file, 'r') as f:
        line = f.readline()

        while line != '':
            line.replace('\n', '')
            if line[0] == 'e':
                ep = int(line.split()[-1])
                if ep != 0:
                    epoch_loss[last_ep] = class_AP
                    class_AP = []
                    last_ep = ep
                line = f.readline()
                continue

            APs = line.replace('{', '').replace('}', '').split('), ')
            for AP_line in APs:
                elements = AP_line.replace('(', '').replace(')', '').replace(',', '').replace(':', '').split()
                AP = elements[1]
                class_AP.append(float(AP))
                if ep == 0:
                    class_total_objects.append(float(elements[-1]))
            line = f.readline()

    epoch_loss[last_ep] = class_AP
    print(last_ep)
    return epoch_loss, class_total_objects, last_ep + 1


def visualize_mean_AP(file):
    epoch_AP, class_total_objects, num_epochs = load_AP_log(file)
    average_AP = []
    weighted_average_AP = []
    for epoch in range(0, num_epochs):
        mean_epoch = np.mean(np.array(epoch_AP[epoch]))
        average_AP.append(mean_epoch)

        # weighted average
        weighted_avg = np.average(np.array(epoch_AP[epoch]), weights=np.array(class_total_objects))
        weighted_average_AP.append(weighted_avg)

    return average_AP, weighted_average_AP

average_AP, weighted_average_AP = visualize_mean_AP('training_mAP.txt')

print("average AP50 after epochs in training:", average_AP)
plot(average_AP, 'val_training_mean_AP')
# (weights are total number of objects of class in ground truth):
print("weighted average AP50 after epochs in training ",
          weighted_average_AP)
plot(weighted_average_AP, 'val_training_weighted_avg_AP')



average_AP, weighted_average_AP = visualize_mean_AP('eval_test_annots.csv_AP.txt')

print("average AP50 test dataset:", average_AP)
print("weighted average AP50 testing dataset", weighted_average_AP)
