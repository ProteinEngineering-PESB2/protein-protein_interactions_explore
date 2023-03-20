import matplotlib.pyplot as plt


class Plot:

    def graficar_loss(self, train, val, cutoff, name, method, type, num_epochs, transform_method):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(train, label='Train Loss')
        ax.plot(val, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_ylim((0, 1))
        # Agrega un título al gráfico
        ax.set_title(name + ' | Loss: cutoff ' + str(cutoff))
        ax.legend()
        if cutoff == 1:
            plt.savefig('plots/' + type + '/loss/' + transform_method + '/pip_' + type + '_' + method + '_' + str(num_epochs) + '_loss_' + str(cutoff) + '.png')
        else:
            plt.savefig('plots/' + type + '/loss/' + transform_method + '/pip_' + type + '_' + method + '_' + str(num_epochs) + '_loss_' + str(cutoff)[:4] + '.png')

    def graficar_accuracy(self, train, val, cutoff, name, method, type, num_epochs, transform_method):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(train, label='Train Accuracy')
        ax.plot(val, label='Validation Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_ylim((0, 1))
        # Agrega un título al gráfico
        ax.set_title(name + ' | Accuracy: cutoff ' + str(cutoff))
        ax.legend()
        if cutoff == 1:
            plt.savefig('plots/' + type + '/accuracy/' + transform_method + '/pip_' + type + '_' + method + '_' + str(num_epochs) + '_accuracy_' + str(cutoff) + '.png')
        else:
            plt.savefig('plots/' + type + '/accuracy/' + transform_method + '/pip_' + type + '_' + method + '_' + str(num_epochs) + '_accuracy_' + str(cutoff)[:4] + '.png')
    
    def graficar_loss_no_cutoff(self, train, val, name, method, type, num_epochs, transform_method):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(train, label='Train Loss')
        ax.plot(val, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        # Agrega un título al gráfico
        ax.set_title(name + ' | Loss')
        ax.legend()
        plt.savefig('plots/' + type + '/loss/' + transform_method + '/pip_' + type + '_' + method + '_' + str(num_epochs) + '_loss.png')

    # def graficar_accuracy_no_cutoff(self, train, val, name, method, type):
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     ax.plot(train, label='Train Accuracy')
    #     ax.plot(val, label='Validation Accuracy')
    #     ax.set_xlabel('Epoch')
    #     ax.set_ylabel('Accuracy')
    #     ax.set_ylim((0, 1))
    #     # Agrega un título al gráfico
    #     ax.set_title(name + ' | Accuracy')
    #     ax.legend()
    #     plt.savefig('plots/pip_' + type + '_' + method + '_accuracy.png')
