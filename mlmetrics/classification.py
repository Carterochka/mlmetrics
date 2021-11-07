from mlmetrics.abstract import Metrics
import keras

class Classifier(Metrics):
    def __init__(self, model) -> None:
        super().__init__(model)


    def compile_model(self, keras_metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()], *args, **kwargs):
        return super().compile_model(keras_metrics=keras_metrics, *args, **kwargs)

    
    def precision_plot(self, show=True, savefolder=None):
        import matplotlib.pyplot as plt

        plt.plot(range(1, len(self.history.history['precision']) + 1), self.history.history['precision'])
        
        if 'val_precision' in self.history.history.keys():
             plt.plot(range(1, len(self.history.history['val_precision']) + 1), self.history.history['val_precision'])
             plt.legend(['Train', 'Test'], loc='upper left')

        plt.xlabel('Epoch')
        plt.ylabel('Precision per epoch')
        if savefolder != None:
            plt.savefig(savefolder + '/precision_plot.png')
        if show:
            plt.show()


    def recall_plot(self, show=True, savefolder=None):
        import matplotlib.pyplot as plt
        plt.plot(range(1, len(self.history.history['recall']) + 1), self.history.history['recall'])

        if 'val_recall' in self.history.history.keys():
             plt.plot(range(1, len(self.history.history['val_recall']) + 1), self.history.history['val_recall'])
             plt.legend(['Train', 'Test'], loc='upper left')

        plt.xlabel('Epoch')
        plt.ylabel('Precision per epoch')
        if savefolder != None:
            plt.savefig(savefolder + '/recall_plot.png')
        if show:
            plt.show()


    def best_recall(self, show=True, savefolder=None):
        import numpy as np
        max_el = np.amax(self.history.history['recall'])
        max_el_ind = np.where(self.history.history['recall'] == max_el)

        if show:
            print('Best recall is ' + str(max_el) + ' on epoch ' + str(*max_el_ind) + '.')

        if savefolder != None:
            with open(savefolder + '/summary.txt', 'a+') as file:
                file.write('Best recall is ' + str(max_el) + ' on epoch ' + str(*max_el_ind) + '.\n')

        return (max_el, max_el_ind)


    def best_precision(self, show=True, savefolder=None):
        import numpy as np
        max_el = np.amax(self.history.history['precision'])
        max_el_ind = np.where(self.history.history['precision'] == max_el)

        if show:
            print('Best precision is ' + str(max_el) + ' on epochs ' + str(*max_el_ind) + '.')

        if savefolder != None:
            with open(savefolder + '/summary.txt', 'a+') as file:
                file.write('Best precision is ' + str(max_el) + ' on epochs ' + str(*max_el_ind) + '.\n')

        return (max_el, max_el_ind)


    def accuracy_plot(self, show=True, savefolder=None):
        import matplotlib.pyplot as plt
        plt.plot(range(1, len(self.history.history['accuracy']) + 1), self.history.history['accuracy'])

        if 'val_accuracy' in self.history.history.keys():
             plt.plot(range(1, len(self.history.history['val_accuracy']) + 1), self.history.history['val_accuracy'])
             plt.legend(['Train', 'Test'], loc='upper left')

        plt.xlabel('Epoch')
        plt.ylabel('Precision per epoch')
        if savefolder != None:
            plt.savefig(savefolder + '/accuracy_plot.png')
        if show:
            plt.show()


    def best_accuracy(self, show=True, savefolder=None):
        import numpy as np
        max_el = np.amax(self.history.history['accuracy'])
        max_el_ind = np.where(self.history.history['accuracy'] == max_el)

        if show:
            print('Best accuracy is ' + str(max_el) + ' on epoch ' + str(*max_el_ind) + '.')

        if savefolder != None:
            with open(savefolder + '/summary.txt', 'a+') as file:
                file.write('Best accuracy is ' + str(max_el) + ' on epochs ' + str(*max_el_ind) + '.\n')

        return (max_el, max_el_ind)

    