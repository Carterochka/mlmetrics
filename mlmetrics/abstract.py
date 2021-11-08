from abc import ABC, abstractmethod
import time

from matplotlib.pyplot import savefig
import keras

# Time tracking recall for Keras model training
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


# Abstract class for metrics tracking
class Metrics(ABC):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.timehistory_callback = TimeHistory()


    # For usage in "with" block
    def __enter__(self):
        return self


    def __exit__(self, type, value, tb):
        del self


    # Function for model training
    def fit(self, *args, **kwargs):
        # Including time tracking callback in the model training
        if 'callbacks' in kwargs.keys():
            kwargs['callbacks'].append(self.timehistory_callback)
        else:
            kwargs['callbacks'] = [self.timehistory_callback]

        self.history = self.model.fit(*args, **kwargs)
        return self.history


    # UI for defining metrics to track and their display order
    def get_metrics(self, metrics_list: list = [], show=True, save=False):
        if len(metrics_list) == 0:
            return None

        # If saving option chosen, creating a folder with current date time title
        savefolder = None
        if save:
            from datetime import datetime
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y %H.%M.%S")
            savefolder = 'Summary ' + dt_string

            import os
            os.mkdir(savefolder)

        # Calling requested metrics one by one
        for metric in metrics_list:
            metric_attr = getattr(self, metric, None)
            if metric_attr == None:
                print('Metric ', metric, ' is not present in our dictionary of metris.')
                continue

            metric_attr(show, savefolder)
    

    def avg_training_time(self, show=True, savefolder=None):
        import numpy as np
        avg_time = np.mean(self.timehistory_callback.times)
        if show:
            print('Average training time is', avg_time, 's.')

        if savefolder != None:
            with open(savefolder + '/summary.txt', 'a+') as file:
                file.write('Average training time is ' + str(avg_time) + ' s.\n')

        return avg_time

    
    def total_training_time(self, show=True, savefolder=None):
        import numpy as np
        total_time = np.sum(self.timehistory_callback.times)
        if show:
            print('Total training time is', total_time, 's.')

        if savefolder != None:
            with open(savefolder + '/summary.txt', 'a+') as file:
                file.write('Total training time is ' + str(total_time) + ' s.\n')
        
        return total_time
    

    def training_time_plot(self, show=True, savefolder=None):
        import matplotlib.pyplot as plt

        plt.plot(range(1, len(self.timehistory_callback.times) + 1), self.timehistory_callback.times)
        plt.xlabel('Epoch')
        plt.ylabel('Training time per epoch (s)')
        if savefolder != None:
            plt.savefig(savefolder + '/training_time_plot.png')
        if show:
            plt.show()


    def loss_plot(self, show=True, savefolder=None):
        import matplotlib.pyplot as plt
        plt.plot(range(1, len(self.history.history['loss']) + 1), self.history.history['loss'])

        if 'val_loss' in self.history.history.keys():
             plt.plot(range(1, len(self.history.history['val_loss']) + 1), self.history.history['val_loss'])
             plt.legend(['Train', 'Test'])

        plt.xlabel('Epoch')
        plt.ylabel('Loss per epoch')
        if savefolder != None:
            plt.savefig(savefolder + '/loss_plot.png')
        if show:
            plt.show()


    def best_loss(self, show=True, savefolder=None):
        import numpy as np
        min_el = np.amin(self.history.history['loss'])
        min_el_ind = np.where(self.history.history['loss'] == min_el)

        if show:
            print('Best loss is ' + str(min_el) + ' on epoch ' + str(*min_el_ind) + '.')

        if savefolder != None:
            with open(savefolder + '/summary.txt', 'a+') as file:
                file.write('Best loss is ' + str(min_el) + ' on epoch ' + str(*min_el_ind) + '.\n')

        return (min_el, min_el_ind)


    def compile_model(self, keras_metrics=[], *args, **kwargs):
        if 'metrics' not in kwargs.keys():
            kwargs['metrics'] = keras_metrics
        else:
            for metric in keras_metrics:
                if metric not in kwargs['metrics']:
                    kwargs['metrics'].append(metric)

        self.model.compile(*args, **kwargs)