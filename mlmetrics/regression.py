from mlmetrics.abstract import Metrics

class Regressor(Metrics):
    def __init__(self, model) -> None:
        super().__init__(model)


    def compile_model(self, keras_metrics=['mse'], *args, **kwargs):
        return super().compile_model(keras_metrics=keras_metrics, *args, **kwargs)

    
    def mse_plot(self, show=True, savefolder=None):
        import matplotlib.pyplot as plt
        plt.plot(range(1, len(self.history.history['mse']) + 1), self.history.history['mse'])

        if 'val_recall' in self.history.history.keys():
             plt.plot(range(1, len(self.history.history['val_mse']) + 1), self.history.history['val_mse'])
             plt.legend(['Train', 'Test'], loc='upper left')

        plt.xlabel('Epoch')
        plt.ylabel('MSE per epoch')
        if savefolder != None:
            plt.savefig(savefolder + '/mse_plot.png')
        if show:
            plt.show()


    def best_mse(self, show=True, savefolder=None):
        import numpy as np
        min_el = np.amax(self.history.history['mse'])
        min_el_ind = np.where(self.history.history['mse'] == min_el)

        if show:
            print('Best mse is ' + str(min_el) + ' on epoch ' + str(*min_el_ind) + '.')

        if savefolder != None:
            with open(savefolder + '/summary.txt', 'a+') as file:
                file.write('Best mse is ' + str(min_el) + ' on epoch ' + str(*min_el_ind) + '.\n')

        return (min_el, min_el_ind)