import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    font_size = 12
    resolution = 300
    plt.rcParams.update({'font.size': font_size})

    lstm_iil = np.load('lstm_iil.npy')
    tr_iil = np.load('tr_iil.npy')
    lstm = np.load('lstm_acc.npy')
    transformer = np.load('tr_acc.npy')
    lstm_times = np.load('lstm_times.npy')
    tr_times = np.load('tr_times.npy')
    greedy = np.load('greedy.npy')

    plt.figure()
    plt.title('Test Mean Absolute Error')
    plt.ylabel('Test MAE')
    plt.xlabel('Input Interval Length')
    plt.plot([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24], lstm_iil, 'v-', label='LSTM')
    plt.plot([1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24], tr_iil, 'v-', label='Transformers')
    plt.legend()
    plt.savefig('input interval length.png', dpi=resolution)

    plt.figure()
    plt.title('Optimal Level of Risk')
    plt.ylabel('Test MAE')
    plt.xlabel('c Value')
    plt.autoscale(axis='x', tight=True)
    plt.plot([k for k in range(0, 50, 5)], greedy, 'v-')
    plt.savefig('greedy.png', dpi=resolution)
    
    plt.figure()
    plt.title('Validation Mean Absolute Error')
    plt.ylabel('Validation MAE')
    plt.xlabel('Epoch')
    plt.autoscale(axis='x',tight=True)
    plt.plot(lstm[:,0], label='LSTM')
    plt.plot(transformer[:,0], label='Transformers')
    plt.legend()
    plt.savefig('MAE.png', dpi=resolution)
    
    plt.figure()
    plt.title('Validation Mean Absolute Percentage Error')
    plt.ylabel('Validation MAPE')
    plt.xlabel('Epoch')
    plt.autoscale(axis='x',tight=True)
    plt.plot(lstm[:,1], label='LSTM')
    plt.plot(transformer[:,1], label='Transformers')
    plt.legend()
    plt.savefig('MAPE.png', dpi=resolution)
    
    plt.figure()
    plt.title('Validation Root Mean Square Error')
    plt.ylabel('Validation RMSE')
    plt.xlabel('Epoch')
    plt.autoscale(axis='x',tight=True)
    plt.plot(lstm[:,2], label='LSTM')
    plt.plot(transformer[:,2], label='Transformers')
    plt.legend()
    plt.savefig('RMSE.png', dpi=resolution)
    

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size

    plt.figure()
    plt.title('Validation Mean Absolute Error')
    plt.ylabel('Validation MAE')
    plt.xlabel('Seconds')
    plt.autoscale(axis='x',tight=True)
    plt.plot(lstm_times, lstm[:,0], label='LSTM')
    plt.plot(tr_times, transformer[:,0], label='Transformers')
    plt.legend()
    plt.savefig('MAE2.png', dpi=resolution)