"""
@author: sourav
"""

"""
Module for plotting functions that I often use in my notebooks on CNN based architectures on image datasets.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_history(num_sessions, metrics=['loss', 'val_loss']):
		
		# num_sessions is the number of separate training sessions, should be <= 5
		# metrics are the quantities to be plotted, such as 'loss', 'val_loss', etc, these should be a list of strings

	for metric in metrics:
		if num_sessions == 1:
			plt.plot(history.history[metric])
		elif num_sessions == 2:
			plt.plot(history.history[metric]+history2.history[metric])
		elif num_sessions == 3:
			plt.plot(history.history[metric]+history2.history[metric]+history3.history[metric])
		elif num_sessions == 4:
			plt.plot(history.history[metric]+history2.history[metric]+history3.history[metric]+history4.history[metric])
		elif num_sessions == 5:
			plt.plot(history.history[metric]+history2.history[metric]+history3.history[metric]+history4.history[metric]+history5.history[metric])
		else:
			raise Exception("Too many separate sessions, should be less than 6.")
			
	plt.title('training history')
	plt.ylabel('metrics')
	plt.xlabel('epoch')
	
	plt.legend(metrics, loc='upper left')
	
	plt.show()



