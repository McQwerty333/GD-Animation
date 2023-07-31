import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
import tkinter as tk
from tkinter import ttk

# GD animation on simple 2 weight network, with cost function built to model various surfaces
# With momentum and other optimizers, wrapped in a class
# Up to all 10 optimizers can be displayed simultaneously
# Easy to change cost function landscape in __init__
# - With GUI to choose which optimizers to show

# np.random.seed(0)


class NetworkAnim:
	def __init__(self):
		self.reset_all()
		# x**2 + y is a good test for each axis

		self.cost_function = lambda x, y: x**2 + y
		self.cost_derivative_x = lambda x: 2*x
		self.cost_derivative_y = lambda y: 1

	def reset_all(self):
		self.i = 1
		self.vx = 0
		self.vy = 0
		self.mx = 0
		self.my = 0
		self.vhatx = 0
		self.vhaty = 0
		self.ddx = 0
		self.ddy = 0

	def trainAMSGrad(self, x, y, alpha):
		dx = self.cost_derivative_x(x)
		dy = self.cost_derivative_y(y)
		beta1 = .9
		beta2 = .999
		epsilon = 1e-7

		self.mx = beta1 * self.mx + (1 - beta1) * dx
		self.my = beta1 * self.my + (1 - beta1) * dy
		self.vx = beta2 * self.vx + (1 - beta2) * dx ** 2
		self.vy = beta2 * self.vy + (1 - beta2) * dy ** 2
		self.vhatx = max(self.vhatx, self.vx)
		self.vhaty = max(self.vhaty, self.vy)
		x -= alpha * self.mx / np.sqrt(self.vhatx + epsilon)
		y -= alpha * self.my / np.sqrt(self.vhaty + epsilon)

		return x, y

	def trainNadam(self, x, y, alpha):
		dx = self.cost_derivative_x(x)
		dy = self.cost_derivative_y(y)
		beta1 = .9
		beta2 = .999
		epsilon = 1e-7

		self.vx = beta2 * self.vx + (1 - beta2) * dx ** 2
		self.vy = beta2 * self.vy + (1 - beta2) * dy ** 2
		self.mx = beta1 * self.mx + (1 - beta1) * dx
		self.my = beta1 * self.my + (1 - beta1) * dy
		vhatx = self.vx / (1 - np.power(beta2, self.i))
		vhaty = self.vy / (1 - np.power(beta2, self.i))
		mhatx = self.mx / (1 - np.power(beta1, self.i))
		mhaty = self.my / (1 - np.power(beta1, self.i))
		x -= alpha / np.sqrt(vhatx + epsilon) * (beta1 * mhatx + (1 - beta1) / (1 - np.power(beta1, self.i)) * dx)
		y -= alpha / np.sqrt(vhaty + epsilon) * (beta1 * mhaty + (1 - beta1) / (1 - np.power(beta1, self.i)) * dy)
		self.i += 1

		return x, y

	def trainAdamax(self, x, y, alpha):
		beta1 = .9
		beta2 = .999
		dx = self.cost_derivative_x(x)
		dy = self.cost_derivative_y(y)

		self.mx = beta1 * self.mx + (1 - beta1) * dx
		self.my = beta1 * self.my + (1 - beta1) * dy
		mhatx = self.mx / (1 - np.power(beta1, self.i))
		mhaty = self.my / (1 - np.power(beta1, self.i))
		self.vx = max(self.vx * beta2, abs(dx))
		self.vy = max(self.vy * beta2, abs(dy))
		x -= alpha * mhatx / (self.vx + 1e-7)
		y -= alpha * mhaty / (self.vy + 1e-7)
		self.i += 1

		return x, y

	def trainAdam(self, x, y, alpha):
		beta1 = 0.9
		beta2 = .999
		epsilon = 1e-7
		dx = self.cost_derivative_x(x)
		dy = self.cost_derivative_y(y)

		self.mx = beta1 * self.mx + (1 - beta1) * dx
		self.vx = beta2 * self.vx + (1 - beta2) * dx ** 2
		self.my = beta1 * self.my + (1 - beta1) * dy
		self.vy = beta2 * self.vy + (1 - beta2) * dy ** 2
		mhatx = self.mx / (1 - np.power(beta1, self.i))
		vhatx = self.vx / (1 - np.power(beta2, self.i))
		mhaty = self.my / (1 - np.power(beta1, self.i))
		vhaty = self.vy / (1 - np.power(beta2, self.i))
		self.i += 1
		x -= alpha * mhatx / np.sqrt(vhatx + epsilon)
		y -= alpha * mhaty / np.sqrt(vhaty + epsilon)

		return x, y

	def trainNesterov(self, x, y, alpha):
		dx = self.cost_derivative_x(x)
		dy = self.cost_derivative_y(y)
		beta = 0.9

		self.mx = beta * self.mx + alpha * (dx - beta * self.mx)
		self.my = beta * self.my + alpha * (dy - beta * self.my)
		x -= self.mx
		y -= self.my

		return x, y

	def trainAdadelta(self, x, y, alpha):
		beta = .9
		epsilon = 1e-7
		dx = self.cost_derivative_x(x)
		dy = self.cost_derivative_y(y)

		self.vx = self.vx * beta + (1 - beta) * dx ** 2
		self.vy = self.vy * beta + (1 - beta) * dy ** 2
		delta_x = np.sqrt(self.ddx + epsilon) * dx / np.sqrt(self.vx + epsilon)
		delta_y = np.sqrt(self.ddy + epsilon) * dy / np.sqrt(self.vy + epsilon)
		self.ddx = beta * self.ddx + (1 - beta) * delta_x ** 2
		self.ddy = beta * self.ddy + (1 - beta) * delta_y ** 2
		x -= delta_x
		y -= delta_y

		return x, y

	def trainRMSProp(self, x, y, alpha):
		dx = self.cost_derivative_x(x)
		dy = self.cost_derivative_y(y)
		beta = .9
		epsilon = 1e-7

		self.vx = beta * self.vx + (1 - beta) * dx ** 2
		self.vy = beta * self.vy + (1 - beta) * dy ** 2
		x -= alpha * dx / np.sqrt(self.vx + epsilon)
		y -= alpha * dy / np.sqrt(self.vy + epsilon)

		return x, y

	def trainAdagrad(self, x, y, alpha):
		dx = self.cost_derivative_x(x)
		dy = self.cost_derivative_y(y)
		epsilon = 1e-7

		self.vx += dx ** 2
		self.vy += dy ** 2
		x -= alpha * dx / np.sqrt(self.vx + epsilon)
		y -= alpha * dy / np.sqrt(self.vy + epsilon)

		return x, y

	def trainMomentum(self, x, y, alpha):
		dx = self.cost_derivative_x(x)
		dy = self.cost_derivative_y(y)
		beta = .9

		self.mx = beta * self.mx - alpha * dx
		self.my = beta * self.my - alpha * dy
		x += self.mx
		y += self.my

		return x, y

	def trainVanilla(self, x, y, alpha):
		dx = self.cost_derivative_x(x)
		dy = self.cost_derivative_y(y)

		x -= alpha * dx
		y -= alpha * dy

		return x, y

	def cost(self, x, y):
		return self.cost_function(x, y)
		# x**2 + y provides good example for speedup of momentum
		# alpha of .001 acts normally
		# alpha of 1 bounces around
		# alpha of .9 bounces but converges in about 30 epochs on vanilla
		# with momentum, bounces around for longer but makes it a lot further (10x)
		# due to learning to travel in the - y direction regardless of x bouncing

	def loss_landscape(self):
		# costs for surface
		m1s = np.linspace(-15, 17, 40)
		m2s = np.linspace(-15, 18, 40)
		M1, M2 = np.meshgrid(m1s, m2s)
		zs_N = np.array([self.cost(np.array([[mp1]]), np.array([[mp2]]))
						 for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
		Z_N = zs_N.reshape(M1.shape)

		# plot details
		fig = plt.figure(figsize=(10, 7.5))
		ax0 = plt.axes(projection='3d')
		fontsize_ = 20
		labelsize_ = 20
		ax0.view_init(elev=30, azim=-20)
		ax0.set_xlabel(r'$x$', fontsize=fontsize_, labelpad=9)
		ax0.set_ylabel(r'$y$', fontsize=fontsize_, labelpad=-5)
		ax0.set_zlabel("costs", fontsize=fontsize_, labelpad=-30)
		ax0.tick_params(axis='x', pad=5, which='major', labelsize=labelsize_)
		ax0.tick_params(axis='y', pad=-5, which='major', labelsize=labelsize_)
		ax0.tick_params(axis='z', pad=5, which='major', labelsize=labelsize_)
		ax0.set_title('Cost = ...', y=0.85, fontsize=15)  # set title of subplot

		# plot loss landscape
		ax0.plot_surface(M1, M2, Z_N, cmap='terrain', antialiased=True, cstride=1, rstride=1, alpha=.75)
		plt.tight_layout()
		for angle in range(360):
			ax0.view_init(elev=30, azim=angle - 180)
			plt.draw()
			plt.pause(.001)

	def gd_anim(self, input_optimizers, learning_rate):
		epochs = 500

		values_x = []
		values_y = []
		costs = []

		# calculate costs for points using 2-weight GD
		start_points = [(10, 10), (8, -4), (0, 6), (-10, -10),(6, 6),
						(-5, 7), (7, 0), (12, 3), (-3, 7), (-8, 1)]
		point_training_functions = [self.trainVanilla, self.trainMomentum, self.trainAdagrad,
									self.trainRMSProp, self.trainAdadelta, self.trainNesterov,
									self.trainAdam, self.trainAdamax, self.trainNadam, self.trainAMSGrad]
		input_nonzero = np.nonzero(input_optimizers)[0]
		num = len(input_nonzero)

		for i, point in enumerate(start_points[:num]):
			x, y = point
			self.reset_all()
			for j in range(epochs):
				# x, y = self.trainAMSGrad(x, y, .9)
				x, y = point_training_functions[input_nonzero[i]](x, y, learning_rate)
				# train_anim - update x and y values according to partial derivatives
				values_x.append(x)
				values_y.append(y)
				costs.append(self.cost(x, y))

		costs = np.split(np.array(costs), num)
		values_x = np.split(np.array(values_x), num)
		values_y = np.split(np.array(values_y), num)

		# define epochs to plot
		p1 = list(np.arange(0, epochs//10, 1))
		p2 = list(np.arange(epochs//10, epochs, 10))
		points = p1 + p2

		# costs for surface
		m1s = np.linspace(-15, 17, 40)
		m2s = np.linspace(np.min(values_y[:num]), np.max(values_y[:num]), 40)
		M1, M2 = np.meshgrid(m1s, m2s)
		zs_N = np.array([self.cost(np.array([[mp1]]), np.array([[mp2]]))
						 for mp1, mp2 in zip(np.ravel(M1), np.ravel(M2))])
		Z_N = zs_N.reshape(M1.shape)

		# plot details
		fig = plt.figure(figsize=(10, 7.5))
		ax0 = plt.axes(projection='3d')
		fontsize_ = 20
		labelsize_ = 20
		line_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
					   'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
					   'tab:olive', 'tab:cyan']
		ax0.view_init(elev=30, azim=-80)
		ax0.set_xlabel(r'$x$', fontsize=fontsize_, labelpad=9)
		ax0.set_ylabel(r'$y$', fontsize=fontsize_, labelpad=-5)
		ax0.set_zlabel("costs", fontsize=fontsize_, labelpad=-30)
		ax0.tick_params(axis='x', pad=5, which='major', labelsize=labelsize_)
		ax0.tick_params(axis='y', pad=-5, which='major', labelsize=labelsize_)
		ax0.tick_params(axis='z', pad=5, which='major', labelsize=labelsize_)
		ax0.set_title('2 \'Weight\' GD', y=0.85, fontsize=15)  # set title of subplot

		camera = Camera(fig)
		for p in points:
			# plot points engaging in descent
			for i in range(num):
				# lines
				ax0.plot(values_x[i][:p], values_y[i][:p], costs[i][:p],
						 linestyle='dashdot', linewidth=2, color=line_colors[input_nonzero[i]])
				# points
				ax0.scatter(values_x[i][p], values_y[i][p], costs[i][p],
							marker='o', s=15 ** 2, color=line_colors[input_nonzero[i]], alpha=1.0)

			# plot surface of loss landscape
			ax0.plot_surface(M1, M2, Z_N, cmap='terrain',
							 antialiased=True, cstride=1, rstride=1, alpha=.5)
			ax0.legend([f'epochs: {p}'], loc=(.25, .8), fontsize=17)
			plt.tight_layout()
			camera.snap()
		anim = camera.animate(interval=5, repeat=False, repeat_delay=0)
		plt.show()


class Applet:
	def __init__(self):
		window = tk.Tk()
		window.geometry("300x300")
		window.title("Gradient Descent Optimizers")
		window.resizable(False, False)
		window.columnconfigure(0)
		window.columnconfigure(1)
		window.columnconfigure(2)
		self.value1 = tk.IntVar()
		self.checkbox_variables = []
		for i in range(10):
			self.checkbox_variables.append(tk.IntVar())
			self.checkbox_variables[i].set(0)

		label1 = ttk.Label(window, text="Select Optimizers to Display:")
		label1.grid(column=0, row=0, pady=3, sticky=tk.E)
		labels = [ttk.Label(window, text="Vanilla (No optimizer)"),
				  ttk.Label(window, text="Momentum"),
				  ttk.Label(window, text="AdaGrad - Adaptive Gradient"),
				  ttk.Label(window, text="RMSProp - Root Mean Square Propagation"),
				  ttk.Label(window, text="AdaDelta - Adaptive Delta"),
				  ttk.Label(window, text="Nesterov Momentum"),
				  ttk.Label(window, text="Adam - Adaptive Moment Estimation"),
				  ttk.Label(window, text="AdaMax - Adam Extension"),
				  ttk.Label(window, text="Nadam - Nesterov Accelerated Adam"),
				  ttk.Label(window, text="AMSGrad - Adam Extension")]

		colors = [ttk.Label(window, background='#1f77b4', text='      '),
				  ttk.Label(window, background='#ff7f0e', text='      '),
				  ttk.Label(window, background='#2ca02c', text='      '),
				  ttk.Label(window, background='#d62728', text='      '),
				  ttk.Label(window, background='#9467bd', text='      '),
				  ttk.Label(window, background='#8c564b', text='      '),
				  ttk.Label(window, background='#e377c2', text='      '),
				  ttk.Label(window, background='#7f7f7f', text='      '),
				  ttk.Label(window, background='#bcbd22', text='      '),
				  ttk.Label(window, background='#17becf', text='      ')]

		for i, label in enumerate(labels):
			label.grid(column=0, row=i + 1, sticky=tk.E)
			checkbox = ttk.Checkbutton(window, variable=self.checkbox_variables[i])
			checkbox.grid(column=1, row=i + 1)
			colors[i].grid(column=2, row=i+1)
		button = ttk.Button(window, text='Go', command=self.handle_button)
		button.grid(column=0, row=i + 2, pady=5)
		window.mainloop()

	def handle_button(self):
		network_01 = NetworkAnim()
		optimizers = []
		for check_var in self.checkbox_variables:
			optimizers.append(int(check_var.get()))
		network_01.gd_anim(optimizers, .1)


if __name__ == '__main__':
	main1 = Applet()


