import math
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import font
import numpy as np
import matplotlib.pyplot as plt

def gather():
    try:
        state = state_list.get()
        return [int(s.strip()) for s in state.split(',')]
    except ValueError:
        return []  # Return an empty list if parsing fails

def hermite_polynomial(n):
    if n == 0:
        return lambda x: np.ones_like(x)  # H_0(x) = 1
    elif n == 1:
        return lambda x: 2 * x  # H_1(x) = 2x
    else:
        H_n_minus_1 = hermite_polynomial(n - 1)
        H_n_minus_2 = hermite_polynomial(n - 2)
        return lambda x: 2 * x * H_n_minus_1(x) - 2 * (n - 1) * H_n_minus_2(x)

def evolve(states, x, time_steps=1000, dt=0.1):
    evolved_sum = []
    t = 0
    for _ in range(time_steps):
        W_sum = np.zeros_like(x, dtype=np.complex128)
        for n in states:
            H_n = hermite_polynomial(n)(x)
            W_n = (
                1 / np.sqrt(2 ** (n / 2) * math.factorial(n))
                * np.exp(-x ** 2)
                * H_n
                * np.exp(1j * n ** 2 * t)
            )
            W_sum += W_n
        evolved_sum.append(W_sum)
        t += dt
    return np.array(evolved_sum)

def animate_array(real_array, imag_array, amplitude, x, interval=10, save_as=None):
    if real_array.ndim != 2 or imag_array.ndim != 2:
        raise ValueError("Input arrays must be 2D.")

    time_steps, num_points = real_array.shape

    fig, ax = plt.subplots()
    real_line, = ax.plot(x, real_array[0], lw=2, label="Real Part", color='#ffea82')
    imag_line, = ax.plot(x, imag_array[0], lw=2, label="Imaginary Part", color='#e3c062')
    Amplitude_line, = ax.plot(x, amplitude[0], lw=3, label="Amplitude", color='white')

    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(np.min(imag_array), np.max(real_array))
    ax.set_title("Time Evolution of Functions")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\Psi$(x)")
    ax.legend()
    fig.patch.set_facecolor('black')  # Figure background
    ax.set_facecolor('black')  # Axes background

    # Customize ticks and labels
    ax.tick_params(colors='white')  # Tick color
    ax.spines['bottom'].set_color('white')  # Axis spine color
    ax.spines['left'].set_color('white')
    ax.xaxis.label.set_color('white')  # X-axis label color
    ax.yaxis.label.set_color('white')  # Y-axis label color

    def update(frame):
        real_line.set_ydata(real_array[frame])
        imag_line.set_ydata(imag_array[frame])
        Amplitude_line.set_ydata(amplitude[frame])
        ax.set_title(f"Time Step: {frame}")
        return real_line, imag_line, Amplitude_line

    anim = FuncAnimation(
        fig, update, frames=time_steps, interval=interval, blit=True
    )

    if save_as:
        anim.save(save_as, writer="ffmpeg")

    plt.show()

# Main program
x = np.linspace(-np.pi, np.pi, 1000)  # x values
default_state = [3, 5]  # Default quantum states
time_steps = 1000  # Number of time steps
dt = 0.01  # Time step size

root = tk.Tk()
root.title("States of the Harmonic Oscillator")
root.geometry(
              "250x200")
custom_font = ("Courier New", 12)
root.configure(bg="#292929")
text_colour = "#337d37"

label1 = tk.Label(root, text="States:", font=custom_font, fg="white", bg=text_colour)
label1.grid(row=0, column=1, padx=10, pady=10)
state_list = tk.Entry(root, width=20, font=custom_font, fg="white", bg=text_colour)
state_list.grid(row=1, column=1, padx=10, pady=10)

def simulate():
    states = gather()
    if not states:  # Handle empty input
        states = default_state
    evolved_sum = evolve(states, x, time_steps=time_steps, dt=dt)
    animate_array(
        evolved_sum.real,
        evolved_sum.imag,
        np.sqrt(evolved_sum.real ** 2 + evolved_sum.imag ** 2),
        x,
        interval=50,
    )

submit_button = tk.Button(
    root,
    text="Simulate",
    command=simulate,
    font=custom_font,
    fg="white",
    bg="#292929",
)
submit_button.grid(row=2, column=1, padx=10, pady=10)
root.mainloop()

