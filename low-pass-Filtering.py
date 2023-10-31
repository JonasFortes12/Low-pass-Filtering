import numpy as np
import math as mt
import matplotlib.pyplot as plt

# Função para plotagem de gráficos
def plot_graph(x, y, title, xlabel, ylabel, label='', color='blue' ):
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label=label, color=color)
    # Adicionar um título ao gráfico
    plt.title(title)
    # Adicionar rótulos aos eixos
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    
    # Adicionar uma legenda
    plt.legend()
    
    plt.show()

#__________________________ Questão 01 _____________________________________

N = 100  # Número de amostras
n = np.arange(N)  # Vetor de amostras de 0 a N-1

# Frequências desejadas em radianos por amostra
frequencies = [0.1 * mt.pi, 0.6 * mt.pi, 0.9 * mt.pi]

# Gera o sinal somando as componentes de frequência
Xn = np.sum([np.cos(omega * n) for omega in frequencies], axis=0)

plot_graph(n, Xn, 'Sinal de entrada X[n]', 'Tempo', 'Amplitude')

#__________________________ Questão 02 _____________________________________

# Calcula o módulo da transformada de Fourier do sinal X[n]
XnFT = np.abs(np.fft.fftshift(np.fft.fft(Xn)))

# Calcula as frequências correspondentes às amostras
freqs = np.fft.fftfreq(N)

plot_graph(freqs, XnFT, 'Módulo da FT do sinal X[n]', 'Frequência', 'Amplitude', color='black')



